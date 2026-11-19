"""Measure native 256K KV allocation, natural long generation and reasoning across tools."""
import gzip
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path

from campaign_report import atomic_json
from qah.executor import ReferenceExecutor
from qah.telemetry import Monitor, gpu_sample, host_inventory
from swift_homelab import gpu_resident
from swift_qwen import MIB, SAMPLING, SHA256, check_placement, template_probe


CAPACITY, INPUT = 262144, 131072
OUTPUT = CAPACITY-INPUT-256
SENTINEL = 'HEADROOM_ARCHIVE_651732'
KEYS = dict(A='fern-bd731a', B='cobalt-4d89f2', C='sable-9f132e')
LONG_TASK = (
    'Implement a production-style Python log-processing CLI in a single self-contained file. '
    'Include streaming JSONL and CSV readers, strict schema validation, UTC timestamp normalization, '
    'inclusive start/exclusive end date filters, aggregation by module and level, deterministic JSON output, '
    'structured malformed-record diagnostics, argparse options, a main entry point and runnable unittest '
    'tests for each behavior. Explain choices, then provide the complete implementation and tests, '
    'including examples. A complete answer should contain approximately 200 to 350 lines of code. '
    'Use only the standard library. The three active archive keys below must be default constants A, B and C '
    'in the implementation. Finish naturally when all requested code and tests are included.'
)
TOOLS = [
    dict(type='function', function=dict(name='archive_lookup', description='Retrieve the signed record for one archive key.',
         parameters=dict(type='object', properties=dict(key=dict(type='string', enum=['A', 'B', 'C'])), required=['key']))),
    dict(type='function', function=dict(name='verify_digest', description='Verify the retrieved record digest.',
         parameters=dict(type='object', properties=dict(digest=dict(type='string')), required=['digest']))),
]
TOOL_TASK = (
    'First call archive_lookup for key B, using the original archive to explain your choice in thinking. '
    'When it returns a digest, call verify_digest with that digest. After verification, return one JSON object '
    'with the original active archive keys A, B and C and a boolean verified field. '
    'Do not include a Markdown fence. Preserve your reasoning across the tool results.'
)


def prefix(executor):
    chunk = ''.join(f'Historical ticket {i:04d}: module=codec-{i % 31}; status closed; reviewed=true.\n'
                    for i in range(512))
    # Build text once. Final length is determined with the actual runtime tokenizer and template.
    return (chunk*32)


def render(executor, messages, tools=None, preserve=True):
    payload = dict(messages=messages, chat_template_kwargs=dict(enable_thinking=True,
                   reasoning_effort='xhigh', preserve_thinking=preserve))
    if tools:
        payload['tools'] = tools
    return executor.json('/apply-template', payload)['prompt']


def make_archive(executor):
    block = prefix(executor)
    segments = []
    for key, value in KEYS.items():
        segments.append(block[:45000] + f'\nSIGNED ACTIVE record: key {key} = {value}.\n')
    return '\n'.join(segments)


def messages_for(executor, task, target):
    archive = make_archive(executor)
    filler = prefix(executor)
    content = 'Read the synthetic archive below.\n' + archive + '\n' + SENTINEL + '\n' + task
    messages = [dict(role='user', content=content.replace(SENTINEL, ''))]
    base = len(executor.tokenize(render(executor, messages)))
    if base >= target:
        raise RuntimeError('initial archive exceeds prompt target')
    padding_ids = executor.tokenize(filler)
    needed = target-base
    padding_ids = (padding_ids*((needed+len(padding_ids)-1)//len(padding_ids)))[:needed]
    padding = executor.json('/detokenize', dict(tokens=padding_ids))['content']
    messages[0]['content'] = content.replace(SENTINEL, padding)
    for _ in range(8):
        prompt = render(executor, messages)
        ids = executor.tokenize(prompt)
        difference = target-len(ids)
        if not difference:
            return messages, ids
        # Adjust tokenized padding to compensate boundary token merges.
        extra = executor.tokenize(filler)
        if difference > 0:
            padding_ids.extend(extra[:difference])
        else:
            padding_ids = padding_ids[:difference]
        padding = executor.json('/detokenize', dict(tokens=padding_ids))['content']
        messages[0]['content'] = content.replace(SENTINEL, padding)
    raise RuntimeError('could not produce an exact 128K rendered prompt')


def native(executor, tokens, output_limit, cached, path):
    from swift_qwen import complete
    with Monitor(executor, output=path.with_suffix('.telemetry.jsonl')) as monitor:
        result = complete(executor, tokens, output_limit, cached)
    result['resources'] = monitor.report()
    result['large_output_allowance_accepted'] = True
    result['natural_stop'] = result['stop_type'] != 'limit' and not result['truncated']
    atomic_json(path, result)
    t = result['timings']
    if t['prompt_n']+t['cache_n'] != len(tokens) or bool(t['cache_n']) != cached or result['truncated']:
        raise RuntimeError('native request failed context/cache validation')
    print(path.stem, dict(input=len(tokens), output=result['output_tokens'], limit=output_limit,
                         ttft=result['ttft_seconds'], natural_stop=result['natural_stop'], timings=t), flush=True)
    return result


def chat(executor, payload, path):
    req = urllib.request.Request(executor.url+'/v1/chat/completions',
                                 json.dumps(dict(payload, stream=True, stream_options=dict(include_usage=True))).encode(),
                                 {'Content-Type': 'application/json'})
    started, first, reason = time.perf_counter(), None, None
    message = dict(role='assistant', content='', reasoning_content='')
    calls, usage, timing = {}, None, None
    with Monitor(executor, output=path.with_suffix('.telemetry.jsonl')) as monitor:
        with urllib.request.urlopen(req, timeout=1800) as response:
            for line in response:
                if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                    continue
                event = json.loads(line[6:])
                if event.get('error'):
                    raise RuntimeError(str(event['error']))
                if event.get('usage'):
                    usage = event['usage']
                if event.get('timings'):
                    timing = event['timings']
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    if first is None and any(delta.get(k) for k in ('content', 'reasoning_content', 'tool_calls')):
                        first = time.perf_counter()-started
                    for key in ('content', 'reasoning_content'):
                        message[key] += delta.get(key) or ''
                    for call in delta.get('tool_calls', []):
                        item = calls.setdefault(call['index'], dict(id='', type='function', function=dict(name='', arguments='')))
                        if call.get('id'):
                            item['id'] = call['id']
                        for key in ('name', 'arguments'):
                            item['function'][key] += call.get('function', {}).get(key) or ''
                    reason = choice.get('finish_reason') or reason
        resources = monitor.report()
    if calls:
        message['tool_calls'] = [calls[i] for i in sorted(calls)]
    result = dict(message=message, finish_reason=reason, usage=usage, timings=timing,
                  ttft_seconds=first, wall_seconds=time.perf_counter()-started, resources=resources)
    atomic_json(path, dict(request=payload, **result))
    return result


def run(args):
    path = args.output / 'headroom'
    path.mkdir()
    executor = ReferenceExecutor(args.server, args.model, CAPACITY, path/'engine.log',
                                 key_type='q4_0', value_type='q4_0', mtp_depth=3,
                                 ubatch=128, batch=512, draft_ubatch=64,
                                 prefix_cache=True, cache_ram=0, checkpoints=4)
    executor.env = {k: v for k, v in os.environ.items() if not k.upper().startswith(
        ('LLAMA_ARG_', 'GGML_', 'CUDA_VISIBLE_DEVICES'))}
    executor.argv += ['--alias', 'swift-headroom', '--reasoning', 'on', '--reasoning-effort', 'xhigh',
                      '--reasoning-budget', '-1', '--reasoning-format', 'deepseek', '--lazy-mode', 'off',
                      '--override-tensor', '^token_embd\\.weight$=CUDA0',
                      '--spec-draft-override-tensor', '^token_embd\\.weight$=CUDA0', '--no-warmup']
    for flag in ('--spec-draft-type-k', '--spec-draft-type-v'):
        executor.argv[executor.argv.index(flag)+1] = 'q4_0'
    atomic_json(path/'config.json', dict(argv=executor.argv, sha256=SHA256, input=INPUT, output_allowance=OUTPUT,
                                       explicit_reasoning_cap=False, sampling=SAMPLING))
    summary = dict(capacity=CAPACITY, target_input=INPUT, total_output_allowance=OUTPUT, memory={})
    try:
        with executor:
            placement = summary['placement'] = check_placement(executor.log_path, CAPACITY)
            summary['memory']['startup'] = gpu_resident(executor, placement)
            template_probe(executor, path)
            # Observe reservation before and after a short real request.
            short_ids = executor.tokenize(render(executor, [dict(role='user', content='What is 17 * 19?')]))
            native(executor, short_ids, 16384, False, path/'short.json')
            summary['memory']['after_short'] = gpu_resident(executor, placement)
            messages, ids = messages_for(executor, LONG_TASK, INPUT)
            atomic_json(path/'long-messages.json', messages)
            with gzip.open(path/'long-input-ids.json.gz', 'wt') as stream:
                json.dump(ids, stream)
            summary['input_sha256'] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            summary['memory']['after_tokenization'] = gpu_resident(executor, placement)
            cold = native(executor, ids, OUTPUT, False, path/'long-cold.json')
            summary['cold'] = {k: cold[k] for k in ('actual_input_tokens', 'output_limit', 'output_tokens',
                              'ttft_seconds', 'wall_seconds', 'timings', 'natural_stop', 'thinking_closed', 'resources')}
            summary['longer_than_512'] = cold['output_tokens'] > 512
            summary['memory']['after_long'] = gpu_resident(executor, placement)
            # An exact replay proves that the large allowance also works with a cached prefix.
            cached = native(executor, ids, OUTPUT, True, path/'long-cached.json')
            summary['cached'] = {k: cached[k] for k in summary['cold']}
            summary['memory']['after_cached'] = gpu_resident(executor, placement)
            tools_messages = [dict(role='user', content=messages[0]['content'].replace(LONG_TASK, TOOL_TASK))]
            tools_result = []
            for turn in range(3):
                rendered = render(executor, tools_messages, TOOLS)
                before = executor.tokenize(rendered)
                previous = [m['reasoning_content'] for m in tools_messages
                            if m.get('role') == 'assistant' and m.get('reasoning_content')]
                retained = [reasoning in rendered for reasoning in previous]
                without = render(executor, tools_messages, TOOLS, preserve=False)
                atomic_json(path/f'tool-{turn}-render.json', dict(input_tokens=len(before),
                            prior_reasoning_exactly_retained=retained,
                            prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                            preserve_false_changes_prompt=without != rendered))
                if previous and not all(retained):
                    raise RuntimeError('earlier reasoning was dropped from the tool continuation prompt')
                limit = CAPACITY-len(before)-256
                if limit <= 0:
                    raise RuntimeError('tool continuation has insufficient output headroom')
                payload = dict(SAMPLING, model='swift-headroom', messages=tools_messages, tools=TOOLS,
                               max_tokens=limit, reasoning_effort='xhigh', cache_prompt=True,
                               chat_template_kwargs=dict(enable_thinking=True, reasoning_effort='xhigh',
                                                         preserve_thinking=True))
                response = chat(executor, payload, path/f'tool-{turn}.json')
                assistant = response['message']
                tools_messages.append(assistant)
                calls = assistant.get('tool_calls') or []
                actual = (response.get('usage') or {}).get('prompt_tokens')
                if actual and actual+limit > CAPACITY:
                    raise RuntimeError('chat input plus output allowance exceeds allocation')
                record = dict(turn=turn, rendered_input_tokens=len(before), actual_input_tokens=actual,
                              max_tokens=limit, nonempty_reasoning=bool(assistant['reasoning_content'].strip()),
                              retained_reasoning=retained, finish_reason=response['finish_reason'],
                              tool_calls=calls, usage=response['usage'], timings=response['timings'])
                tools_result.append(record)
                if not calls:
                    try:
                        final = json.loads(assistant['content'])
                    except ValueError:
                        final = None
                    summary['tool_final_correct'] = final == dict(KEYS, verified=True)
                    break
                for call in calls:
                    arguments = json.loads(call['function']['arguments'])
                    name = call['function']['name']
                    if name == 'archive_lookup' and arguments == dict(key='B') and turn == 0:
                        value = dict(key='B', value=KEYS['B'], digest='sha256:86c3f1724b59')
                    elif name == 'verify_digest' and arguments == dict(digest='sha256:86c3f1724b59') and turn == 1:
                        value = dict(verified=True)
                    else:
                        raise RuntimeError(f'unexpected tool call {name} with {arguments}')
                    tools_messages.append(dict(role='tool', tool_call_id=call['id'], content=json.dumps(value)))
                print('tool',turn,'calls',[call['function']['name'] for call in calls],
                      'reasoning',bool(assistant['reasoning_content']),'retained',retained,flush=True)
            summary['tools'] = tools_result
            summary['reasoning_across_tools'] = (len(tools_result)==3 and
                all(turn['nonempty_reasoning'] for turn in tools_result) and
                all(all(turn['retained_reasoning']) for turn in tools_result))
            summary['memory']['after_tools'] = gpu_resident(executor, placement)
    finally:
        summary['server_stopped'] = executor.process is None or executor.process.poll() is not None
        atomic_json(path/'summary.json', summary)
