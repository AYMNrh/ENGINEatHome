"""Measure Swift's all-GPU context and MTP profiles with its embedded chat template."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from campaign_report import atomic_json
from qah.executor import ReferenceExecutor
from qah.memory import estimate
from qah.model import architecture, inspect_gguf
from qah.telemetry import Monitor, gpu_sample, host_inventory
from run import allocations, hardware_inventory


REVISION = 'd74895bbe5db4bec1e0024e7cc87d59c02d7631a'
SHA256 = '9aecf1cd41b2cb2f32a74e0d889e33855ebef43b26f43b43feb5720239e677e5'
FILENAME = 'Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf'
REPOSITORY = 'https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF'
CONTEXTS = (65536, 92160, 131072, 196608, 262144)
SAMPLING = dict(temperature=1.0, top_p=.95, top_k=20, min_p=0.0,
                presence_penalty=0.0, repeat_penalty=1.0, seed=42)
MESSAGES = [dict(role='user', content='Compute 17 * 19. Explain briefly and give the result.')]
MARKER = 'rbk-e1b7c92a'
SENTINEL = 'SWIFT_QAH_INSERT_ARCHIVE_9453'
MIB = 2**20


def check_placement(log, context):
    text = log.read_text(encoding='utf-8', errors='replace')
    buffers = allocations(log)
    layers = re.findall(r'offloaded (\d+)/(\d+) layers to GPU', text)
    if (not layers or any(loaded != total for loaded, total in layers) or
            not all(buffers.get(f'CUDA0_{key}_bytes') for key in ('model', 'KV', 'RS', 'compute')) or
            any(value for key, value in buffers.items()
                if key.startswith(('CPU', 'CUDA_Host')) and key.endswith(('_model_bytes', '_KV_bytes', '_RS_bytes'))) or
            not re.search(rf'n_ctx_slot = {context}\b', text)):
        raise RuntimeError('full GPU layer/cache residency and exact slot context could not be verified')
    return dict(layers=int(layers[0][0]), buffers=buffers)


def wddm_sample(pid):
    command = (
        "(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage',"
        "'\\GPU Process Memory(*)\\Shared Usage').CounterSamples | "
        f"Where-Object {{ $_.InstanceName -like 'pid_{pid}_*' -and $_.Status -in 0,1 }} | "
        "Select-Object Path,CookedValue | ConvertTo-Json -AsArray -Compress"
    )
    data = subprocess.check_output(['pwsh', '-NoProfile', '-Command', command], text=True,
                                   timeout=12, creationflags=subprocess.CREATE_NO_WINDOW)
    totals = {}
    for row in json.loads(data):
        key = row['Path'].rsplit('\\', 1)[-1]
        totals[key] = totals.get(key, 0) + round(row['CookedValue'])
    return totals


def template_probe(executor, output):
    def render(effort, thinking=True, kwargs=False):
        payload = dict(messages=MESSAGES, chat_template_kwargs=dict(enable_thinking=thinking))
        if kwargs:
            payload['chat_template_kwargs']['reasoning_effort'] = effort
        elif effort:
            payload['reasoning_effort'] = effort
        return executor.json('/apply-template', payload)['prompt']

    from urllib.error import HTTPError
    results = {}
    for name, effort, thinking, kwargs in (
        ('default', None, True, False), ('top_xhigh', 'xhigh', True, False),
        ('top_low', 'low', True, False), ('kwargs_xhigh', 'xhigh', True, True),
        ('kwargs_low', 'low', True, True), ('thinking_off', None, False, False),
    ):
        try:
            results[name] = render(effort, thinking, kwargs)
        except HTTPError as error:
            results[name] = dict(error=error.read().decode(errors='replace'), status=error.code)
    atomic_json(output / 'rendered-prompts.json', results)
    for mode in ('kwargs', 'top'):
        high, low = results[mode + '_xhigh'], results[mode + '_low']
        if (isinstance(high, str) and isinstance(low, str) and
                'Reasoning effort is set to xhigh.' in high and
                'Reasoning effort is set to low.' in low and
                high.rstrip().endswith('<think>') and low.rstrip().endswith('<think>')):
            return mode
    raise RuntimeError('xhigh and low controls did not render distinct reasoning instructions; see rendered-prompts.json')


def render_tokens(executor, mode, count):
    if count is None:
        messages = MESSAGES
    else:
        messages = [dict(role='user', content=(
            'Read the synthetic repository archive below.\n' + SENTINEL +
            '\nWhat is the exact rollback key in the active release record? '
            'Return the key by itself in the final answer.'))]
    payload = dict(messages=messages, chat_template_kwargs=dict(enable_thinking=True))
    target = payload['chat_template_kwargs'] if mode == 'kwargs' else payload
    target['reasoning_effort'] = 'xhigh'
    rendered = executor.json('/apply-template', payload)['prompt']
    if 'Reasoning effort is set to xhigh.' not in rendered or not rendered.rstrip().endswith('<think>'):
        raise RuntimeError('live benchmark prompt did not render the requested xhigh thinking')
    if count is None:
        return executor.tokenize(rendered)
    left, right = rendered.split(SENTINEL)
    prefix, suffix = executor.tokenize(left), executor.tokenize(right)
    needle = executor.tokenize('\nActive release record. Exact rollback key: ' + MARKER + '\n')
    block = executor.tokenize(''.join(
        f'Historical ticket {i:04d}: module=codec-{i % 31}; unit tests passed; status closed.\n'
        for i in range(512)))
    needed = count - len(prefix) - len(suffix) - len(needle)
    filler = (block * ((needed + len(block) - 1) // len(block)))[:needed]
    middle = needed * 4 // 5
    return prefix + filler[:middle] + needle + filler[middle:] + suffix


def complete(executor, tokens, limit, cached, force_length=False):
    payload = dict(SAMPLING, prompt=tokens, n_predict=limit, cache_prompt=cached,
                   stream=True, return_tokens=True, id_slot=0, ignore_eos=force_length)
    req = urllib.request.Request(executor.url + '/completion', json.dumps(payload).encode(),
                                 {'Content-Type': 'application/json'})
    start, ttft, final = time.perf_counter(), None, None
    chunks, ids = [], []
    with urllib.request.urlopen(req, timeout=executor.request_timeout) as response:
        for line in response:
            if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                continue
            event = json.loads(line[6:])
            if 'error' in event:
                raise RuntimeError(str(event['error']))
            piece, returned = event.get('content', ''), event.get('tokens') or []
            if ttft is None and (piece or returned):
                ttft = time.perf_counter() - start
            chunks.append(piece)
            ids.extend(returned)
            if event.get('stop'):
                final = event
    if final is None:
        raise RuntimeError('completion stream ended without terminal event')
    raw = ''.join(chunks)
    thinking, separator, answer = raw.partition('</think>')
    answer = answer.split('<|im_end|>', 1)[0]
    return dict(actual_input_tokens=len(tokens), input_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                output_limit=limit, cached_requested=cached, ttft_seconds=ttft,
                wall_seconds=time.perf_counter() - start, timings=final['timings'],
                backend_input_tokens=final.get('tokens_evaluated'), output_tokens=final.get('tokens_predicted'),
                output_token_ids=ids, truncated=final.get('truncated'), stop_type=final.get('stop_type'),
                output=raw, final_answer=answer.strip() if separator else None,
                thinking_closed=bool(separator), thinking_text_nonempty=bool(thinking.strip()) and bool(separator))


def measure(executor, directory, name, count, tokens, limit, cached=False):
    samples, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            try:
                samples.append(dict(unix=time.time(), **wddm_sample(executor.process.pid)))
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                samples.append(dict(unix=time.time(), error=str(error)))
            stop.wait(2)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        with Monitor(executor, output=directory / f'{name}-telemetry.jsonl') as monitor:
            result = complete(executor, tokens, limit, cached)
        result['resources'] = monitor.report()
    finally:
        stop.set()
        thread.join(timeout=14)
        atomic_json(directory / f'{name}-wddm.json', samples)
    wddm = [row for row in samples if 'shared usage' in row and 'dedicated usage' in row]
    result['wddm_shared_peak_bytes'] = max((row['shared usage'] for row in wddm), default=None)
    result['wddm_dedicated_peak_bytes'] = max((row['dedicated usage'] for row in wddm), default=None)
    result['retrieval_pass'] = result['final_answer'] == MARKER if count is not None else None
    atomic_json(directory / f'{name}.json', result)
    timing = result['timings']
    if (result['truncated'] or timing['prompt_n'] + timing.get('cache_n', 0) != len(tokens) or
            not result['output_tokens'] or result['ttft_seconds'] is None or
            (cached and not timing.get('cache_n')) or (not cached and timing.get('cache_n'))):
        raise RuntimeError(f'{name}: failed exact context, generation, or cache validation')
    if not wddm or result['wddm_shared_peak_bytes'] > 512 * MIB:
        raise RuntimeError(f'{name}: sampled WDDM residency check failed')
    print(name, dict(input=len(tokens), cap=limit, output=result['output_tokens'],
                     ttft=round(result['ttft_seconds'], 3), timing=timing,
                     retrieval=result['retrieval_pass']), flush=True)
    return result


def profile(args, context, kv, depth, predicted):
    name = f'{context}-{kv}-mtp{depth}'
    directory = args.output / name
    directory.mkdir()
    row = dict(context=context, kv=kv, mtp_depth=depth, estimate=predicted,
               batch=args.batch, microbatch=args.microbatch, requests={})
    executor = ReferenceExecutor(args.server, args.model, context, directory / 'engine.log',
                                 key_type=kv, value_type=kv, batch=args.batch, ubatch=args.microbatch,
                                 mtp_depth=depth, draft_ubatch=min(64, args.microbatch), prefix_cache=True,
                                 cache_ram=0, checkpoints=8)
    executor.env = {key: value for key, value in os.environ.items()
                    if not key.upper().startswith(('LLAMA_ARG_', 'GGML_', 'CUDA_VISIBLE_DEVICES'))}
    executor.argv += ['--reasoning', 'on', '--reasoning-effort', 'xhigh', '--reasoning-budget', '-1',
                      '--lazy-mode', 'off', '--override-tensor', '^token_embd\\.weight$=CUDA0', '--no-warmup']
    if depth:
        executor.argv += ['--spec-draft-override-tensor', '^token_embd\\.weight$=CUDA0']
        for flag in ('--spec-draft-type-k', '--spec-draft-type-v'):
            executor.argv[executor.argv.index(flag) + 1] = args.draft_kv
    row['argv'] = executor.argv
    atomic_json(args.output / 'status.json', dict(status='running', profile=name))
    try:
        with executor:
            row['placement'] = check_placement(executor.log_path, context)
            start = row['wddm_start'] = wddm_sample(executor.process.pid)
            buffers = row['placement']['buffers']
            minimum = sum(buffers[f'CUDA0_{key}_bytes'] for key in ('model', 'KV', 'RS'))
            gpu = row['gpu_start'] = gpu_sample()
            if (start.get('dedicated usage', 0) < minimum or start.get('shared usage', float('inf')) > 512 * MIB
                    or not gpu or gpu['free'] < 512 * MIB):
                free = gpu['free'] // MIB if gpu else 'unavailable'
                dedicated = start.get('dedicated usage', 0) // MIB
                shared = start.get('shared usage', 0) // MIB
                raise RuntimeError(f'stopped before inference: {free} MiB device free (512 required), '
                                   f'{dedicated} MiB process dedicated ({minimum//MIB} required), '
                                   f'{shared} MiB process shared (512 maximum)')
            mode = row['template_parameter'] = template_probe(executor, directory)
            atomic_json(directory / 'startup.json', row)
            if context == CONTEXTS[0] and kv == 'q4_0' and not depth and not args.tight_vram:
                smoke = measure(executor, directory, 'thinking', None,
                                render_tokens(executor, mode, None), 16384)
                row['requests']['thinking'] = smoke
                if not smoke['thinking_text_nonempty'] or '323' not in (smoke['final_answer'] or ''):
                    raise RuntimeError('xhigh thinking generation did not close and answer the smoke correctly')
            count = context - 1024
            tokens = render_tokens(executor, mode, count)
            for cached in (False, True):
                mode_label = 'cached' if cached else 'cold'
                row['requests'][mode_label] = measure(executor, directory, mode_label, count, tokens, 512,
                                                     cached=cached)
            if depth and not row['requests']['cold']['timings'].get('draft_n'):
                raise RuntimeError('MTP produced no measured drafts')
            if context == CONTEXTS[0] and kv == 'q4_0' and not depth and not args.tight_vram:
                for limit in (16384, 32768):
                    count = context - limit - 256
                    name = f'output-cap-{limit}'
                    row['requests'][name] = measure(executor, directory, name, count,
                                                    render_tokens(executor, mode, count), limit)
        row['status'] = 'measured'
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        row.update(status='failed', error=str(error))
    finally:
        row['server_stopped'] = executor.process is None or executor.process.poll() is not None
        atomic_json(directory / 'result.json', row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('F:/MODELS/Swift-1.5-Qwen3.8-27B') / FILENAME)
    parser.add_argument('--server', type=Path, default=Path('F:/llama.cpp/b12110-cuda13/llama-server.exe'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--continue-baseline', type=Path, help='measured baseline result.json from a prior invocation')
    parser.add_argument('--tight-vram', action='store_true', help='64K matched MTP and 128K Q4 retry with a smaller microbatch')
    parser.add_argument('--smaller-mtp-context', action='store_true',
                        help='compare matched 32K target/MTP profiles when 64K cannot fit')
    parser.add_argument('--batch', type=int, choices=(128, 512), default=512)
    parser.add_argument('--microbatch', type=int, choices=(32, 64, 128), default=128)
    parser.add_argument('--draft-kv', choices=('q4_0', 'q8_0'), default='q8_0')
    args = parser.parse_args()
    if args.microbatch > args.batch:
        parser.error('--microbatch cannot exceed --batch')
    args.output.mkdir(parents=True)
    atomic_json(args.output / 'status.json', dict(status='verifying_checkpoint'))
    model = inspect_gguf(args.model, checksum=True)
    if model['sha256'] != SHA256 or model['file_bytes'] != 12120016896:
        raise ValueError('checkpoint size or SHA256 differs from the pinned published MTP file')
    gpu = gpu_sample()
    if not gpu:
        raise RuntimeError('NVIDIA device telemetry is unavailable')
    metadata = model['metadata']
    template = metadata['tokenizer.chat_template']
    (args.output / 'embedded-chat-template.jinja').write_text(template, encoding='utf-8')
    if any(s not in template for s in ('xhigh', 'reasoning_effort', 'enable_thinking')):
        raise RuntimeError('embedded template does not advertise xhigh and thinking controls')
    plans = [(context, kv, 0) for context in CONTEXTS for kv in ('q4_0', 'q8_0')]
    plans += [(CONTEXTS[0], 'q4_0', depth) for depth in (3, 4, 5)]
    if args.tight_vram:
        if args.continue_baseline or args.baseline_only or args.microbatch == 128:
            parser.error('--tight-vram needs --microbatch 32 or 64 and its own output directory')
        plans = [(65536, 'q4_0', depth) for depth in (0, 3, 4, 5)] + [(131072, 'q4_0', 0)]
    if args.smaller_mtp_context:
        if args.continue_baseline or args.baseline_only or args.tight_vram:
            parser.error('--smaller-mtp-context is a separate matched sweep')
        plans = [(32768, 'q4_0', depth) for depth in (0, 3, 4, 5)]
    estimates = {(context, kv, depth): estimate(model, context, kv, kv, mtp=bool(depth),
                 reserve_bytes=512 * MIB, gpu_budget_bytes=gpu['free']) for context, kv, depth in plans}
    atomic_json(args.output / 'provenance.json', dict(
        revision=REVISION, checkpoint=REPOSITORY + '/resolve/' + REVISION + '/' + FILENAME,
        sha256=SHA256, file_bytes=model['file_bytes'], architecture=architecture(model),
        sampling=SAMPLING, runtime=subprocess.run([str(args.server), '--version'],
            text=True, capture_output=True, check=True).stderr.strip(),
        hardware=hardware_inventory(), prelaunch_gpu=gpu, host=host_inventory(),
        estimates=[dict(mtp_depth=key[2], **value) for key, value in estimates.items()]))
    results = []
    if args.continue_baseline:
        baseline = json.loads(args.continue_baseline.read_text(encoding='utf-8'))
        if (baseline.get('context'), baseline.get('kv'), baseline.get('mtp_depth'), baseline.get('status')) != (
                65536, 'q4_0', 0, 'measured') or not baseline.get('server_stopped') or (
                '--override-tensor' not in baseline.get('argv', [])):
            raise ValueError('a measured, stopped 64K Q4 non-MTP baseline is required')
        results.append(baseline)
        plans = plans[1:]
    elif args.baseline_only:
        plans = plans[:1]
    for key in plans:
        predicted = estimates[key]
        if predicted['status'] in ('unsupported_positional_extension', 'refused_budget') and key != (65536, 'q4_0', 0):
            row = dict(context=key[0], kv=key[1], mtp_depth=key[2], status='not_started_estimate',
                       estimate=predicted)
        else:
            row = profile(args, *key, predicted)
        results.append(row)
        atomic_json(args.output / 'results.json', results)
        if row.get('server_stopped') is False or (len(results) == 1 and row['status'] != 'measured'):
            break
    atomic_json(args.output / 'status.json', dict(status='complete' if results[0]['status'] == 'measured'
                                                else 'baseline_failed', profiles=len(results)))


if __name__ == '__main__':
    main()
