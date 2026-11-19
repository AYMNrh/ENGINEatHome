"""Run the native CUDA Swift matrix and restore the live server configuration."""
import argparse
import gzip
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from campaign_report import atomic_json
from qah.executor import ReferenceExecutor
from qah.memory import estimate
from qah.model import inspect_gguf
from qah.telemetry import Monitor, gpu_sample, host_inventory
from run import hardware_inventory
from swift_qwen import FILENAME, MARKER, MIB, SAMPLING, SHA256, check_placement, complete, render_tokens, template_probe


def gpu_resident(executor, placement):
    counters = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                        '--format=csv,noheader,nounits'], text=True, timeout=10)
    pid = executor.process.pid
    values = [int(memory.strip()) * MIB for line in counters.splitlines() if ',' in line
              for owner, memory in [line.split(',', 1)] if owner.strip() == str(pid)]
    minimum = sum(placement['buffers'][f'CUDA0_{key}_bytes'] for key in ('model', 'RS', 'KV'))
    gpu = gpu_sample()
    if not values or sum(values) < minimum or not gpu or gpu['free'] < 512 * MIB:
        raise RuntimeError(f'dedicated-memory guard: PID {pid}, used bytes {values}, required {minimum}, GPU {gpu}')
    return dict(pid=pid, process_device_bytes=sum(values), shared_gpu_memory_bytes=None, device=gpu)


def run_profile(args, path, context, kv, depth):
    path.mkdir()
    executor = ReferenceExecutor(args.server, args.model, context, path / 'engine.log', key_type=kv, value_type=kv,
                                 ubatch=128, batch=512, mtp_depth=depth, draft_ubatch=64,
                                 prefix_cache=True, cache_ram=0, checkpoints=4)
    executor.env = {key: value for key, value in os.environ.items()
                    if not key.upper().startswith(('LLAMA_ARG_', 'GGML_', 'CUDA_VISIBLE_DEVICES'))}
    executor.argv += ['--reasoning', 'on', '--reasoning-effort', 'xhigh', '--reasoning-budget', '-1',
                      '--lazy-mode', 'off', '--override-tensor', '^token_embd\\.weight$=CUDA0', '--no-warmup']
    if depth:
        executor.argv += ['--spec-draft-override-tensor', '^token_embd\\.weight$=CUDA0']
        if args.filled_mtp:
            for flag in ('--spec-draft-type-k', '--spec-draft-type-v'):
                executor.argv[executor.argv.index(flag)+1] = 'q4_0'
    row = dict(context=context, kv=kv, mtp_depth=depth, argv=executor.argv, requests={}, status='failed')
    try:
        with executor:
            placement = row['placement'] = check_placement(executor.log_path, context)
            row['startup'] = gpu_resident(executor, placement)
            mode = row['template_parameter'] = template_probe(executor, path)
            count = min(120000, context-1024) if context <= 131072 else context-1024
            if path.name.startswith('mtp-'):
                count = 64000
            limit = 512
            kinds = ('cold', 'cached', 'sustained')
            if args.long_quality:
                limit = 16384
                count = context - limit - 256
                kinds = ('cold', 'cached')
            if args.filled_mtp:
                count, limit = context-2048, 1792
                tokens, expected = filled_retrieval(executor, mode, count)
                with gzip.open(path / 'input-token-ids.json.gz', 'wt', encoding='utf-8') as stream:
                    json.dump(tokens, stream)
            else:
                tokens = render_tokens(executor, mode, count)
                expected = MARKER
            fixture = dict(retrieval_tokens=count, total_output_limit=limit,
                           expected=expected, needle_depths=[.1, .5, .9] if args.filled_mtp else [.8],
                           input_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
            atomic_json(path / 'fixture.json', fixture)
            for kind in kinds:
                cached = kind != 'cold'
                request_limit = 512 if kind == 'sustained' else limit
                with Monitor(executor, output=path / f'{kind}-telemetry.jsonl') as monitor:
                    result = complete(executor, tokens, request_limit, cached, force_length=kind == 'sustained')
                answer = result['final_answer']
                if args.filled_mtp:
                    try:
                        answer = json.loads(answer) if answer else None
                    except ValueError:
                        answer = None
                    result['needle_matches'] = {key: isinstance(answer, dict) and answer.get(key) == value
                                                for key, value in expected.items()} if kind != 'sustained' else None
                result.update(resources=monitor.report(), retrieval_pass=(answer == expected)
                              if kind != 'sustained' else None)
                row['requests'][kind] = result
                atomic_json(path / f'{kind}.json', result)
                timings = result['timings']
                if (result['truncated'] or timings['prompt_n'] + timings['cache_n'] != count
                        or bool(timings['cache_n']) != cached or not result['output_tokens']
                        or (kind == 'sustained' and result['output_tokens'] != 512)
                        or (depth and not timings.get('draft_n'))):
                    raise RuntimeError(f'{kind} request failed token/cache/decode coverage validation')
                print(path.name, kind, timings, 'exact', result['retrieval_pass'], flush=True)
            row['end'] = gpu_resident(executor, placement)
        row['status'] = 'measured'
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        row['error'] = str(error)
        print(path.name, row['error'], flush=True)
    finally:
        row['server_stopped'] = executor.process is None or executor.process.poll() is not None
        atomic_json(path / 'result.json', row)
    return row


def filled_retrieval(executor, mode, count):
    sentinel = 'HOMELAB_FILLED_ARCHIVE_846291'
    expected = dict(A='fern-bd731a', B='cobalt-4d89f2', C='sable-9f132e')
    content = ('Read this archive. Only signed active records are authoritative.\n' + sentinel +
               '\nReturn the active keys A, B and C as one JSON object with string values. '
               'Ignore revoked records. Do not include a Markdown fence.')
    payload = dict(messages=[dict(role='user', content=content)],
                   chat_template_kwargs=dict(enable_thinking=True))
    (payload['chat_template_kwargs'] if mode == 'kwargs' else payload)['reasoning_effort'] = 'xhigh'
    rendered = executor.json('/apply-template', payload)['prompt']
    if 'Reasoning effort is set to xhigh.' not in rendered or not rendered.rstrip().endswith('<think>'):
        raise RuntimeError('filled prompt did not render xhigh thinking')
    left, right = rendered.split(sentinel)
    tokens = executor.tokenize(left)
    suffix = executor.tokenize(right)
    block = executor.tokenize(''.join(
        f'Historical ticket {i:04d}: module=codec-{i % 31}; unit tests passed; status closed.\n'
        for i in range(512)))
    for depth, (key, value) in zip((.1, .5, .9), expected.items()):
        needed = int(count*depth)-len(tokens)
        tokens.extend((block*((needed+len(block)-1)//len(block)))[:needed])
        tokens.extend(executor.tokenize(f'\nREVOKED record: key {key} = obsolete-{key}-777.\n'
                                       f'SIGNED ACTIVE record: key {key} = {value}.\n'))
    needed = count-len(tokens)-len(suffix)
    if needed < 0:
        raise ValueError('filled retrieval has insufficient headroom')
    tokens.extend((block*((needed+len(block)-1)//len(block)))[:needed])
    tokens.extend(suffix)
    assert len(tokens) == count
    return tokens, expected


def run_coding(args):
    from opencode_agent import CLI, run
    from swift_qwen_agent import reasoning_counts

    path = args.output / 'coding-64k-q4'
    path.mkdir()
    if subprocess.check_output([str(CLI), '--version'], text=True).strip() != '1.18.16':
        raise RuntimeError('matched coding requires OpenCode 1.18.16')
    manifest_path = Path(__file__).resolve().parent / 'results/campaign-64k/tasks/manifest.json'
    fixture = json.loads(manifest_path.read_text())
    tasks = [task for task in fixture['tasks']
             if task['instance_id'] in ('pytest-dev__pytest-6202', 'sphinx-doc__sphinx-10323')]
    for task in tasks:
        subprocess.run(['docker', 'image', 'inspect', task['image_id']], stdout=subprocess.DEVNULL, check=True)
        if not (manifest_path.parent / task['instance_id'] / 'oracle.json').is_file():
            raise RuntimeError(f'missing frozen grader for {task["instance_id"]}')
    label = 'swift-iq3s-homelab'
    executor = ReferenceExecutor(args.server, args.model, 65536, path / 'engine.log', key_type='q4_0',
                                 value_type='q4_0', ubatch=128, prefix_cache=True, cache_ram=0, checkpoints=8)
    executor.argv += ['--alias', label, '--reasoning', 'on', '--reasoning-effort', 'xhigh',
                      '--reasoning-budget', '-1', '--reasoning-format', 'deepseek', '--lazy-mode', 'off',
                      '--override-tensor', '^token_embd\\.weight$=CUDA0', '--no-warmup']
    for key, value in SAMPLING.items():
        if key != 'seed':
            executor.argv += ['--' + ('temp' if key == 'temperature' else key.replace('_', '-')), str(value)]
    atomic_json(path / 'config.json', dict(argv=executor.argv, sha256=SHA256, context=65536,
                output_limit=16384, explicit_reasoning_cap=False, sampling=SAMPLING))
    atomic_json(path / 'status.json', dict(status='loading', deadline=time.time()+7200))
    try:
        with executor:
            placement = check_placement(executor.log_path, 65536)
            atomic_json(path / 'gpu-start.json', gpu_resident(executor, placement))
            mode = template_probe(executor, path)
            request = dict(SAMPLING, model=label, max_tokens=16384,
                           messages=[dict(role='user', content='What is 17 * 19?')],
                           chat_template_kwargs=dict(enable_thinking=True))
            (request['chat_template_kwargs'] if mode == 'kwargs' else request)['reasoning_effort'] = 'xhigh'
            smoke = executor.json('/v1/chat/completions', request)
            atomic_json(path / 'chat-smoke.json', dict(request=request, response=smoke))
            message = smoke['choices'][0]['message']
            if not message.get('reasoning_content') or '323' not in (message.get('content') or ''):
                raise RuntimeError('chat generation did not prove separated thinking and a correct answer')
            run(path, executor.url, executor.process.pid, executor.log_path, efficiency=True, skip_stability=True,
                workflow_guidance=True, trials=3, context=65536, reference=True, reference_profile=label, model=label,
                output_tokens=16384, reasoning_effort='xhigh', unrestricted_reasoning=True, sampling=SAMPLING,
                start_trial=args.start_trial)
            reasoning_counts(path, executor)
    finally:
        atomic_json(path / 'runtime-stop.json', dict(stopped=executor.process is None or executor.process.poll() is not None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    llm = Path.home() / 'llm'
    parser.add_argument('--server', type=Path, default=llm / 'llama.cpp/build/bin/llama-server')
    parser.add_argument('--model', type=Path, default=llm / 'models' / FILENAME)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replace-pid', type=int, required=True)
    parser.add_argument('--live-audit', type=Path, required=True)
    parser.add_argument('--coding', action='store_true')
    parser.add_argument('--coding-only', action='store_true')
    parser.add_argument('--start-trial', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--long-quality', type=int, choices=(196608, 262144),
                        help='one Q4 retrieval pair with a 16K total output allowance')
    parser.add_argument('--filled-mtp', action='store_true',
                        help='filled 128K/192K/256K Q4 target/draft KV, depths 0/3/5, three needles per prompt')
    parser.add_argument('--headroom', action='store_true',
                        help='256K Q4 MTP3 with 128K input, a large output allowance and tool continuations')
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    for name, reuse in [('cold', False), ('cached', True)]:
        evidence = json.loads((args.live_audit / f'{name}.json').read_text())
        timing = evidence['timings']
        if (timing['prompt_n'] + timing['cache_n'] != 120000 or bool(timing['cache_n']) != reuse
                or not evidence['retrieval_pass']):
            raise RuntimeError('live 120K cache and exact-retrieval validation is required first')
    model = inspect_gguf(args.model, checksum=True)
    if model['sha256'] != SHA256:
        raise RuntimeError('unexpected Swift checkpoint')
    proc = Path(f'/proc/{args.replace_pid}')
    argv_bytes = (proc / 'cmdline').read_bytes()
    argv = [os.fsdecode(arg) for arg in argv_bytes.rstrip(b'\0').split(b'\0')]
    env = {os.fsdecode(k): os.fsdecode(v) for item in (proc / 'environ').read_bytes().split(b'\0')
           if b'=' in item for k, v in [item.split(b'=', 1)]}
    cwd = (proc / 'cwd').readlink()
    live = ReferenceExecutor(args.server, args.model, 131072, args.output / 'restored.log')
    live.url = 'http://127.0.0.1:8080'
    if (Path(argv[0]).resolve() != args.server.resolve() or str(args.model) not in argv or
            argv[argv.index('--port')+1] != '8080' or live.json('/slots')[0]['is_processing']):
        raise RuntimeError('the selected PID must be the idle live Swift server')
    plans = [(context, kv, 0, '') for context in (131072, 196608, 262144) for kv in ('q8_0', 'q4_0')]
    plans += [(131072, 'q4_0', depth, 'mtp-') for depth in (0, 3, 4, 5)]
    if args.long_quality:
        plans = [(args.long_quality, 'q4_0', 0, '')]
    if args.coding_only:
        plans = []
    if args.filled_mtp:
        if args.coding or args.coding_only or args.long_quality:
            parser.error('--filled-mtp is a standalone inference comparison')
        plans = [(context, 'q4_0', depth, 'filled-') for context in (131072, 196608, 262144)
                 for depth in (0, 3, 5)]
    if args.headroom:
        if args.filled_mtp or args.coding or args.coding_only or args.long_quality:
            parser.error('--headroom is a standalone experiment')
        plans = []
    atomic_json(args.output / 'provenance.json', dict(
        sha256=SHA256, hardware=hardware_inventory(), host=host_inventory(), original_argv=argv,
        version=subprocess.run([str(args.server), '--version'], capture_output=True, text=True, check=True).stderr,
        estimates=[dict(mtp_depth=depth, **estimate(model, context, kv, kv, mtp=bool(depth)))
                   for context, kv, depth, _ in plans]))
    if (proc / 'cmdline').read_bytes() != argv_bytes:
        raise RuntimeError('live server PID changed during preflight')
    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f'benchmark interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupt)
    atomic_json(args.output / 'status.json', dict(status='replacing_live_server'))
    failure = None
    os.kill(args.replace_pid, signal.SIGTERM)
    try:
        for _ in range(60):
            if not proc.exists() or (proc / 'stat').read_text().split(') ')[1][0] == 'Z':
                break
            time.sleep(1)
        else:
            raise RuntimeError('live server did not unload')
        results = []
        for context, kv, depth, prefix in plans:
            name = f'{prefix}{context}-{kv}-mtp{depth}'
            atomic_json(args.output / 'status.json', dict(status='running', profile=name))
            row = run_profile(args, args.output / name, context, kv, depth)
            results.append(row)
            atomic_json(args.output / 'results.json', results)
            if not row['server_stopped']:
                raise RuntimeError('benchmark server did not unload')
            if not args.long_quality and not args.filled_mtp and len(results) == 1 and (
                    row['status'] != 'measured' or not row['requests']['cold']['retrieval_pass']):
                raise RuntimeError('128K Q8 baseline gate failed')
        if args.coding or args.coding_only:
            atomic_json(args.output / 'status.json', dict(status='running', profile='coding-64k-q4'))
            run_coding(args)
        if args.headroom:
            from swift_headroom import run as run_headroom
            atomic_json(args.output / 'status.json', dict(status='running', profile='headroom-256k-q4-mtp3'))
            run_headroom(args)
    except BaseException as error:
        failure = error
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        atomic_json(args.output / 'status.json', dict(status='restoring', error=str(failure) if failure else None))
        if proc.exists() and (proc / 'stat').read_text().split(') ')[1][0] != 'Z':
            raise RuntimeError('original live server still exists; restoration not attempted')
        with (args.output / 'restored.log').open('wb') as log:
            restored = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(150):
            if restored.poll() is not None:
                break
            try:
                health = live.json('/health')
            except OSError:
                time.sleep(1)
                continue
            atomic_json(args.output / 'restoration.json', dict(pid=restored.pid, argv=argv, health=health))
            break
        else:
            raise RuntimeError('original configuration did not return to healthy within 150 seconds')
        if restored.poll() is not None:
            raise RuntimeError(f'original restoration failed; see {args.output}/restored.log')
        atomic_json(args.output / 'status.json', dict(status='failed' if failure else 'complete',
                    error=str(failure) if failure else None, restored_pid=restored.pid))


if __name__ == '__main__':
    main()
