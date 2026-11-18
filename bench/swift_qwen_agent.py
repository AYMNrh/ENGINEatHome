"""Run the same four fresh OpenCode attempts on Swift and original dense IQ3_S."""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from campaign_report import atomic_json
from opencode_agent import run
from qah.executor import ReferenceExecutor
from qah.model import sha256
from qah.telemetry import gpu_sample
from swift_qwen import FILENAME, MIB, SAMPLING, SHA256, check_placement, template_probe, wddm_sample


ORIGINAL_SHA256 = 'd847e2c1e4aa276e4b7b8e9ad7628050e61e165d49ab995407bc36677a6f3864'


def reasoning_counts(directory, executor):
    cases = []
    for path in sorted(directory.glob('*/opencode.stdout.jsonl')):
        steps, texts = [], []
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.startswith('{'):
                continue
            event = json.loads(line)
            if event.get('type') == 'step_finish':
                steps.append(event['part'].get('tokens') or {})
            if event.get('type') == 'reasoning':
                texts.append(event['part']['text'])
        reported = [step.get('reasoning') for step in steps]
        provider = sum(reported) if reported and all(isinstance(x, int) for x in reported) else None
        estimate = sum(len(executor.tokenize(text)) for text in texts) if texts else None
        cases.append(dict(case=path.parent.name, provider_reasoning_tokens=provider,
                          reasoning_trace_retokenized_estimate=estimate,
                          reasoning_parts=len(texts), step_count=len(steps)))
    atomic_json(directory / 'reasoning-counts.json', cases)


def agent_profile(args, label, model, model_sha):
    directory = args.output / label
    directory.mkdir()
    if sha256(model) != model_sha:
        raise RuntimeError(f'{label} model checksum mismatch')
    atomic_json(directory / 'status.json', dict(status='loading', deadline=time.time() + 7200))
    executor = ReferenceExecutor(args.server, model, 65536, directory / 'engine.log',
                                 key_type='q4_0', value_type='q4_0', batch=512, ubatch=128,
                                 prefix_cache=True, cache_ram=0, checkpoints=8)
    executor.env = {key: value for key, value in os.environ.items()
                    if not key.upper().startswith(('LLAMA_ARG_', 'GGML_', 'CUDA_VISIBLE_DEVICES'))}
    executor.argv += ['--alias', label, '--reasoning', 'on', '--reasoning-effort', 'xhigh',
                      '--reasoning-budget', '-1', '--reasoning-format', 'deepseek',
                      '--lazy-mode', 'off', '--override-tensor', '^token_embd\\.weight$=CUDA0', '--no-warmup']
    for key, value in SAMPLING.items():
        if key != 'seed':
            executor.argv += ['--' + ('temp' if key == 'temperature' else key.replace('_', '-')), str(value)]
    atomic_json(directory / 'config.json', dict(argv=executor.argv, sha256=model_sha,
                model=label, sampling=SAMPLING, context=65536, output_limit=args.output_limit,
                explicit_reasoning_cap=False))
    try:
        with executor:
            placement = check_placement(executor.log_path, 65536)
            wddm = wddm_sample(executor.process.pid)
            gpu = gpu_sample()
            buffers = placement['buffers']
            minimum = sum(buffers[f'CUDA0_{key}_bytes'] for key in ('model', 'KV', 'RS'))
            if (not gpu or gpu['free'] < 512*MIB or wddm.get('shared usage', float('inf')) > 512*MIB
                    or wddm.get('dedicated usage', 0) < minimum):
                raise RuntimeError('agent runtime failed the pre-inference dedicated residency check')
            atomic_json(directory / 'gpu-start.json', dict(placement=placement, wddm=wddm, gpu=gpu))
            mode = template_probe(executor, directory)
            chat_options = dict(enable_thinking=True)
            request = dict(SAMPLING, messages=[dict(role='user', content='What is 17 * 19?')],
                           model=label, max_tokens=args.output_limit, chat_template_kwargs=chat_options)
            (chat_options if mode == 'kwargs' else request)['reasoning_effort'] = 'xhigh'
            smoke = executor.json('/v1/chat/completions', request)
            atomic_json(directory / 'chat-smoke.json', dict(request=request, response=smoke))
            message = smoke['choices'][0]['message']
            if not message.get('reasoning_content') or not re.search(r'\b323\b', message.get('content') or ''):
                raise RuntimeError('chat API did not emit separated reasoning and a correct answer')
            run(directory, executor.url, executor.process.pid, executor.log_path,
                efficiency=True, skip_stability=True, workflow_guidance=True, trials=3,
                context=65536, reference=True, reference_profile=label, model=label,
                output_tokens=args.output_limit, reasoning_effort='xhigh',
                unrestricted_reasoning=True, sampling=SAMPLING)
            reasoning_counts(directory, executor)
            atomic_json(directory / 'gpu-end.json', wddm_sample(executor.process.pid))
    finally:
        atomic_json(directory / 'runtime-stop.json', dict(
            stopped=executor.process is None or executor.process.poll() is not None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speed-results', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--output-limit', type=int, choices=(16384, 32768), default=16384)
    parser.add_argument('--server', type=Path, default=Path('F:/llama.cpp/b12110-cuda13/llama-server.exe'))
    parser.add_argument('--swift', type=Path, default=Path('F:/MODELS/Swift-1.5-Qwen3.8-27B') / FILENAME)
    parser.add_argument('--original', type=Path, default=Path(
        'F:/Strata-QAH-comparison/models/Dense-UD-IQ3_S/Qwen3.8-27B-UD-IQ3_S.gguf'))
    args = parser.parse_args()
    status = json.loads((args.speed_results / 'status.json').read_text(encoding='utf-8'))
    speed = json.loads((args.speed_results / 'results.json').read_text(encoding='utf-8'))
    baseline = speed[0]
    if status['status'] != 'complete' or baseline['status'] != 'measured' or (
            not baseline['requests']['cold']['retrieval_pass'] or not baseline['server_stopped']):
        raise RuntimeError('validated and unloaded 64K speed baseline required')
    args.output.mkdir(parents=True)
    lock = Path(__file__).resolve().parents[1] / 'build/agent-debug.lock'
    lock.parent.mkdir(exist_ok=True)
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'], check=True, timeout=30)
        for label, model, checksum in [('swift-iq3s', args.swift, SHA256),
                                       ('qwen38-iq3s', args.original, ORIGINAL_SHA256)]:
            atomic_json(args.output / 'status.json', dict(status='running', profile=label))
            agent_profile(args, label, model, checksum)
        atomic_json(args.output / 'status.json', dict(status='complete'))
    except Exception as error:
        atomic_json(args.output / 'status.json', dict(status='failed', error=str(error)))
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
