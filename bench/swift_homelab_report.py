"""Compare Homelab Swift results with the published RTX 5070 Ti measurements."""
import argparse
import json
from pathlib import Path


def read(path, fallback=None):
    return json.loads(path.read_text()) if path.is_file() else fallback


def row(label, requests):
    cold, cached, sustained = (requests.get(mode) for mode in ('cold', 'cached', 'sustained'))
    if not cold:
        return f'| {label} | — | — | — | — | — | — | — |'
    timing = cold['timings']
    replay = f'{cached["ttft_seconds"]:.3f} s / {cached["timings"]["cache_n"]:,}' if cached else '—'
    sustained_rate = f'{sustained["timings"]["predicted_per_second"]:.2f}' if sustained else '—'
    peak = cold['resources']['device_vram_peak_sampled_bytes']
    retrieval = 'pass' if cold['retrieval_pass'] and cached and cached['retrieval_pass'] else 'failed or incomplete'
    return (f'| {label} | {cold["actual_input_tokens"]:,} | {timing["prompt_per_second"]:.1f} | '
            f'{timing["predicted_per_second"]:.2f} | {replay} | {sustained_rate} | {peak/2**30:.2f} | {retrieval} |')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--live-audit', type=Path, default=Path('live-audit'))
    parser.add_argument('--reference', type=Path, default=Path('docs/results/swift-qwen-20261118.json'))
    parser.add_argument('--coding-resume', type=Path)
    args = parser.parse_args()
    measured = read(args.run / 'results.json', [])
    original = read(args.reference)
    lines = ['# Swift 1.5 Qwen3.8-27B / RTX 2080 Ti 22 GB', '',
             '| Profile | Actual input | Cold prefill tok/s | Cold decode tok/s | Cached TTFT / reused tokens | '
             'Sustained decode tok/s | Cold peak VRAM GiB | Retrieval |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    lines.append(row('Live 128K Q8', {name: read(args.live_audit / f'{name}.json') for name in ('cold', 'cached')}))
    for result in measured:
        label = f'{result["context"]:,} {result["kv"]} MTP{result["mtp_depth"]}'
        lines.append(row(label, result['requests']))
    for reference in original['speed']:
        key = (reference['context'], reference['kv'], reference['mtp_depth'],
               reference.get('batch', 512), reference.get('microbatch', 128))
        if reference['status'] == 'measured' and key in {(65536, 'q4_0', 0, 512, 128),
                (92160, 'q4_0', 0, 512, 128), (65536, 'q4_0', 3, 128, 32)}:
            lines.append(row(f'5070 Ti {key[0]:,} Q4 MTP{key[2]}', reference['requests']))
    lines += ['', 'Cold requests require zero reused tokens. Sustained decode forces 512 output tokens. '
              'VRAM is sampled device-wide. The 5070 Ti used a different llama.cpp build. Its MTP3 run used '
              'Q4 draft KV and 128/32 batch; Homelab MTP uses Q8 draft KV and 512/128 batch.', '']
    lines += [f'- {r["context"]:,} {r["kv"]} MTP{r["mtp_depth"]} — {r.get("error", r["status"])}'
              for r in measured if r['status'] != 'measured']
    for result in measured:
        cold = result['requests'].get('cold', {})
        if cold and not cold['retrieval_pass'] and cold['stop_type'] == 'limit' and not cold['thinking_closed']:
            lines.append(f'- {result["context"]:,} {result["kv"]} reached its {cold["output_limit"]:,}-token '
                         'total output allowance during thinking. Retrieval remains unverified.')
    verified = [result for result in measured if result['status'] == 'measured'
                and all(result['requests'][mode]['retrieval_pass'] for mode in ('cold', 'cached'))]
    if verified:
        widest = max(verified, key=lambda result: result['requests']['cold']['actual_input_tokens'])
        lines += ['', f'Largest profile that passed retrieval and the residency guard: '
                  f'{widest["context"]:,} capacity with {widest["kv"]} and '
                  f'{widest["requests"]["cold"]["actual_input_tokens"]:,} actual input tokens.', '']
    mtp = [result for result in verified if result['mtp_depth']]
    if mtp:
        lines += ['', '| MTP depth | Sustained accepted / proposed drafts | Cold accepted / proposed drafts |',
                  '| ---: | ---: | ---: |']
        for result in mtp:
            cold, sustained = (result['requests'][mode]['timings'] for mode in ('cold', 'sustained'))
            lines.append(f'| {result["mtp_depth"]} | {sustained["draft_n_accepted"]}/{sustained["draft_n"]} '
                         f'| {cold["draft_n_accepted"]}/{cold["draft_n"]} |')
    coding = args.run / 'coding-64k-q4'
    win = original['agent']['swift-iq3s']
    lines += ['', '## Coding', '', f'- RTX 5070 Ti {win["strict"]}/{win["tasks"]} strict and '
              f'{win["patch"]}/{win["tasks"]} patches in {win["seconds"]/60:.2f} minutes.']
    coding_dirs = [coding]
    if args.coding_resume:
        coding_dirs.append(args.coding_resume / 'coding-64k-q4')
    events = [json.loads(line) for directory in coding_dirs if (directory / 'events.jsonl').is_file()
              for line in (directory / 'events.jsonl').read_text().splitlines()]
    tasks = [item for item in events if item['kind'] == 'task_result']
    recovered = read(coding / 'sphinx-timeout-offline-grade/result.json')
    if recovered:
        tasks.insert(1, recovered)
    if tasks:
        generated = sum(event['timings']['predicted_n'] for event in events if event['kind'] == 'request_result')
        lines.append(f'- RTX 2080 Ti {sum(t["passed"] for t in tasks)}/{len(tasks)} strict and '
                     f'{sum(t["grade"]["passed"] for t in tasks)}/{len(tasks)} patches in '
                     f'{sum(t["elapsed_seconds"] for t in tasks)/60:.2f} minutes.')
        lines += ['', '| Issue | Trial | Strict pass | Patch pass | Time min | Tools | Timed out |',
                  '| --- | ---: | --- | --- | ---: | ---: | --- |']
        for task in tasks:
            lines.append(f'| {task["instance_id"]} | {task["trial"]} | {task["passed"]} | {task["grade"]["passed"]}'
                         f' | {task["elapsed_seconds"]/60:.2f} | {task["tool_calls"]} | {task["agent_timeout"]} |')
        if recovered:
            lines += ['', 'Sphinx reached the 20-minute limit. Its retained patch was graded offline '
                      'after a Windows-only cleanup command failed on Linux. The two remaining Pytest '
                      'runs used the corrected process-group cleanup.']
        if generated and not recovered:
            lines += ['', f'Homelab generated {generated:,} native output tokens, including reasoning.']
        reasoning = [case for directory in coding_dirs for case in read(directory / 'reasoning-counts.json', [])]
        counts = [case['reasoning_trace_retokenized_estimate'] for case in reasoning]
        if counts and all(value is not None for value in counts):
            lines.append(f'Retokenizing captured traces from {len(counts)} attempts estimates '
                         f'{sum(counts):,} reasoning tokens.')
    else:
        lines.append('- RTX 2080 Ti coding results pending.')
    state = read(args.run / 'status.json', {})
    resumed = read(args.coding_resume / 'status.json', {}) if args.coding_resume else {}
    lines += ['', f'Run state {state.get("status", "pending")}. '
              'Original endpoint restoration is recorded in restoration.json.',
              *([f'Coding continuation state {resumed["status"]}.'] if 'status' in resumed else []), '',
              '## Reproduction', '',
              'Use bench/swift_homelab.py with the verified checkpoint and exact llama-server build in '
              'provenance.json. Each profile directory records the launch arguments, prompt hash, '
              'rendered effort controls, output IDs, timings, and sampled telemetry. The live audit '
              'records the original 120K cold and cached requests. The coding files retain the frozen '
              'task manifest hash, OpenCode traces, patches, and individual grader outcomes.', '',
              'From the Homelab benchmark workspace, regenerate this report with '
              'python3 bench/swift_homelab_report.py results-20261118 --coding-resume coding-resume-20261118.', '']
    args.run.joinpath('report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
