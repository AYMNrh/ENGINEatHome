"""Render the Swift context, MTP and matched-agent results into ENGINEatHome reports."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIB = 2**20


def read(path, fallback=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else fallback


def speed_summary(path):
    rows = []
    sources = read(path / 'results.json', [])
    controls = {(row['context'], row['kv']): row.get('requests', {}) for row in sources if not row['mtp_depth']}
    for source in sources:
        row = {key: value for key, value in source.items() if key != 'requests'}
        requests = row['requests'] = {}
        for name, result in source.get('requests', {}).items():
            requests[name] = {key: value for key, value in result.items()
                              if key not in ('output', 'output_token_ids', 'final_answer')}
            if source['mtp_depth'] and name in ('cold', 'cached'):
                control = controls.get((source['context'], source['kv']), {}).get(name, {})
                if control.get('output_token_ids') and result.get('output_token_ids'):
                    requests[name]['output_ids_equal_to_control'] = (
                        result['output_token_ids'] == control['output_token_ids'])
        rows.append(row)
    return rows


def agent_summary(path):
    if path is None:
        return {}
    profiles = {}
    for profile in ('swift-iq3s', 'qwen38-iq3s'):
        directory = path / profile
        state = read(directory / 'status.json')
        if state is None:
            continue
        events_path = directory / 'events.jsonl'
        events = [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines()
                  if line.endswith('}')] if events_path.is_file() else []
        tasks = [event for event in events if event['kind'] == 'task_result']
        requests = [event for event in events if event['kind'] == 'request_result']
        reasoning = read(directory / 'reasoning-counts.json', [])
        by_case = {case['case']: case for case in reasoning}
        reason_count = [case['provider_reasoning_tokens'] for case in reasoning]
        trace_count = [case['reasoning_trace_retokenized_estimate'] for case in reasoning]
        profiles[profile] = dict(
            status=state['status'], blockers=state.get('blockers'), tasks=len(tasks),
            strict=sum(task['passed'] for task in tasks),
            patch=sum(task['grade']['passed'] for task in tasks),
            seconds=sum(task['elapsed_seconds'] for task in tasks),
            tools=sum(task['tool_calls'] for task in tasks),
            generated=sum(event['timings']['predicted_n'] for event in requests) if requests else None,
            provider_reasoning=sum(reason_count)
            if reason_count and all(total is not None for total in reason_count) else None,
            estimated_reasoning=sum(trace_count)
            if trace_count and all(total is not None for total in trace_count) else None,
            results=[dict(task=task['instance_id'], trial=task['trial'], strict=task['passed'],
                          patch=task['grade']['passed'], seconds=task['elapsed_seconds'], tools=task['tool_calls'],
                          exit=task['turn_exit_reason'], timeout=task['agent_timeout'],
                          shell_errors=task['shell_command_failures'], tool_errors=task['tool_call_errors'],
                          reasoning=by_case.get(f'{profile}-{task["instance_id"]}-t{task["trial"]}'))
                     for task in tasks],
            configuration=read(directory / 'config.json'),
            stopped=read(directory / 'runtime-stop.json'))
    return profiles


def number(value, digits=0):
    return f'{value:,.{digits}f}' if value is not None else '—'


def benchmark_evidence(speed, agent):
    manifest = ROOT / 'bench/results/campaign-64k/tasks/manifest.json'
    fixture = read(manifest, {})
    fixture_identity = {key: fixture[key] for key in ('dataset', 'revision', 'parquet_sha256')
                        if key in fixture}
    fixture_identity['tasks'] = [
        {key: task[key] for key in ('instance_id', 'repo', 'base_commit', 'image', 'image_id',
                                    'regression', 'preservation')}
        for task in fixture.get('tasks', [])
    ]
    if manifest.is_file():
        fixture_identity['manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    profiles = {}
    for label in ('swift-iq3s', 'qwen38-iq3s'):
        directory = agent / label
        events = directory / 'events.jsonl'
        if not events.is_file():
            continue
        tasks = []
        for line in events.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['kind'] != 'task_result':
                continue
            task = {key: row[key] for key in ('instance_id', 'trial', 'grade', 'passed', 'agent_finished',
                                               'task_manifest_sha256')}
            patch = directory / f'{label}-{row["instance_id"]}-t{row["trial"]}' / 'agent.patch'
            if patch.is_file():
                task['agent_patch'] = patch.read_text(encoding='utf-8')
            tasks.append(task)
        profiles[label] = dict(rendered_prompts=read(directory / 'rendered-prompts.json'),
                               chat_smoke=read(directory / 'chat-smoke.json'), tasks=tasks,
                               runtime_stop=read(directory / 'runtime-stop.json'))
    return dict(fixture=fixture_identity,
                speed_rendered_prompts=read(speed / '65536-q4_0-mtp0/rendered-prompts.json'),
                agent=profiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('speed', type=Path)
    parser.add_argument('--agent', type=Path)
    parser.add_argument('--tight-vram', type=Path)
    parser.add_argument('--supplement', type=Path, action='append', default=[])
    parser.add_argument('--name', default='swift-qwen-20261118')
    args = parser.parse_args()
    provenance = read(args.speed / 'provenance.json')
    speed, agent = speed_summary(args.speed), agent_summary(args.agent)
    if args.tight_vram:
        speed += speed_summary(args.tight_vram)
    for extra in args.supplement:
        speed += speed_summary(extra)
    data = dict(provenance=provenance, speed=speed, agent=agent,
                speed_status=read(args.speed / 'status.json'),
                agent_status=read(args.agent / 'status.json') if args.agent else None)
    output = ROOT / 'docs/results' / (args.name + '.json')
    output.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    if args.agent:
        evidence = output.with_name(args.name + '-evidence.json')
        evidence.write_text(json.dumps(benchmark_evidence(args.speed, args.agent), indent=2) + '\n',
                            encoding='utf-8')
    runtime = provenance['runtime'].splitlines()[0]
    lines = [
        '# Swift 1.5 Qwen3.8-27B on the RTX 5070 Ti', '',
        f'[Pinned IQ3_S MTP checkpoint]({provenance["checkpoint"]}) · {provenance["file_bytes"]:,} bytes',
        f'\nSHA-256 `{provenance["sha256"]}` · {runtime}.', '',
        'Native Windows CUDA, one sequence, full CUDA weight placement including the token embedding, '
        'Flash Attention, batch 512 and baseline microbatch 128. Both explicit xhigh and low were checked '
        'through rendered prompts. The xhigh render instructed the model to validate assumptions and '
        'consider alternatives, then opened the thinking block. The runtime used unrestricted thinking '
        '(`--reasoning-budget -1`). '
        'Sampling used temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0, '
        'repeat penalty 1, and seed 42.', '',
        '![Swift filled-context inference measurements](assets/swift-qwen-inference.png)', '',
        '## Inference results', '',
        '| Capacity | KV | MTP | Batch / microbatch | Actual input | Cold prefill tok/s | Cold decode tok/s | Cold TTFT s | '
        'Cached TTFT s | Cached fresh/reused | Output cold/cached | Retrieval cold/cached | Draft accepted / proposed |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |',
    ]
    for row in speed:
        cold, cached = (row.get('requests', {}).get(mode, {}) for mode in ('cold', 'cached'))
        timing, replay = cold.get('timings', {}), cached.get('timings', {})
        values = [
            number(row['context']), row['kv'], row['mtp_depth'],
            f'{row.get("batch", 512)} / {row.get("microbatch", 128)}',
            number(cold.get('actual_input_tokens')),
            number(timing.get('prompt_per_second'), 1), number(timing.get('predicted_per_second'), 2),
            number(cold.get('ttft_seconds'), 3), number(cached.get('ttft_seconds'), 3),
            f'{number(replay.get("prompt_n"))} / {number(replay.get("cache_n"))}',
            f'{number(cold.get("output_tokens"))} / {number(cached.get("output_tokens"))}',
            ' / '.join('pass' if item.get('retrieval_pass') else 'fail' if item else '—'
                       for item in (cold, cached)),
            f'{number(timing.get("draft_n_accepted"))} / {number(timing.get("draft_n"))}',
        ]
        lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', 'The cold and cached requests use the same prompt, including repeated synthetic archive text. '
              'Outputs stop naturally with a 512-token ceiling. Cached prefill measures only fresh tokens. '
              'The matrix contains one cold request and one replay per profile. Sampled MTP reasoning and output '
              'length can differ from the control even when the exact retrieval answer agrees. '
              'Timings, output-ID equality and draft acceptance are in the JSON.', '',
              '## Memory and output headroom', '',
              '| Capacity | KV | MTP | Batch / microbatch | CUDA weights / KV / recurrent / compute MiB | '
              'Device peak / min free MiB | Startup free MiB | Shared startup / peak MiB | Min host free GiB | Status |',
              '| ---: | --- | ---: | ---: | --- | --- | ---: | --- | ---: | --- |']
    for row in speed:
        buffers = row.get('placement', {}).get('buffers', {})
        requests = list(row.get('requests', {}).values())
        resources = [item['resources'] for item in requests]
        def aggregate(items, key, reducer, unit):
            values = [item[key] for item in items if item.get(key) is not None]
            return number(reducer(values)/unit, 2 if unit > MIB else 0) if values else '—'
        values = [number(row['context']), row['kv'], row['mtp_depth'],
                  f'{row.get("batch", 512)} / {row.get("microbatch", 128)}',
                  ' / '.join(number(buffers[key]/MIB) if key in buffers else '—'
                             for key in ('CUDA0_model_bytes', 'CUDA0_KV_bytes', 'CUDA0_RS_bytes',
                                         'CUDA0_compute_bytes')),
                  ' / '.join([aggregate(resources, 'device_vram_peak_sampled_bytes', max, MIB),
                              aggregate(resources, 'device_vram_min_free_bytes', min, MIB)]),
                  aggregate([row.get('gpu_start', {})], 'free', min, MIB),
                  ' / '.join([aggregate([row.get('wddm_start', {})], 'shared usage', max, MIB),
                              aggregate(requests, 'wddm_shared_peak_bytes', max, MIB)]),
                  aggregate(resources, 'host_min_available_bytes', min, 2**30),
                  row.get('error', row['status']).replace('|', '/')]
        lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', 'VRAM is sampled device-wide; WDDM shared memory is sampled per server process. '
              'Compute sizes are the last logged reservation. Estimate refusals are based on the '
              'checkpoint, KV and recurrent allocations and device headroom before inference.', '']
    if speed:
        for cap in (16384, 32768):
            request = speed[0].get('requests', {}).get(f'output-cap-{cap}')
            if request:
                lines.append(f'- At the {cap:,}-token total output limit, the {request["actual_input_tokens"]:,}-token '
                             f'prompt generated {request["output_tokens"]:,} tokens before stopping. '
                             'This validates the request with headroom and does not measure sustained output to the limit.')
        qualified = [row for row in speed if row['status'] == 'measured' and row['mtp_depth'] == 0
                     and all(row['requests'].get(mode, {}).get('retrieval_pass') for mode in ('cold', 'cached'))]
        if qualified:
            largest = max(qualified, key=lambda row: row['context'])
            lines += ['', f'**Largest context verified by a filled retrieval/replay pair** '
                      f'{largest["context"]:,} capacity with {largest["kv"]} KV. '
                      'Sustained coding stability requires separate evidence. '
                      'The 64K Q4 single-sequence baseline is the configuration used for matched agent testing.']
        mtp = [row for row in speed if row['status'] == 'measured' and row['mtp_depth']
               and all(row['requests'].get(mode, {}).get('retrieval_pass') for mode in ('cold', 'cached'))]
        if mtp:
            lines += ['', '**MTP scope**']
            for context in sorted({row['context'] for row in mtp}):
                candidates = [row for row in mtp if row['context'] == context]
                fastest = max(candidates, key=lambda row: row['requests']['cold']['timings']['predicted_per_second'])
                r = fastest['requests']['cold']
                lines.append(f'- At {context:,} capacity, the fastest measured cold decode was '
                             f'{r["timings"]["predicted_per_second"]:.2f} tok/s with MTP{fastest["mtp_depth"]}; '
                             f'minimum sampled free VRAM was {r["resources"]["device_vram_min_free_bytes"]//MIB:,} MiB.')
            lines.append('The MTP samples passed exact retrieval. The matched coding benchmark used MTP off.')
    lines += ['', '## Matched coding attempts', '',
              '![Swift and original Qwen matched coding results](assets/swift-qwen-coding.png)', '']
    if agent:
        lines += ['Four guided OpenCode attempts per model when complete, with Pytest #6202 ×3 and Sphinx #10323 ×1.',
                  '', '| Model | Graded | Strict | Patch | Task minutes | Tools | Native generated | '
                  'Provider reasoning | Retokenized reasoning estimate |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for label, row in agent.items():
            lines.append(f'| {label} | {row["tasks"]} | {row["strict"]} | {row["patch"]} | '
                         f'{row["seconds"]/60:.2f} | {row["tools"]} | {number(row["generated"])} | '
                         f'{number(row["provider_reasoning"])} | {number(row["estimated_reasoning"])} |')
        lines += ['', 'The API can omit a separate reasoning usage counter. If OpenCode reports zero '
                  'despite a nonempty reasoning trace, the retokenized trace estimate is the usable measure. '
                  'Native generated tokens include reasoning where generated.']
        swift, original = agent.get('swift-iq3s'), agent.get('qwen38-iq3s')
        if swift and original and all(row['tasks'] == 4 and row['status'] == 'completed'
                                     for row in (swift, original)):
            reduction = 1 - swift['seconds'] / original['seconds']
            change = f'{abs(reduction):.1%} {"less" if reduction >= 0 else "more"}'
            lines += ['', f'Swift used {change} total task time and passed '
                      f'{swift["strict"]}/4 strictly. Original Qwen passed {original["strict"]}/4. '
                      f'Both used a {swift["configuration"]["output_limit"]:,}-token output allowance per model '
                      'request, the same user workflow '
                      'guidance, and the same offline grader.']
            for label, model in (('Swift', swift), ('Original Qwen', original)):
                for case in model['results']:
                    if not case['strict']:
                        outcome = 'timed out' if case['timeout'] else f'ended with {case["exit"]}'
                        grade = 'passed' if case['patch'] else 'failed'
                        lines.append(f'- {label} {outcome} on `{case["task"]}` trial {case["trial"]}. '
                                     f'Its patch {grade} the held-out grade.')
            if swift['strict'] == 4 and swift['patch'] == 4 and reduction >= 0:
                lines += ['', '**Recommended coding configuration** Swift IQ3_S at 64K, Q4_0 KV, '
                          'xhigh thinking, unrestricted reasoning, and MTP off. This recommendation '
                          'covers the four guided attempts and does not establish hour-long stability.']
    else:
        lines += ['Matched coding results are pending.']
    lines += ['', '## Reproduce', '', '```powershell', "$env:PYTHONPATH='src'",
              'python bench/swift_qwen.py --output bench/results/swift-fresh',
              'python bench/swift_qwen_agent.py --speed-results bench/results/swift-fresh '
              '--output F:/Strata-QAH-comparison/agent-results/swift-fresh',
              'python bench/swift_qwen.py --tight-vram --microbatch 32 --draft-kv q4_0 '
              '--output bench/results/swift-tight-fresh',
              'python bench/swift_qwen.py --tight-vram --batch 128 --microbatch 32 --draft-kv q4_0 '
              '--output bench/results/swift-smallbatch-fresh',
              'python bench/swift_qwen.py --smaller-mtp-context --batch 128 --microbatch 32 --draft-kv q4_0 '
              '--output bench/results/swift-mtp32k-fresh  # if 64K MTP cannot fit',
              'python bench/swift_qwen_report.py bench/results/swift-fresh --tight-vram bench/results/swift-tight-fresh '
              '--supplement bench/results/swift-smallbatch-fresh --supplement bench/results/swift-mtp32k-fresh '
              '--agent F:/Strata-QAH-comparison/agent-results/swift-fresh', '```', '',
              'The coding suite requires Docker, OpenCode 1.18.16 and the frozen fixture images. '
              'For a fresh fixture, obtain the SWE-bench Verified Parquet revision and checksum listed in the '
              'evidence, then use the fixture preparation commands below. Exact replay requires the recorded '
              'image IDs. A rebuild may resolve different base images or dependency versions.', '',
              '```powershell',
              'python bench/agent_tasks.py select <verified-parquet> bench/results/campaign-64k/task-pool.json',
              'python bench/agent_tasks.py prepare bench/results/campaign-64k/task-pool.json bench/results/campaign-64k/tasks',
              '```', '',
              'Full launch arguments, request timings and individual task results are in the measurements.', '',
              f'[Measurements JSON](results/{args.name}.json)', '']
    if args.agent:
        lines += [f'[Fixture identity, rendered prompts and grading evidence](results/{args.name}-evidence.json)', '']
    (ROOT / 'docs' / (args.name + '.md')).write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
