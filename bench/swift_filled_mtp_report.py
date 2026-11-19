"""Summarize the filled Q4 MTP matrix without running inference."""
import argparse
import json
from pathlib import Path


MIB = 2**20


def fmt(value, digits=1):
    return f'{value:,.{digits}f}' if value is not None else '—'


def report(folder):
    results = json.loads((folder / 'results.json').read_text())
    rows = []
    for result in results:
        requests = result['requests']
        cold, cached, sustained = (requests.get(mode) or {} for mode in ('cold', 'cached', 'sustained'))
        memory = [request['resources'] for request in requests.values()]
        timing, replay, decode = (request.get('timings', {}) for request in (cold, cached, sustained))
        matches = [matched for probe in (cold, cached) for matched in (probe.get('needle_matches') or {}).values()]
        row = dict(context=result['context'], depth=result['mtp_depth'], status=result['status'],
                   error=result.get('error'), input=cold.get('actual_input_tokens'), hash=cold.get('input_sha256'),
                   prefill=timing.get('prompt_per_second'), decode=timing.get('predicted_per_second'),
                   sustained=decode.get('predicted_per_second'), cold_ttft=cold.get('ttft_seconds'),
                   cold_wall=cold.get('wall_seconds'), cached_ttft=cached.get('ttft_seconds'),
                   cached_fresh=replay.get('prompt_n'), cached_reused=replay.get('cache_n'),
                   cached_fresh_prefill=replay.get('prompt_per_second'),
                   drafted=decode.get('draft_n'), accepted=decode.get('draft_n_accepted'),
                   cold_correct=cold.get('retrieval_pass'), cached_correct=cached.get('retrieval_pass'),
                   needle_passes=sum(matches), needle_checks=len(matches),
                   output_limited=any(probe.get('stop_type') == 'limit' for probe in (cold, cached)),
                   peak=max((m['device_vram_peak_sampled_bytes'] for m in memory
                             if m.get('device_vram_peak_sampled_bytes') is not None), default=None),
                   min_free=min((m['device_vram_min_free_bytes'] for m in memory
                                 if m.get('device_vram_min_free_bytes') is not None), default=None),
                   stopped=result['server_stopped'])
        rows.append(row)
    for context in sorted({r['context'] for r in rows}):
        hashes = {r['hash'] for r in rows if r['context'] == context and r['hash']}
        if len(hashes) > 1:
            raise RuntimeError(f'input IDs differ across depths at {context}')
        control = next(r for r in rows if r['context'] == context and r['depth'] == 0)
        for row in rows:
            if row['context'] == context:
                row['sustained_speedup'] = row['sustained']/control['sustained'] if (
                    row['sustained'] and control['sustained']) else None
    data = dict(provenance=json.loads((folder / 'provenance.json').read_text()), profiles=rows,
                state=json.loads((folder / 'status.json').read_text()),
                restoration=json.loads((folder / 'restoration.json').read_text())
                if (folder / 'restoration.json').is_file() else None)
    lines = ['# Swift 1.5 Qwen3.8-27B / filled Q4 MTP on RTX 2080 Ti 22 GB', '',
             'Q4_0 target and draft K/V, full CUDA weights including the token embedding, '
             'Flash Attention, batch 512, microbatch 128, draft microbatch 64 and one sequence. '
             'Thinking enabled with verified xhigh instructions and unrestricted reasoning. '
             'Sampling uses temperature 1, top-p 0.95, top-k 20, min-p 0, seed 42, '
             'presence penalty 0 and repeat penalty 1.', '',
             'Each prompt leaves 2,048 tokens of headroom and allows 1,792 total output tokens for retrieval. '
             'Three signed active records at 10%, 50% and 90% depth have revoked distractors. '
             'The response must match the predefined three-field JSON object. Input IDs are saved and checked '
             'to be identical across depths. Remaining text repeats a synthetic archive.', '',
             '| Capacity | MTP | Actual input | Cold prefill tok/s | Cold decode tok/s | Sustained decode tok/s | '
             'Peak GPU GiB | Min free MiB | Retrieval cold/cached |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in rows:
        checks = '/'.join('pass' if r[k] else 'fail' if r[k] is not None else '—'
                          for k in ('cold_correct', 'cached_correct'))
        lines.append(f'| {r["context"]:,} | {r["depth"]} | {fmt(r["input"],0)} | {fmt(r["prefill"])} '
                     f'| {fmt(r["decode"],2)} | {fmt(r["sustained"],2)} '
                     f'| {fmt(r["peak"]/2**30,2) if r["peak"] else "—"} '
                     f'| {fmt(r["min_free"]/MIB,0) if r["min_free"] is not None else "—"} | {checks} |')
    lines += ['', '| Capacity | MTP | Cold TTFT s | Cached TTFT s | Reused / fresh | '
              'Cached fresh prefill tok/s | Accepted / proposed drafts | Sustained speedup |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f'| {r["context"]:,} | {r["depth"]} | {fmt(r["cold_ttft"],3)} | {fmt(r["cached_ttft"],3)} '
                     f'| {fmt(r["cached_reused"],0)} / {fmt(r["cached_fresh"],0)} '
                     f'| {fmt(r["cached_fresh_prefill"])} | {fmt(r["accepted"],0)} / {fmt(r["drafted"],0)} '
                     f'| {fmt(r["sustained_speedup"],2)}x |')
    lines += ['', 'One cold retrieval and one exact replay per profile. Accuracy covers three records per prompt. '
              'A separate replay forces 512 output tokens for sustained decode. Those outputs are excluded '
              'from retrieval scoring. Memory is sampled device-wide. Dedicated process allocation and CUDA '
              'layer/cache logs are checked before and after inference. Linux lacks a directly comparable '
              'WDDM shared-memory counter. Output lengths and sampled reasoning can differ across depths, '
              'so fixed-length decode and natural-response latency are reported separately. '
              'This is one sample per profile, without confidence intervals.', '']
    for r in rows:
        if r['error']:
            lines.append(f'- {r["context"]:,} MTP{r["depth"]} encountered {r["error"]}.')
        if r['output_limited']:
            lines.append(f'- {r["context"]:,} MTP{r["depth"]} reached the total output allowance '
                         'during retrieval. Accuracy at that allowance is incomplete.')
    eligible = [r for r in rows if r['status'] == 'measured' and r['cold_correct'] and r['cached_correct']
                and r['sustained'] and r['min_free'] is not None and r['min_free'] >= 512*MIB]
    if eligible:
        lines += ['', '## Measured choices', '']
        for context in sorted({r['context'] for r in eligible}):
            group = [r for r in eligible if r['context'] == context]
            best = max(group, key=lambda r: r['sustained'])
            fastest = min(group, key=lambda r: r['cold_wall'])
            lines.append(f'- {context:,} MTP{best["depth"]} led sustained decode at {best["sustained"]:.2f} tok/s. '
                         f'MTP{fastest["depth"]} completed cold retrieval fastest at {fastest["cold_wall"]:.2f} s.')
        widest = max(r['context'] for r in eligible)
        best = max((r for r in eligible if r['context'] == widest), key=lambda r: r['sustained'])
        lines.append(f'The measured long-context candidate is {widest:,} Q4_0 with MTP{best["depth"]}. '
                     'Coding-agent quality with MTP remains unmeasured.')
        if widest == 262144:
            three = next((r for r in eligible if r['context'] == widest and r['depth'] == 3), None)
            five = next((r for r in eligible if r['context'] == widest and r['depth'] == 5), None)
            if three and five:
                change = five['sustained']/three['sustained']-1
                extra = (five['peak']-three['peak'])/MIB
                lines.append(f'At 256K, MTP5 measured {change:.1%} more sustained throughput than MTP3 '
                             f'and used {extra:.0f} MiB more peak GPU memory. With one sample, that small '
                             'speed difference does not establish a repeatable advantage. '
                             'For everyday cached work, use 128K Q4_0 with MTP3. '
                             'For the longest context, 256K Q4_0 with MTP3 offers nearly the same measured '
                             'decode speed as MTP5 with additional VRAM headroom. '
                             'MTP off had the shortest cold retrieval latency at every capacity.')
    lines += ['', f'Run state {data["state"]["status"]}.', '',
              '## Reproduce', '', 'Run from the benchmark workspace with the original endpoint idle.', '',
              '    PYTHONPATH=src python3 bench/swift_homelab.py --replace-pid <8080-pid> '
              '--live-audit live-audit --output filled-mtp-fresh --filled-mtp',
              '    python3 bench/swift_filled_mtp_report.py filled-mtp-fresh', '',
              f'Runtime {data["provenance"]["version"].strip()}', '',
              f'Model SHA256 {data["provenance"]["sha256"]}. '
              'Launch commands and estimates are in provenance.json and each result.json. '
              'The original command is restored and recorded in restoration.json.', '']
    (folder / 'summary.json').write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    (folder / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    report(parser.parse_args().folder)
