"""Persist honest campaign totals and hourly summaries without running GPU work."""
import argparse
import datetime
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from qah.telemetry import resource_summary


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A concurrent writer may not yet have completed its last line.
                if line.endswith('\n'):
                    raise
    return rows


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.%d.tmp' % os.getpid())
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def aggregate(events, samples, window_seconds=None):
    invalidated = {e['excluded_phase'] for e in events if e['kind'] == 'infrastructure_invalidation' and 'excluded_phase' in e}
    invalid_attempts = {tuple(e['excluded_attempt']) for e in events if e['kind'] == 'infrastructure_invalidation' and 'excluded_attempt' in e}
    def unscored(event):
        return event['kind'] in ('task_started', 'task_result') and (event.get('phase') in invalidated or
            tuple(event.get(k) for k in ('profile', 'phase', 'instance_id', 'trial')) in invalid_attempts)
    excluded = [e for e in events if unscored(e)]
    events = [e for e in events if not unscored(e)]
    tasks = [e for e in events if e['kind'] == 'task_result']
    probes = [e for e in events if e['kind'] == 'request_result']
    groups = defaultdict(list)
    for event in probes:
        prompt = event.get('prompt_tokens')
        band = 'unknown' if prompt is None else ('0-8K' if prompt < 8192 else '8-32K' if prompt < 32768 else '32-48K' if prompt < 49152 else '48-64K')
        groups[(event['profile'], event.get('context'), band)].append(event)
    rates = []
    for (profile, context, prompt), group in groups.items():
        decode_ms = sum(e.get('timings', {}).get('predicted_ms', 0) for e in group)
        tokens = sum(e.get('timings', {}).get('predicted_n', 0) for e in group)
        prompt_ms = sum(e.get('timings', {}).get('prompt_ms', 0) for e in group)
        prompt_n = sum(e.get('timings', {}).get('prompt_n', 0) for e in group)
        lengths = [e['prompt_tokens'] for e in group if e.get('prompt_tokens') is not None]
        rates.append(dict(profile=profile, context=context, prompt_tokens=prompt,
                          prompt_tokens_min=min(lengths, default=None), prompt_tokens_max=max(lengths, default=None),
                          accepted_output_tokens_per_second=tokens * 1000 / decode_ms if decode_ms else None,
                          prefill_tokens_per_second=prompt_n * 1000 / prompt_ms if prompt_ms else None,
                          generated_tokens=tokens, requests=len(group)))
    elapsed = sum(e['elapsed_seconds'] for e in tasks)
    passed = sum(e.get('passed') is True for e in tasks)
    cache_events = [e for e in probes if e.get('cached_input_tokens') is not None]
    result = dict(task_attempts=len([e for e in events if e['kind'] == 'task_started']),
                  infrastructure_task_starts=sum(e['kind'] == 'task_started' for e in excluded),
                  infrastructure_task_results=sum(e['kind'] == 'task_result' for e in excluded),
                  tasks_completed=len(tasks), tasks_passed=passed,
                  tasks_failed=sum(e.get('passed') is False for e in tasks),
                  tasks_ungraded=sum(e.get('passed') is None for e in tasks),
                  completed_successful_tasks_per_hour=passed * 3600 / elapsed if elapsed else None,
                  task_elapsed_median_seconds=statistics.median(e['elapsed_seconds'] for e in tasks) if tasks else None,
                  rates=rates, resources=resource_summary(samples),
                  tool_call_errors=sum(e.get('tool_call_errors') or 0 for e in tasks),
                  tool_calls=sum(e.get('tool_calls') or 0 for e in tasks),
                  compactions=sum(e.get('compactions') or 0 for e in tasks),
                  compaction_counters_missing=sum(e.get('compactions') is None for e in tasks),
                  cached_input_tokens=sum(e.get('cached_input_tokens') or 0 for e in (cache_events or tasks)),
                  cached_input_tokens_source='request stream' if cache_events else 'episode usage')
    by_profile = defaultdict(list)
    for task in tasks:
        by_profile[(task.get('profile'), task.get('phase'))].append(task)
    result['task_groups'] = [dict(profile=profile, phase=phase, completed=len(rows),
        passed=sum(r.get('passed') is True for r in rows),
        elapsed_seconds=sum(r['elapsed_seconds'] for r in rows))
        for (profile, phase), rows in by_profile.items()]
    rss = [e.get('memory', {}).get('process_peak_rss_bytes') for e in events]
    rss = [v for v in rss if v is not None]
    if rss:
        result['resources']['process_peak_rss_bytes'] = max(rss)
    energy = result['resources']['gpu_energy_wh']
    if window_seconds is not None:
        result['resources']['power_coverage_fraction'] = min(1, result['resources']['power_covered_seconds'] / window_seconds) if window_seconds > 0 else None
    coverage = result['resources']['power_coverage_fraction']
    result['gpu_wh_per_passed_task'] = energy / passed if passed and energy is not None and coverage is not None and coverage >= .95 else None
    return result


def write_report(directory, now=None):
    directory = Path(directory)
    now = time.time() if now is None else now
    state = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
    events = read_rows(directory / 'events.jsonl')
    samples = read_rows(directory / 'telemetry.jsonl')
    report = dict(timestamp=now, phase=state['phase'], status=state['status'],
                  active_profile=state.get('active_profile'), remaining_hours=max(0, state['deadline'] - now) / 3600,
                  blockers=state.get('blockers', []), next_experiment=state.get('next_experiment'),
                  cumulative=aggregate(events, samples, max(0, now - state['started_at'])),
                  last_hour=aggregate([e for e in events if e['kind'] == 'infrastructure_invalidation' or now - 3600 < e['time'] <= now],
                                      [s for s in samples if now - 3600 < s['time'] <= now], min(3600, max(0, now - state['started_at']))))
    atomic_json(directory / 'latest-report.json', report)
    hour = int(max(0, now - state['started_at']) // 3600)
    reports = directory / 'hourly'
    reports.mkdir(exist_ok=True)
    atomic_json(reports / ('hour-%02d.json' % hour), report)
    def show(value):
        return 'unavailable' if value is None else str(round(value, 3)) if isinstance(value, float) else str(value)
    lines = ['# 64K campaign status', '', datetime.datetime.fromtimestamp(now, datetime.timezone.utc).isoformat(),
             '', 'Phase: %s; status: %s; profile: %s' % (state['phase'], state['status'], state.get('active_profile')),
             'Remaining hours: ' + show(report['remaining_hours'])]
    for name in ('last_hour', 'cumulative'):
        data = report[name]
        lines += ['', '## ' + name, '', 'Tasks attempted/completed/passed/failed/ungraded: %s / %s / %s / %s / %s' % tuple(
            data[k] for k in ('task_attempts', 'tasks_completed', 'tasks_passed', 'tasks_failed', 'tasks_ungraded')),
                  'Successful tasks/hour: ' + show(data['completed_successful_tasks_per_hour']),
                  'Median task elapsed seconds: ' + show(data['task_elapsed_median_seconds']),
                  'Tool calls/errors/compactions: %s / %s / %s' % (data['tool_calls'], data['tool_call_errors'], data['compactions']),
                  'Cached input tokens: ' + show(data['cached_input_tokens'])]
        for rate in data['rates']:
            lines.append('%s context=%s prompt=%s: accepted output %s tok/s; prefill %s tok/s' % (
                rate['profile'], rate['context'], rate['prompt_tokens'], show(rate['accepted_output_tokens_per_second']), show(rate['prefill_tokens_per_second'])))
        for group in data['task_groups']:
            lines.append('%s / %s: passed %s of %s; elapsed %s seconds' % (
                group['profile'], group['phase'], group['passed'], group['completed'], show(group['elapsed_seconds'])))
        for key, value in data['resources'].items():
            lines.append('%s: %s' % (key, show(value)))
        lines.append('GPU Wh per passed task (includes campaign overhead): ' + show(data['gpu_wh_per_passed_task']))
    lines += ['', 'Blockers: ' + '; '.join(state.get('blockers', [])), 'Next: ' + str(state.get('next_experiment'))]
    (directory / 'latest-report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', default='bench/results/campaign-64k')
    args = parser.parse_args()
    print(json.dumps(write_report(args.directory), indent=2))
