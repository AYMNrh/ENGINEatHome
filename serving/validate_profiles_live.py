"""Live multi-profile validation. Run ON the homelab against a gateway URL; uses short inference requests only (no long-context benchmark).

  python validate_profiles_live.py --base http://127.0.0.1:18090 [--full] [--oom-profile selftest-oom] [--out results.json]

--full runs every stage (switch chain, reasoning levels, protection, failure injection); without it only the quick smoke
(Swift -> Inkling Balanced -> Swift) runs. Credentials come from credentials.json next to this file; nothing secret is printed.
Aborts if the GPU is not idle or another gateway holds the model. Always leaves Swift applied.
"""
import argparse
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
KEYS = json.loads(next(p for p in (HERE / 'credentials.json', Path.home() / 'llm/serving/credentials.json') if p.exists()).read_text())
PROMPT = 'What is 17*3? Answer with just the number.'
LONG = ' '.join(f'Paragraph {i}: the quick brown fox number {i} jumps over lazy dog {i * 7} near river {i * 3}.' for i in range(110))
results = {}


def call(base, method, path, key='admin', body=None, timeout=900):
    request = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={'Authorization': 'Bearer ' + KEYS[key], 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), json.loads(response.read() or b'null')
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:return error.code, dict(error.headers), json.loads(raw)
        except ValueError:return error.code, dict(error.headers), {'raw': raw[:200].decode(errors='replace')}


def gpu():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,power.limit', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split(',')
    procs = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.strip().splitlines()
    return dict(used_mib=int(out[0]), total_mib=int(out[1]), power_limit_w=float(out[2]), processes=len(procs))


def llama_processes():
    out = subprocess.run(['pgrep', '-af', 'llama-server'], capture_output=True, text=True).stdout.strip().splitlines()
    return [line for line in out if 'pgrep' not in line]


def status(base):
    return call(base, 'GET', '/admin/status')[2]


def apply(base, key):
    began = time.time()
    code, _, body = call(base, 'POST', '/admin/profile/select', body={'profile': key})
    assert code == 200 and body['profile']['selected'] == key, body
    code, _, body = call(base, 'POST', '/admin/profile/apply')
    return code, round(time.time() - began, 1), body


def chat(base, model, effort=None, prompt=PROMPT, max_tokens=500, key='python', extra=None):
    body = dict(model=model, messages=[dict(role='user', content=prompt)], max_tokens=max_tokens)
    if effort is not None:body['reasoning_effort'] = effort
    body.update(extra or {})
    began = time.time()
    code, headers, data = call(base, 'POST', '/v1/chat/completions', key, body)
    out = dict(status=code, seconds=round(time.time() - began, 1), effort_header=headers.get('X-Reasoning-Effort'), profile_header=headers.get('X-Model-Profile'))
    if code == 200:
        message, usage = data['choices'][0]['message'], data['usage']
        out.update(finish=data['choices'][0]['finish_reason'], prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'),
                   cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens'), reasoning_chars=len(message.get('reasoning_content') or ''),
                   answer_has_51='51' in (message.get('content') or ''))
    else:
        out['error'] = (data.get('error') or {}).get('code') or data
    return out


def stage(name, fn):
    print(f'== {name}', flush=True)
    began = time.time()
    results[name] = fn()
    results[name]['_seconds'] = round(time.time() - began, 1)
    print(json.dumps(results[name], indent=1)[:2600], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--oom-profile', default=None)
    parser.add_argument('--production-base', default='http://127.0.0.1:8080')
    parser.add_argument('--out', default='/tmp/eah-live-validation.json')
    args = parser.parse_args()
    base = args.base
    profiles = {p['key']: p for p in status(base)['profiles']}
    swift, balanced, maximum = (profiles[k]['model_id'] for k in ('swift', 'inkling-balanced', 'inkling-maximum'))

    if args.production_base != base:   # a staging run must not fight the production gateway for the GPU
        other = call(args.production_base, 'GET', '/admin/status')[2]['model']
        assert other['state'] == 'unloaded' and other['active_requests'] == 0, 'production gateway holds the GPU; aborting'
    st = status(base)
    assert st['model']['state'] in ('unloaded', 'ready') and st['model']['active_requests'] == 0 and st['model']['session_leases'] == 0
    results['preflight'] = dict(gpu=gpu(), llama_processes=len(llama_processes()), state=st['model']['state'])
    print('preflight', results['preflight'])

    def switch_stage(key, model, levels=None, cache_probe=False):
        before = gpu()
        code, seconds, body = apply(base, key)
        out = dict(apply_status=code, apply_seconds=seconds)
        if code != 200:
            out['error'] = body.get('error');return out
        after = status(base)
        out.update(state=after['model']['state'], active=after['profile']['active'], applied=after['profile']['applied'], model_id=after['model']['active_model_id'],
                   context=after['model']['context_capacity'], gpu_after=gpu(), llama_processes=len(llama_processes()), changed=body['apply']['changed'])
        out['models_endpoint'] = [(m['id'], m['loaded']) for m in call(base, 'GET', '/v1/models', 'opencode')[2]['data']]
        out['first_request'] = chat(base, model)
        if cache_probe:
            a = chat(base, model, 'none', prompt=LONG + ' Reply: ok', max_tokens=8)
            b = chat(base, model, 'none', prompt=LONG + ' Reply: ok', max_tokens=8)
            out['cache_probe'] = dict(first_cached=a.get('cached'), second_cached=b.get('cached'), prompt_tokens=a.get('prompt_tokens'))
        if levels:
            out['levels'] = {str(level): chat(base, model, level) for level in levels}
        return out

    stage('1_apply_swift', lambda: switch_stage('swift', swift, ['none', 'low', 'medium', 'xhigh'] if args.full else None, cache_probe=True))
    stage('2_swift_to_balanced', lambda: switch_stage('inkling-balanced', balanced, ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 0.5] if args.full else None, cache_probe=True))
    results['2b_inactive_model'] = chat(base, swift)
    print('inactive-model request ->', results['2b_inactive_model'])
    if args.full:
        stage('3_balanced_to_maximum', lambda: switch_stage('inkling-maximum', maximum, ['none', 'high'], cache_probe=True))
    stage('4_back_to_swift', lambda: switch_stage('swift', swift, ['low', 'xhigh'] if args.full else None, cache_probe=True))

    if args.full:
        def protection():
            out = {}
            box = {}
            def long_request():
                box['r'] = chat(base, swift, 'medium', prompt='Write a detailed 300 word story about a lighthouse keeper.', max_tokens=900)
            thread = threading.Thread(target=long_request);thread.start()
            time.sleep(6)
            out['active_requests_seen'] = status(base)['model']['active_requests']
            code, _, body = apply(base, 'inkling-balanced');out['apply_during_request'] = (code, (body.get('error') or {}).get('code'))
            code, _, body = call(base, 'POST', '/admin/unload', body={});out['unload_during_request'] = (code, (body.get('error') or {}).get('code'))
            code, _, body = call(base, 'POST', '/admin/restart', body={});out['restart_during_request'] = (code, (body.get('error') or {}).get('code'))
            thread.join()
            out['request_completed_untouched'] = box['r'].get('status') == 200
            code, _, lease = call(base, 'POST', '/v1/leases', 'python', {'ttl_seconds': 120})
            code2, _, body = apply(base, 'inkling-balanced');out['apply_with_lease'] = (code2, (body.get('error') or {}).get('code'))
            call(base, 'DELETE', '/v1/leases/' + lease['lease_id'], 'python')
            out['still_swift'] = status(base)['profile']['active'] == 'swift'
            call(base, 'POST', '/admin/profile/select', body={'profile': 'swift'})
            return out
        stage('5_protection', protection)

        def failure():
            key = args.oom_profile
            if not key:return dict(skipped='no failure profile configured')
            before = dict(processes=len(llama_processes()), gpu=gpu())
            code, seconds, body = apply(base, key)
            after = status(base)
            out = dict(apply_status=code, seconds=seconds, error=(body.get('error') or {}).get('reason'), restored=(body.get('error') or {}).get('restored'),
                       message=(body.get('error') or {}).get('message'), before=before, state_after=after['model']['state'], active_after=after['profile']['active'],
                       applied_after=after['profile']['applied'], last_switch=after['profile']['last_switch'], gpu_after=gpu(), processes_after=len(llama_processes()))
            out['swift_inference_after_failure'] = chat(base, swift)
            call(base, 'POST', '/admin/profile/select', body={'profile': 'swift'})
            return out
        stage('6_failure_injection', failure)

    final = status(base)
    results['final'] = dict(state=final['model']['state'], active=final['profile']['active'], applied=final['profile']['applied'], last_error=final['model']['last_error'])
    results['final_gpu'] = gpu()
    Path(args.out).write_text(json.dumps(results, indent=1))
    print('\nWROTE', args.out)


if __name__ == '__main__':main()
