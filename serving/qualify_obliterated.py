"""Production qualification of the Qwen3.8-27B-OBLITERATED profile through a *staging* gateway (own port and database, real GPU).

Run ON the homelab with the production gateway stopped (one GPU, one model):
    python qualify_obliterated.py --base http://127.0.0.1:18095 --kv f16 --out qual-f16.json [--stages 1,2,7] [--soak-minutes 15]

Stages: 1 apply+MTP  2 short answers  3 acceptance  4 16K coding  5 multi-turn cache  6 interleaved clients  7 65K coding+needle
        8 tool calling (streaming)  9 thinking levels  10 soak  11 failed-load recovery  12 switching to/from Swift (cache never crosses models)
Credentials come from credentials.json (never printed). Always leaves Swift applied when it finishes.
"""
import argparse
import glob
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
KEYS = json.loads(next(p for p in (HERE / 'credentials.json', Path.home() / 'llm/serving/credentials.json') if p.exists()).read_text())
SRC = Path.home() / 'llm/llama.cpp'
results = {}
OBL, SWIFT = 'qwen3.8-27b-obliterated', 'swift-1.5-qwen3.8-27b'
CHARS_PER_TOKEN = float(os.environ.get('QUAL_CHARS_PER_TOKEN', '3.05'))


def call(base, method, path, key='admin', body=None, timeout=3600, stream=False):
    request = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={'Authorization': 'Bearer ' + KEYS[key], 'Content-Type': 'application/json'})
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:return error.code, dict(error.headers), json.loads(raw)
        except ValueError:return error.code, dict(error.headers), {'raw': raw[:200].decode(errors='replace')}
    with response:
        if stream:return response.status, dict(response.headers), response.read().decode()
        return response.status, dict(response.headers), json.loads(response.read() or b'null')


def nvidia():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,power.draw,power.limit,temperature.gpu', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.split(',')
    return dict(used_mib=int(out[0]), total_mib=int(out[1]), power_w=float(out[2]), power_limit_w=float(out[3]), temp_c=int(out[4]))


def host_ram():
    info = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, _, rest = line.partition(':')
        if name in ('MemTotal', 'MemAvailable'):info[name] = int(rest.split()[0]) // 1024
    return info


def llama_pids():
    return [int(x) for x in subprocess.run(['pgrep', '-x', 'llama-server'], capture_output=True, text=True).stdout.split()]


def rss_mib():
    total = 0
    for pid in llama_pids():
        try:
            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                if line.startswith('VmRSS:'):total += int(line.split()[1]) // 1024
        except OSError:pass
    return total


class Sampler(threading.Thread):
    def __init__(self, period=2.0):
        super().__init__(daemon=True)
        self.rows, self.stop, self.period = [], False, period

    def run(self):
        while not self.stop:
            try:self.rows.append((time.time(), nvidia()['used_mib'], rss_mib(), host_ram()['MemAvailable']))
            except Exception:pass
            time.sleep(self.period)

    def summary(self):
        if not self.rows:return {}
        return dict(vram_peak_mib=max(r[1] for r in self.rows), rss_peak_mib=max(r[2] for r in self.rows), host_available_min_mib=min(r[3] for r in self.rows),
                    samples=len(self.rows))


def corpus_text(tokens_chars):
    files = sorted(glob.glob(str(SRC / 'src/*.cpp')) + glob.glob(str(SRC / 'ggml/src/ggml-cuda/*.cu')))
    random.Random(11).shuffle(files)
    text = ''
    for f in files:
        text += f'// ==== FILE {os.path.basename(f)} ====\n' + open(f, errors='ignore').read() + '\n'
        if len(text) >= tokens_chars:break
    return text[:tokens_chars]


def other_corpus_text(chars, skip=0):
    files = sorted(glob.glob(str(SRC / 'gguf-py/gguf/*.py')) + glob.glob(str(SRC / 'scripts/*.py')) + glob.glob(str(SRC / 'tools/server/tests/unit/*.py')))[skip:]
    text = ''
    for f in files:
        text += f'# ---- {os.path.basename(f)} ----' + chr(10) + open(f, errors='ignore').read() + chr(10)
        if len(text) >= chars:break
    return text[:chars]


def prompt_with_needle(tokens, code):
    """~tokens of real C++/CUDA source with a unique code word in the middle; returns (text, code word)."""
    body = corpus_text(int(tokens * CHARS_PER_TOKEN))   # measured: ~3.05 characters per token for this C++/CUDA corpus
    half = len(body) // 2
    return body[:half] + f'\n// NOTE FOR REVIEWERS: the release passphrase is {code}.\n' + body[half:]


def chat(base, model, messages, key='opencode', max_tokens=200, effort=None, **extra):
    body = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=0, **extra)
    if effort is not None:body['reasoning_effort'] = effort
    began = time.time()
    code, headers, data = call(base, 'POST', '/v1/chat/completions', key, body)
    out = dict(status=code, seconds=round(time.time() - began, 2))
    if code != 200:
        out['error'] = (data.get('error') or {}).get('code') or data
        return out, data
    message, usage, timings = data['choices'][0]['message'], data.get('usage', {}), data.get('timings', {})
    out.update(finish=data['choices'][0]['finish_reason'], prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'),
               cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens', timings.get('cache_n')), prompt_ms=round(timings.get('prompt_ms', 0)),
               prefill_tps=round(timings.get('prompt_per_second', 0), 1), decode_tps=round(timings.get('predicted_per_second', 0), 2),
               draft_n=timings.get('draft_n'), draft_accepted=timings.get('draft_n_accepted'),
               reasoning_chars=len(message.get('reasoning_content') or ''), content=(message.get('content') or '')[:300], full=(message.get('content') or ''),
               tool_calls=message.get('tool_calls'), profile=headers.get('X-Model-Profile'), effort=headers.get('X-Reasoning-Effort'))
    return out, data


def status(base):
    return call(base, 'GET', '/admin/status')[2]


def apply(base, key):
    call(base, 'POST', '/admin/profile/select', body={'profile': key})
    began = time.time()
    code, _, body = call(base, 'POST', '/admin/profile/apply')
    return code, round(time.time() - began, 1), body


def stage(name, fn):
    print(f'== {name}', flush=True)
    began = time.time()
    sampler = Sampler();sampler.start()
    try:results[name] = fn()
    except Exception as error:results[name] = dict(exception=repr(error))
    sampler.stop = True
    results[name]['_seconds'] = round(time.time() - began, 1)
    results[name]['_resources'] = sampler.summary()
    print(json.dumps(results[name], indent=1, default=str)[:2400], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--kv', required=True)
    parser.add_argument('--stages', default='1,2,3,4,5,6,7,8,9,10,11,12,13')
    parser.add_argument('--soak-minutes', type=float, default=15)
    parser.add_argument('--oom-profile', default='selftest-oom')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    base, wanted = args.base, {int(x) for x in args.stages.split(',')}
    st = status(base)
    assert st['model']['active_requests'] == 0 and st['model']['session_leases'] == 0
    assert not llama_pids() or st['model']['state'] == 'ready', 'another llama-server is running'
    results['environment'] = dict(kv=args.kv, gpu=nvidia(), ram=host_ram(), profiles=[p['key'] for p in st['profiles']])
    rng = random.Random(5)
    shared = {}

    def s1():
        code, seconds, body = apply(base, 'obliterated')
        if code != 200:return dict(apply=code, seconds=seconds, error=body.get('error'))
        after = status(base)
        out = dict(apply_status=code, apply_seconds=seconds, state=after['model']['state'], active=after['profile']['active'], model_id=after['model']['active_model_id'],
                   context=after['model']['context_capacity'], gpu_after=nvidia(), llama_processes=len(llama_pids()), rss_mib=rss_mib(), host=host_ram(),
                   models_endpoint=[(m['id'], m['loaded']) for m in call(base, 'GET', '/v1/models', 'opencode')[2]['data']])
        out['first_request'], _ = chat(base, OBL, [dict(role='user', content='Reply with the single word: ok')], max_tokens=24)
        return out

    def s2():
        out = {}
        for label, prompt, check in (('math', 'What is 17*3? Answer with just the number.', '51'), ('capital', 'What is the capital of France? One word.', 'Paris'),
                                     ('code', 'Write a Python function is_prime(n). Code only.', 'def is_prime')):
            for effort in ('none', 'low'):
                r, _ = chat(base, OBL, [dict(role='user', content=prompt)], max_tokens=700, effort=effort)
                r['correct'] = check.lower() in (r.get('content') or '').lower() or r.get('reasoning_chars', 0) > 0 and r.get('finish') == 'length'
                out[f'{label}_{effort}'] = r
        return out

    def s3():
        runs = []
        for i in range(3):
            code = f'ACC-{rng.randrange(10**6):06d}'
            text = prompt_with_needle(3500 + 500 * i, code)
            r, _ = chat(base, OBL, [dict(role='user', content=text + '\n\nExplain what the code above does in about 150 words, then state the release passphrase.')],
                        max_tokens=320, effort='none')
            r['needle_ok'] = code in (r.get('full') or '')
            runs.append(r)
        drafted = sum(r.get('draft_n') or 0 for r in runs)
        return dict(runs=runs, drafted=drafted, accepted=sum(r.get('draft_accepted') or 0 for r in runs),
                    acceptance=round(sum(r.get('draft_accepted') or 0 for r in runs) / drafted, 3) if drafted else None, mtp_active=drafted > 0)

    def s4():
        code = f'Q16-{rng.randrange(10**6):06d}'
        text = prompt_with_needle(16000, code)
        r, _ = chat(base, OBL, [dict(role='user', content=text + '\n\nQuestion 1: what is the release passphrase? Question 2: name three functions in the code above and what they do.')],
                    max_tokens=300, effort='none')
        r['needle_ok'] = code in (r.get('full') or '')
        return r

    def s5():
        code = f'MT-{rng.randrange(10**6):06d}'
        messages = [dict(role='user', content=prompt_with_needle(12000, code) + '\n\nSummarise the purpose of this code in two sentences.')]
        turns = []
        for i in range(4):
            r, data = chat(base, OBL, messages, max_tokens=160, effort='none')
            turns.append({k: r.get(k) for k in ('prompt_tokens', 'cached', 'prompt_ms', 'decode_tps', 'draft_n', 'draft_accepted', 'seconds', 'status')})
            if r['status'] != 200:break
            messages.append(dict(role='assistant', content=r['content']))
            messages.append(dict(role='user', content=f'Follow-up {i + 1}: now ' + ['list two risks.', 'what is the release passphrase?', 'give one improvement.', 'summarise again.'][i] + ' Extra context: ' + corpus_text(5000)[i * 1200:(i + 1) * 1200]))
        shared['multiturn'] = (messages[:-1], code)
        return dict(turns=turns, followups_reuse_prefix=all((t['cached'] or 0) > 0.7 * (t['prompt_tokens'] or 1) for t in turns[1:]))

    def s6():
        messages, code = shared['multiturn']
        before, _ = chat(base, OBL, messages + [dict(role='user', content='Which words did I give you in my first message as the passphrase? Answer briefly.')], key='opencode', max_tokens=60, effort='none')
        other_text, other_code = other_corpus_text(int(9000 * CHARS_PER_TOKEN)) + f' OTH-{rng.randrange(10**6):06d}', None
        side, _ = chat(base, OBL, [dict(role='user', content='Generate a 3-word title for a chat about C++ code review.')], key='opencode', max_tokens=12, effort='low')
        unrelated, _ = chat(base, OBL, [dict(role='user', content=other_text + '\n\nName one function above.')], key='python', max_tokens=60, effort='none')
        after, _ = chat(base, OBL, messages + [dict(role='user', content='Which words did I give you in my first message as the passphrase? Answer briefly.')], key='opencode', max_tokens=60, effort='none')
        return dict(before_cached=before.get('cached'), before_ttft_ms=before.get('prompt_ms'), side=dict(cached=side.get('cached'), ms=side.get('prompt_ms')),
                    unrelated=dict(cached=unrelated.get('cached'), prompt_tokens=unrelated.get('prompt_tokens'), ms=unrelated.get('prompt_ms')),
                    after_cached=after.get('cached'), after_prompt_tokens=after.get('prompt_tokens'), after_prompt_ms=after.get('prompt_ms'),
                    restored=(after.get('cached') or 0) > 0.7 * (after.get('prompt_tokens') or 1), no_foreign_marker=other_text[:0] == '' and 'OTH-' not in (after.get('content') or ''))

    def s7():
        code = f'Q65-{rng.randrange(10**6):06d}'
        text = prompt_with_needle(66000, code)
        messages = [dict(role='user', content=text + '\n\nQuestion 1: what is the release passphrase? Question 2: name three functions in the code above.')]
        r, _ = chat(base, OBL, messages, max_tokens=300, effort='none')
        r['needle_ok'] = code in (r.get('full') or '')
        messages += [dict(role='assistant', content=r.get('content') or ''), dict(role='user', content='Thanks. Which passphrase did you report? One line.')]
        f, _ = chat(base, OBL, messages, max_tokens=60, effort='none')
        gpu = nvidia()
        return dict(cold=r, follow_up=dict(cached=f.get('cached'), prompt_tokens=f.get('prompt_tokens'), prompt_ms=f.get('prompt_ms'), content=f.get('content')),
                    gpu_free_mib=gpu['total_mib'] - gpu['used_mib'], gpu=gpu, rss_mib=rss_mib(), host=host_ram())

    tools = [dict(type='function', function=dict(name='read_file', description='Read a text file from the repository.',
                   parameters=dict(type='object', properties=dict(path=dict(type='string', description='file path')), required=['path'])))]

    def sse_tool_call(effort):
        body = dict(model=OBL, messages=[dict(role='system', content='You are a coding agent. Use tools to inspect files before answering.'),
                                         dict(role='user', content='Read the file src/main.cpp and tell me how many lines it has.')],
                    tools=tools, tool_choice='auto', stream=True, max_tokens=800, temperature=0, reasoning_effort=effort)
        code, _, raw = call(base, 'POST', '/v1/chat/completions', 'opencode', body, stream=True)
        calls, text, reasoning = {}, '', ''
        for line in raw.splitlines():
            if not line.startswith('data:') or line.strip() == 'data: [DONE]':continue
            event = json.loads(line[5:])
            for choice in event.get('choices') or []:
                delta = choice.get('delta') or {}
                text += delta.get('content') or '';reasoning += delta.get('reasoning_content') or ''
                for tc in delta.get('tool_calls') or []:
                    entry = calls.setdefault(tc.get('index', 0), dict(name='', arguments=''))
                    entry['name'] += (tc.get('function') or {}).get('name') or ''
                    entry['arguments'] += (tc.get('function') or {}).get('arguments') or ''
        parsed = []
        for entry in calls.values():
            try:parsed.append(dict(name=entry['name'], arguments=json.loads(entry['arguments'] or '{}')))
            except ValueError:parsed.append(dict(name=entry['name'], arguments='INVALID_JSON:' + entry['arguments'][:80]))
        return code, parsed, text, reasoning

    def s8():
        out = {}
        for effort in ('none', 'xhigh'):
            code, parsed, text, reasoning = sse_tool_call(effort)
            ok = code == 200 and len(parsed) >= 1 and parsed[0]['name'] == 'read_file' and isinstance(parsed[0]['arguments'], dict) and 'path' in parsed[0]['arguments']
            follow = None
            if ok:
                msgs = [dict(role='system', content='You are a coding agent. Use tools to inspect files before answering.'),
                        dict(role='user', content='Read the file src/main.cpp and tell me how many lines it has.'),
                        dict(role='assistant', content=None, tool_calls=[dict(id='call_1', type='function', function=dict(name='read_file', arguments=json.dumps(parsed[0]['arguments'])))]),
                        dict(role='tool', tool_call_id='call_1', content='\n'.join(f'line {i}' for i in range(1, 43)))]
                follow, _ = chat(base, OBL, msgs, max_tokens=600, effort=effort, tools=tools)
            out[effort] = dict(http=code, tool_call_valid=ok, calls=parsed, text_before=text[:120], reasoning_chars=len(reasoning),
                               final_answer=(follow or {}).get('content'), final_mentions_42='42' in ((follow or {}).get('content') or ''), final_has_tool_call=bool((follow or {}).get('tool_calls')))
        return out

    def s9():
        out = {}
        for effort in ('none', 'low', 'medium', 'xhigh'):
            r, _ = chat(base, OBL, [dict(role='user', content='A farmer has 17 sheep; all but 9 run away. How many are left? Explain briefly.')], max_tokens=1500, effort=effort)
            out[effort] = {k: r.get(k) for k in ('status', 'effort', 'completion_tokens', 'reasoning_chars', 'finish', 'content', 'error')}
        out['none_has_no_reasoning'] = out['none']['reasoning_chars'] == 0
        return out

    def s10():
        deadline = time.time() + args.soak_minutes * 60
        rows, errors, i = [], 0, 0
        base_code = f'SOAK-{rng.randrange(10**6):06d}'
        messages = [dict(role='user', content=prompt_with_needle(48000, base_code) + '\n\nSummarise this code in two sentences.')]
        while time.time() < deadline:
            i += 1
            r, _ = chat(base, OBL, messages, max_tokens=300, effort='none', key='opencode' if i % 3 else 'python')
            if r['status'] != 200:
                errors += 1;rows.append(dict(i=i, error=r.get('error')));break
            rows.append({k: r.get(k) for k in ('prompt_tokens', 'cached', 'prompt_ms', 'decode_tps', 'seconds')})
            messages += [dict(role='assistant', content=r['content']), dict(role='user', content=f'Tool result {i}:\n{corpus_text(3000)[i * 40 % 1500:i * 40 % 1500 + 1500]}\nNow explain one detail in two sentences.')]
            if sum(1 for _ in messages) > 14 or (r['prompt_tokens'] or 0) > 61000:messages = messages[:1]
            if i % 4 == 0:chat(base, OBL, [dict(role='user', content='Give a 3-word title for a chat about GPUs.')], key='opencode', max_tokens=12, effort='low')
        return dict(requests=len(rows), errors=errors, median_decode_tps=statistics.median([r['decode_tps'] for r in rows if r.get('decode_tps')] or [0]),
                    median_prompt_ms=statistics.median([r['prompt_ms'] for r in rows if r.get('prompt_ms') is not None] or [0]), end_state=status(base)['model']['state'],
                    gpu=nvidia(), rss_mib=rss_mib(), host=host_ram())

    def s11():
        before = dict(processes=len(llama_pids()), gpu=nvidia())
        code, seconds, body = apply(base, args.oom_profile)
        after = status(base)
        out = dict(apply_status=code, seconds=seconds, error=(body.get('error') or {}).get('reason'), restored=(body.get('error') or {}).get('restored'), before=before,
                   state_after=after['model']['state'], active_after=after['profile']['active'], applied_after=after['profile']['applied'], processes_after=len(llama_pids()), gpu_after=nvidia())
        serving = after['model']['active_model_id'] or after['model']['model_id']
        out['inference_after_failure'], _ = chat(base, serving, [dict(role='user', content='Reply with the single word: ok')], max_tokens=24, effort='none')
        return out

    def s12():
        probe_text = prompt_with_needle(6000, 'SW-123456') + '\n\nReply: ok'
        out = {}
        apply(base, 'obliterated')
        a, _ = chat(base, OBL, [dict(role='user', content=probe_text)], max_tokens=8, effort='none')
        b, _ = chat(base, OBL, [dict(role='user', content=probe_text)], max_tokens=8, effort='none')
        out['obliterated_first_cached'], out['obliterated_repeat_cached'] = a.get('cached'), b.get('cached')
        code, seconds, _ = apply(base, 'swift')
        out['to_swift'] = dict(status=code, seconds=seconds, processes=len(llama_pids()), gpu=nvidia())
        c, _ = chat(base, SWIFT, [dict(role='user', content=probe_text)], max_tokens=8)
        d, _ = chat(base, SWIFT, [dict(role='user', content=probe_text)], max_tokens=8)
        out['swift_first_cached'], out['swift_repeat_cached'] = c.get('cached'), d.get('cached')
        code, seconds, _ = apply(base, 'obliterated')
        e, _ = chat(base, OBL, [dict(role='user', content=probe_text)], max_tokens=8, effort='none')
        out['back_to_obliterated'] = dict(status=code, seconds=seconds, first_cached=e.get('cached'), processes=len(llama_pids()))
        refused, _ = chat(base, SWIFT, [dict(role='user', content='hi')], max_tokens=8)
        out['wrong_model_request'] = refused.get('error') or refused.get('status')
        out['cache_never_crosses_models'] = (a.get('cached') or 0) < 64 and (c.get('cached') or 0) < 64 and (e.get('cached') or 0) < 64
        return out

    def s13():
        out = {}
        code, seconds, body = apply(base, 'swift')
        out['apply_swift'] = dict(status=code, seconds=seconds)
        if code != 200:return dict(out, error=body.get('error'))
        out['rss_after_load_mib'] = rss_mib()
        convs = {}
        for name, size in (('A', 20000), ('B', 22000), ('C', 18000)):
            mark = f'SWC-{name}-{rng.randrange(10**6):06d}'
            text = other_corpus_text(int(size * CHARS_PER_TOKEN), skip={'B': 0, 'C': 9}[name]) if name != 'A' else prompt_with_needle(size, mark)
            m = [dict(role='user', content=text + f'{chr(10)}{chr(10)}The reference code is {mark}. Say the reference code and one word about the text.')]
            r, _ = chat(base, SWIFT, m, max_tokens=60, key='opencode' if name != 'C' else 'python', effort='none')
            m.append(dict(role='assistant', content=r.get('full') or ''))
            convs[name] = (m, mark, r)
        out['cold'] = {n: dict(prompt_tokens=v[2].get('prompt_tokens'), cached=v[2].get('cached'), prompt_ms=v[2].get('prompt_ms'), own_marker=v[1] in (v[2].get('full') or '')) for n, v in convs.items()}
        out['rss_after_three_conversations_mib'] = rss_mib()
        out['host_after'] = host_ram()
        back = {}
        for name in ('A', 'B', 'C', 'A'):
            m, mark, _ = convs[name]
            r, _ = chat(base, SWIFT, m + [dict(role='user', content='Repeat the reference code only.')], max_tokens=40, key='opencode', effort='none')
            back.setdefault(name, []).append(dict(prompt_tokens=r.get('prompt_tokens'), cached=r.get('cached'), prompt_ms=r.get('prompt_ms'), own_marker=mark in (r.get('full') or ''),
                                                  foreign_marker=any(o in (r.get('full') or '') for n2, (_, o, _) in convs.items() if n2 != name)))
        out['resumed'] = back
        out['all_restored'] = all((v['cached'] or 0) > 0.8 * (v['prompt_tokens'] or 1) for vs in back.values() for v in vs)
        out['no_foreign_marker'] = not any(v['foreign_marker'] for vs in back.values() for v in vs)
        out['rss_peak_mib'] = rss_mib()
        out['vram'] = nvidia()
        return out

    plan = [(1, '1_apply_and_mtp', s1), (2, '2_short_answers', s2), (3, '3_mtp_acceptance', s3), (4, '4_coding_16k', s4), (5, '5_multi_turn_cache', s5),
            (6, '6_interleaved_clients', s6), (7, '7_coding_65k', s7), (8, '8_tool_calling', s8), (9, '9_thinking_levels', s9), (10, '10_soak', s10),
            (11, '11_failed_load_recovery', s11), (12, '12_switching', s12), (13, '13_swift_cache', s13)]
    for number, name, fn in plan:
        if number in wanted:stage(name, fn)
    apply(base, 'swift')
    chat(base, SWIFT, [dict(role='user', content='Reply with the single word: ok')], max_tokens=24)
    final = status(base)
    results['final'] = dict(state=final['model']['state'], active=final['profile']['active'], applied=final['profile']['applied'], gpu=nvidia(), llama_processes=len(llama_pids()))
    Path(args.out).write_text(json.dumps(results, indent=1, default=str))
    print('\nWROTE', args.out)


if __name__ == '__main__':main()
