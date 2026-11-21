#!/usr/bin/env python3
"""Decode-path equivalence: greedy generations with per-token top logprobs from one llama-server build.
usage: k_decode_check.py run <bindir> <out.json> [env=val ...]   |   k_decode_check.py cmp <ref.json> <new.json>"""
import json, os, subprocess, sys, time, urllib.request

R = os.path.expanduser('~/llm/experiments/inkling-kernels')
M = R + '/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf'
PROMPTS = [
    "def quicksort(arr):\n    \"\"\"Sort a list of integers.\"\"\"\n",
    "The three most important ideas in thermodynamics are",
    "Q: A train leaves at 9:40 and arrives at 13:05. How long is the trip?\nA:",
    "#include <stdio.h>\n\nint main(void) {\n",
]
PORT = 18131
if os.environ.get('LONG_CHECK'):
    filler = ' '.join(f'Record {i}: the warehouse in sector {i % 97} stores {(i * 7919) % 1000} crates of item {(i * 31) % 211}.' for i in range(420))
    PROMPTS = [filler + '\n\nQuestion: how many crates does record 17 list, and which sector is it in?\nAnswer:',
               filler + '\n\nWrite a short Python function that sums the crates of records 3, 5 and 8.\n```python\n']


def post(path, obj, timeout=600):
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}', data=json.dumps(obj).encode(), headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def run(bindir, out, envs):
    env = dict(os.environ, LD_LIBRARY_PATH=bindir)
    for e in envs:
        k, v = e.split('=', 1); env[k] = v
    log = open(out + '.server.log', 'w')
    p = subprocess.Popen([bindir + '/llama-server', '-m', M, '-ngl', '99', '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0', '-c', '16384',
                          '-ub', '512', '--parallel', '1', '--host', '127.0.0.1', '--port', str(PORT)], env=env, stdout=log, stderr=log)
    try:
        for _ in range(300):
            try:
                if json.load(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2)).get('status') == 'ok':
                    break
            except Exception:
                pass
            if p.poll() is not None:
                raise SystemExit('server died')
            time.sleep(1)
        res = []
        for pr in PROMPTS:
            r = post('/completion', dict(prompt=pr, n_predict=48, temperature=0, top_k=1, n_probs=3, cache_prompt=False, seed=1))
            toks = [x['id'] for x in r['completion_probabilities']]
            lps = [x.get('logprob', x.get('top_logprobs', [{}])[0].get('logprob', 0.0)) for x in r['completion_probabilities']]
            res.append(dict(prompt=pr, text=r['content'], tokens=toks, logprobs=lps, tps=r['timings']['predicted_per_second']))
        json.dump(res, open(out, 'w'))
        print('decode tok/s per prompt:', [round(x['tps'], 1) for x in res])
    finally:
        p.terminate(); p.wait(timeout=30)


def cmp(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    worst = 0.0; ok = True
    for x, y in zip(A, B):
        n = min(len(x['tokens']), len(y['tokens']))
        first = next((i for i in range(n) if x['tokens'][i] != y['tokens'][i]), None)
        upto = first if first is not None else n
        d = max((abs(x['logprobs'][i] - y['logprobs'][i]) for i in range(upto)), default=0.0)
        worst = max(worst, d)
        status = 'identical' if first is None else f'diverges at token {first}/{n}'
        ok = ok and first is None
        print(f'  {status:24s} max |dlogprob| before divergence {d:.2e}   {x["prompt"][:40]!r}')
    print('RESULT:', 'IDENTICAL TOKENS' if ok else 'TOKENS DIFFER', f'(max logprob delta {worst:.2e})')


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        cmp(sys.argv[2], sys.argv[3])
