#!/usr/bin/env python3
"""CUDA graph / prefill-decode stress: alternating prefill and decode, mixed ubatch sizes, prefix reuse, multi-turn, short and long prompts.
usage: stress.py TAG BINDIR CTX UB [ENV=VAL ...]   -> results/stress_TAG.json (+ server log), prints crash status and graph-event counts"""
import json, os, subprocess, sys, time, http.client
R = os.path.expanduser("~/llm/experiments/inkling-kernels"); SA = os.path.expanduser("~/llm/experiments/strata-ab")
M = R + "/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf"
tag, bindir, ctx, ub = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
env = dict(os.environ, LD_LIBRARY_PATH=bindir, GGML_CUDA_GRAPH_DEBUG="1")
for e in sys.argv[5:]:
    k, v = e.split("=", 1); env[k] = v
big = json.load(open(SA + "/prompt_131072.json"))["a"]
def sl(n, k): return big[k * 4001: k * 4001 + n]
log = open("%s/results/stress_%s.log" % (R, tag), "w")
p = subprocess.Popen([bindir + "/llama-server", "-m", M, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", str(ctx), "-ub", ub, "-np", "1", "--port", "18098", "-lv", "4"],
                     env=env, stdout=log, stderr=subprocess.STDOUT)
for _ in range(180):
    try:
        h = http.client.HTTPConnection("127.0.0.1", 18098, timeout=2); h.request("GET", "/health")
        if h.getresponse().status == 200: break
    except Exception: pass
    if p.poll() is not None: break
    time.sleep(1)
lim = ctx - 200
lens = [18, 3, 120, 9, 700, 40, 1500, 18, 333, 3, 1, 9, 120, 2, 1100, 40, 18, 600, 5, 250, 18, 3, 90, 400]
npred = [8, 1, 24, 4, 16, 2, 8, 24, 4, 16, 1, 8, 4, 24, 8, 2, 16, 4, 24, 8, 1, 16, 4, 8]
res, prev = [], None
for i, (n, k) in enumerate(zip(lens, npred)):
    n = min(n, lim - 60)
    multi = i in (5, 6, 11, 12, 17) and prev is not None
    if multi:
        ids = prev["ids"] + prev["tokens"] + sl(12, i + 30)          # multi-turn: previous prompt + answer + a new tail
        cp = True
    else:
        ids = sl(n, i); cp = (i % 3 != 0)                           # alternate prompt-cache on and off
    ids = ids[: lim - 40]
    body = dict(prompt=ids, n_predict=k, temperature=0, top_k=1, n_probs=2, cache_prompt=cp, seed=1, return_tokens=True)
    try:
        h = http.client.HTTPConnection("127.0.0.1", 18098, timeout=300); h.request("POST", "/completion", json.dumps(body), {"Content-Type": "application/json"})
        d = json.loads(h.getresponse().read())
        toks = d["tokens"]; lp = [[round(c["logprob"], 5) for c in x["top_logprobs"][:2]] for x in d["completion_probabilities"]]
        res.append(dict(i=i, n_prompt=len(ids), cache=cp, multi=multi, n_pred=k, tokens=toks, lp=lp, prompt_n=d["timings"]["prompt_n"], cache_n=d["timings"].get("cache_n")))
        prev = dict(ids=ids, tokens=toks)
    except Exception as e:
        res.append(dict(i=i, n_prompt=len(ids), error=type(e).__name__)); break
alive = p.poll() is None
p.terminate()
try: p.wait(20)
except Exception: p.kill()
txt = open(log.name).read()
out = dict(tag=tag, ctx=ctx, ub=ub, env=sys.argv[5:], completed=len(res), crashed=(not alive) or any("error" in r for r in res), results=res,
           realloc_events=txt.count("sched_graph_realloc"), graph_reused=txt.count("reused"), warmup_complete=txt.count("warmup complete"), warmup_reset=txt.count("warmup reset"), cuda_error=("CUDA error" in txt))
json.dump(out, open("%s/results/stress_%s.json" % (R, tag), "w"))
print(tag, {k: v for k, v in out.items() if k != "results"}, flush=True)
