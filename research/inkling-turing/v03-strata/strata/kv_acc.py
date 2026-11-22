#!/usr/bin/env python3
"""KV cache type accuracy probe: greedy 128 tokens on the shared 8K/32K prompts with top-2 logprobs, for one KV type. usage: kv_acc.py KVTYPE"""
import json, os, subprocess, sys, time, http.client
kt = sys.argv[1]
R = os.path.expanduser("~/llm/experiments/inkling-kernels"); B = R + "/bin-hk"; M = R + "/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf"
SA = os.path.expanduser("~/llm/experiments/strata-ab")
p = subprocess.Popen([B + "/llama-server", "-m", M, "-ngl", "99", "-fa", "on", "-ctk", kt, "-ctv", kt, "-c", "40960", "-ub", "512", "-np", "1", "--port", "18093", "--no-warmup"],
                     env=dict(os.environ, LD_LIBRARY_PATH=B), stdout=open("%s/results/kvacc_%s.log" % (SA, kt), "w"), stderr=subprocess.STDOUT)
for _ in range(200):
    try:
        h = http.client.HTTPConnection("127.0.0.1", 18093, timeout=2); h.request("GET", "/health")
        if h.getresponse().status == 200: break
    except Exception: pass
    time.sleep(1)
out = {}
for n in (8192, 32768):
    ids = json.load(open(SA + "/prompt_%d.json" % n))["a"]
    body = dict(prompt=ids, n_predict=128, temperature=0, top_k=1, cache_prompt=False, n_probs=2, stream=False, return_tokens=True)
    h = http.client.HTTPConnection("127.0.0.1", 18093, timeout=300); t0 = time.time()
    h.request("POST", "/completion", json.dumps(body), {"Content-Type": "application/json"}); d = json.loads(h.getresponse().read())
    probs = d.get("completion_probabilities", [])
    out[n] = dict(tokens=d.get("tokens"), top=[[(c["id"], round(c["logprob"], 4)) for c in pr["top_logprobs"][:2]] for pr in probs] if probs and "top_logprobs" in probs[0] else None, secs=round(time.time() - t0, 1), prompt_ms=d["timings"]["prompt_ms"], decode_tps=d["timings"]["predicted_per_second"])
    print(kt, n, "tokens", len(out[n]["tokens"] or []), "decode_tps", out[n]["decode_tps"], flush=True)
p.terminate()
json.dump(out, open("%s/results/kvacc_%s.json" % (SA, kt), "w"))
