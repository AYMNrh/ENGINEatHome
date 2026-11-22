#!/usr/bin/env python3
"""Allocation probe: start llama-server (our e0be55a build) with given -c/-ub/KV type, record buffer sizes and nvidia-smi used, no prompts."""
import json, os, re, subprocess, sys, time, http.client
R = os.path.expanduser("~/llm/experiments/inkling-kernels"); B = R + "/bin-hk"; M = R + "/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf"
OUT = os.path.expanduser("~/llm/experiments/strata-ab/results"); os.makedirs(OUT, exist_ok=True)
res = []
for spec in sys.argv[1:]:
    c, ub, kt = spec.split(":"); log = "%s/memprobe_%s.log" % (OUT, spec.replace(":", "_"))
    p = subprocess.Popen([B + "/llama-server", "-m", M, "-ngl", "99", "-fa", "on", "-ctk", kt, "-ctv", kt, "-c", c, "-ub", ub, "-np", "1", "--port", "18092", "--no-warmup"],
                         env=dict(os.environ, LD_LIBRARY_PATH=B), stdout=open(log, "w"), stderr=subprocess.STDOUT)
    ok, used, t0 = False, None, time.time()
    while time.time() - t0 < 120:
        if p.poll() is not None: break
        try:
            h = http.client.HTTPConnection("127.0.0.1", 18092, timeout=2); h.request("GET", "/health")
            if h.getresponse().status == 200:
                ok = True; time.sleep(1)
                used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()); break
        except Exception: pass
        time.sleep(1)
    p.terminate()
    try: p.wait(30)
    except Exception: p.kill()
    txt = open(log).read()
    bufs = [re.sub(r"^\S+\s+", "", l)[:150] for l in txt.splitlines() if re.search(r"buffer size|compute buffer|KV buffer|RS buffer|failed to allocate|out of memory", l)]
    r = dict(spec=spec, loaded=ok, used_mib=used, buffers=bufs[:14]); res.append(r); print(json.dumps(r), flush=True)
json.dump(res, open(OUT + "/memprobe.json", "w"), indent=1)
