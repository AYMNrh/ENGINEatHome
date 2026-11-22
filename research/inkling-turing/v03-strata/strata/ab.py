#!/usr/bin/env python3
"""A/B driver: Strata strata-inkling (stdin line protocol) vs our llama.cpp fork (llama-server) on identical token-id prompts.
Usage: ab.py strata --draft off|dspark --model PATH --sizes 8192,32768 --ctx N --tag T
       ab.py ours --sizes 8192,32768 --ctx N --tag T"""
import argparse, http.client, json, os, subprocess, sys, threading, time

ROOT = os.path.expanduser("~/llm/experiments")
SA = ROOT + "/strata-ab"
OUT = SA + "/results"
os.makedirs(OUT, exist_ok=True)


class Vram(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.peak = 0; self.base = None; self.stop = False
    def run(self):
        while not self.stop:
            try:
                v = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip())
                self.base = v if self.base is None else self.base
                self.peak = max(self.peak, v)
            except Exception:
                pass
            time.sleep(0.4)


def load_prompts(sizes):
    return {n: json.load(open(SA + "/prompt_%d.json" % n)) for n in sizes}


def strata(a):
    prompts = load_prompts(a.sizes)
    env = dict(os.environ, LD_LIBRARY_PATH=SA + "/bin13")
    cmd = [SA + "/bin13/strata-inkling", "--serve", "--native", a.model, "--max-context", str(a.ctx), "--parallel", "1", "--draft", a.draft, "--no-anti-loop"] + (["--kv", a.kv] if a.kv else [])
    vr = Vram(); vr.start(); time.sleep(1)
    log = open("%s/strata_%s_server.log" % (OUT, a.tag), "w")
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=env, bufsize=1)
    lines = []
    def rd():
        for l in p.stdout:
            lines.append((time.time(), l.rstrip("\n")))
    threading.Thread(target=rd, daemon=True).start()
    def wait(pred, timeout, i0=0):
        end = time.time() + timeout
        while time.time() < end:
            for t, l in lines[i0:]:
                if pred(l): return t, l
            if p.poll() is not None and not any(pred(l) for _, l in lines[i0:]): return None, "exited rc=%s" % p.returncode
            time.sleep(0.05)
        return None, "timeout"
    t, l = wait(lambda s: s.startswith("READY") or s.startswith("ERR"), 300)
    res = dict(engine="strata", draft=a.draft, model=os.path.basename(a.model), load_s=round(t - t0, 1) if t else None, ready=l, runs=[])
    print("ready:", l, "load_s", res["load_s"], flush=True)
    if t and l.startswith("READY"):
        for n in a.sizes:
            for which in ("a", "b"):
                ids = prompts[n][which]
                mark = len(lines)
                ts = time.time()
                p.stdin.write("BGEN 0 %d slot_read=0 temperature=0 top_k=1 %s\n" % (a.gen, ",".join(map(str, ids)))); p.stdin.flush()
                t, l = wait(lambda s: s.startswith("BDONE") or s.startswith("ERR") or s == "BADM 0 0", a.timeout, mark)
                new = [(tt, ll) for tt, ll in lines[mark:]]
                toks = [int(ll.split()[1]) for _, ll in new if ll.startswith("T ")] + [int(ll.split()[2]) for _, ll in new if ll.startswith("BT ")]
                first = next((tt for tt, ll in new if ll.startswith("T ")), None)
                adm = next((ll for _, ll in new if ll.startswith("DONE ")), "")
                admit_prompt_ms = float(adm.split()[3]) if adm else None
                resume = [ll for _, ll in new if ll.startswith("RESUME")]
                r = dict(size=n, which=which, admit_prompt_ms=admit_prompt_ms, final=l, ttft_s=round(first - ts, 3) if first else None, total_s=round(t - ts, 2) if t else None, n_tokens=len(toks), tokens=toks, resume=resume, vram_peak_mib=vr.peak)
                if l.startswith("BDONE"):
                    w = l.split(); r["generated"] = int(w[2]); r["dec_ms"] = float(w[4]); r["prompt_ms"] = admit_prompt_ms
                    if "drafts=" in l:
                        o, ac = l.split("drafts=")[1].split(":"); r["drafts_offered"], r["drafts_accepted"] = int(o), int(ac)
                    r["decode_tps"] = round((r["generated"] - 1) / (r["dec_ms"] / 1000), 1) if r["dec_ms"] else None
                    r["prefill_tps"] = round(n / (r["prompt_ms"] / 1000), 1) if r["prompt_ms"] else None
                res["runs"].append(r)
                print({k: v for k, v in r.items() if k != "tokens"}, flush=True)
                if not (l.startswith("BDONE") or l == "BADM 0 0"): break
    try:
        p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=30)
    except Exception:
        p.kill()
    vr.stop = True
    res.update(vram_base_mib=vr.base, vram_peak_mib=vr.peak, stderr_tail=open(log.name).read()[-600:])
    json.dump(res, open("%s/strata_%s.json" % (OUT, a.tag), "w"), indent=1)


def ours(a):
    prompts = load_prompts(a.sizes)
    B = ROOT + "/inkling-kernels/bin-hk"
    M = ROOT + "/inkling-kernels/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf"
    vr = Vram(); vr.start(); time.sleep(1)
    log = open("%s/ours_%s_server.log" % (OUT, a.tag), "w")
    t0 = time.time()
    p = subprocess.Popen([B + "/llama-server", "-m", M, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", str(a.ctx), "-ub", str(a.ub), "-np", "1", "--port", "18091", "--no-warmup"],
                         env=dict(os.environ, LD_LIBRARY_PATH=B), stdout=log, stderr=log)
    ok = False
    for _ in range(300):
        try:
            c = http.client.HTTPConnection("127.0.0.1", 18091, timeout=2); c.request("GET", "/health")
            if c.getresponse().status == 200: ok = True; break
        except Exception:
            pass
        if p.poll() is not None: break
        time.sleep(1)
    res = dict(engine="ours e0be55a", model=os.path.basename(M), ub=a.ub, load_s=round(time.time() - t0, 1), ready=ok, runs=[])
    print("ready", ok, res["load_s"], flush=True)
    if ok:
        for n in a.sizes:
            for which in ("a", "b"):
                body = dict(prompt=prompts[n][which], n_predict=a.gen, temperature=0, top_k=1, cache_prompt=True, stream=True, return_tokens=True)
                ts = time.time(); c = http.client.HTTPConnection("127.0.0.1", 18091, timeout=a.timeout)
                c.request("POST", "/completion", json.dumps(body), {"Content-Type": "application/json"}); rsp = c.getresponse()
                toks, first, tm = [], None, None
                for line in rsp:
                    line = line.strip()
                    if not line.startswith(b"data:"): continue
                    d = json.loads(line[5:])
                    if d.get("tokens"):
                        first = first or time.time(); toks += d["tokens"]
                    if d.get("timings"): tm = d["timings"]
                r = dict(size=n, which=which, ttft_s=round(first - ts, 3) if first else None, total_s=round(time.time() - ts, 2), n_tokens=len(toks), tokens=toks, vram_peak_mib=vr.peak)
                if tm:
                    r.update(prompt_n=tm.get("prompt_n"), cache_n=tm.get("cache_n"), prompt_ms=tm.get("prompt_ms"), prefill_tps=round(tm.get("prompt_per_second") or 0, 1), decode_tps=round(tm.get("predicted_per_second") or 0, 1))
                res["runs"].append(r)
                print({k: v for k, v in r.items() if k != "tokens"}, flush=True)
    p.terminate()
    try: p.wait(30)
    except Exception: p.kill()
    vr.stop = True
    res.update(vram_base_mib=vr.base, vram_peak_mib=vr.peak)
    json.dump(res, open("%s/ours_%s.json" % (OUT, a.tag), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("engine"); ap.add_argument("--draft", default="off"); ap.add_argument("--model", default=SA + "/pack/Inkling-2-Mini-SQ-15.strata")
    ap.add_argument("--sizes", default="8192,32768"); ap.add_argument("--ctx", type=int, default=40960); ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--kv", default=""); ap.add_argument("--tag", default="x"); ap.add_argument("--ub", type=int, default=512); ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args(); a.sizes = [int(x) for x in a.sizes.split(",")]
    (strata if a.engine == "strata" else ours)(a)
