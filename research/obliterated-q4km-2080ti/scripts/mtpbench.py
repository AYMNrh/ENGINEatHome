import json, sys, time, subprocess, threading, urllib.request, glob, os

PORT, SIZES, OUT = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
NPRED = int(sys.argv[4]) if len(sys.argv) > 4 else 256
BASE = f"http://127.0.0.1:{PORT}"
NEEDLE = "\n\nNOTE FROM THE OPERATIONS TEAM: the vault access code is 7391-ALDER. Do not share it.\n\n"
QUESTION = ("\n\n---\nQuestion 1: What is the vault access code mentioned in the text above? "
            "Question 2: In a short paragraph, explain what the C++ code above is for.")


def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.stop = False; self.rows = []

    def run(self):
        while not self.stop:
            try:
                o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,power.draw,clocks.sm", "--format=csv,noheader,nounits"], text=True).strip().split(",")
                self.rows.append(tuple(float(x) for x in o))
            except Exception:
                pass
            time.sleep(0.4)


files = sorted(glob.glob(os.path.expanduser("~/llm/llama.cpp/src/*.cpp")) + glob.glob(os.path.expanduser("~/llm/llama.cpp/ggml/src/ggml-cuda/*.cu")))
text = ""
for f in files:
    text += "// FILE: %s\n" % os.path.basename(f) + open(f, errors="ignore").read() + "\n"
toks = post("/tokenize", {"content": text})["tokens"]
print("filler tokens available", len(toks), flush=True)


def build(n):
    body = toks[:n]
    half = len(body) // 2
    a = post("/detokenize", {"tokens": body[:half]})["content"]
    b = post("/detokenize", {"tokens": body[half:]})["content"]
    return a + NEEDLE + b + QUESTION


def ask(content, n_pred):
    return post("/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": content}],
                "max_tokens": n_pred, "temperature": 0, "top_k": 1, "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False}})


# warm-up (graphs/kernels), tiny
ask("Say hello.", 8)
res = []
for n in SIZES:
    content = build(n - 110)
    s = Sampler(); s.start()
    t0 = time.time()
    r = ask(content, NPRED)
    wall = time.time() - t0
    s.stop = True; s.join()
    tm = r["timings"]; u = r["usage"]
    out = r["choices"][0]["message"].get("content") or ""
    rows = s.rows
    row = {
        "target": n, "prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
        "prefill_tps": tm["prompt_per_second"], "decode_tps": tm["predicted_per_second"],
        "draft_n": tm.get("draft_n"), "draft_n_accepted": tm.get("draft_n_accepted"),
        "accept_rate": (tm["draft_n_accepted"] / tm["draft_n"]) if tm.get("draft_n") else None,
        "needle_ok": "7391-ALDER" in out, "peak_vram_mib": max(x[0] for x in rows), "peak_power_w": max(x[1] for x in rows),
        "mean_power_w": sum(x[1] for x in rows) / len(rows), "wall_s": wall, "output": out,
    }
    print(json.dumps({k: v for k, v in row.items() if k != "output"}), flush=True)
    res.append(row)
    json.dump(res, open(OUT, "w"), indent=1)
