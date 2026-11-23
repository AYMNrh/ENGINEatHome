#!/usr/bin/env python3
"""Agent-turn simulator for llama-server (OpenCode-like): one growing conversation.
Turn 0 = cold prefill of START tokens of source-code context; each later turn appends the previous
assistant answer + a ~TAIL-token 'tool result' + an instruction, so the prefix is cached and only the tail
is prefilled. Records TTFT components, decode tok/s, draft acceptance, GPU telemetry per turn.
usage: agentsim.py PORT START_TOKENS TURNS GEN LABEL LEDGER [spec_n_max|-] [cold_only]
"""
import json, sys, time, subprocess, threading, urllib.request, glob, os, statistics, hashlib

PORT, START, TURNS, GEN, LABEL, LEDGER = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
DEPTH = None if len(sys.argv) < 8 or sys.argv[7] == "-" else int(sys.argv[7])
TAIL = 1500
BASE = f"http://127.0.0.1:{PORT}"
TASKS = [
    "Write a complete C++ function that implements an LRU cache with get/put, with comments and a short usage example.",
    "Explain in detail, step by step, how the code in the tool result above works, including edge cases and failure modes.",
    "Respond ONLY with a JSON array of 6 tool-call objects {\"tool\":...,\"args\":{...},\"reason\":...} planning how to profile this code.",
    "Produce a unified diff patch that adds input validation and error handling to the most important function above, then explain the patch.",
    "Summarise our findings so far as a bulleted list with at least 15 detailed bullets.",
    "Write unit tests for the code discussed so far, using a plain assert-based C++ test file.",
]


def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


class Telemetry:
    Q = "timestamp,memory.used,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,temperature.gpu"

    def __init__(self):
        self.rows = []
        self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,temperature.gpu,clocks_throttle_reasons.active",
                                   "--format=csv,noheader,nounits", "-lms", "300"], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._rd, daemon=True).start()

    def _rd(self):
        for line in self.p.stdout:
            try:
                v = [x.strip() for x in line.split(",")]
                self.rows.append((time.time(), *[float(x) for x in v[:7]], v[7]))
            except Exception:
                pass

    def window(self, t0, t1):
        r = [x for x in self.rows if t0 <= x[0] <= t1]
        if not r:
            return {}
        col = lambda i: [x[i] for x in r]
        return {"vram_peak_mib": max(col(1)), "power_mean_w": round(statistics.mean(col(2)), 1), "power_peak_w": max(col(2)),
                "sm_mhz_mean": round(statistics.mean(col(3))), "sm_mhz_min": min(col(3)), "mem_mhz": statistics.mode(col(4)),
                "gpu_util_mean": round(statistics.mean(col(5))), "mem_util_mean": round(statistics.mean(col(6))),
                "temp_max": max(col(7)), "throttle": sorted(set(x[8] for x in r)), "n": len(r)}


files = sorted(glob.glob(os.path.expanduser("~/llm/llama.cpp/src/*.cpp")) + glob.glob(os.path.expanduser("~/llm/llama.cpp/ggml/src/ggml-cuda/*.cu")))
text = "".join("// FILE: %s\n%s\n" % (os.path.basename(f), open(f, errors="ignore").read()) for f in files)
toks = post("/tokenize", {"content": text})["tokens"]
det = lambda a, b: post("/detokenize", {"tokens": toks[a:b]})["content"]

SYS = "You are a coding agent working in a C++ repository. Use the tool results provided to answer precisely."
msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": "Repository context (tool result: read_files):\n" + det(0, START - 400) + "\n\nTask: " + TASKS[0]}]
cursor = START - 400
tel = Telemetry()
time.sleep(1.0)
out_rows = []
for turn in range(TURNS):
    if turn > 0:
        chunk = det(cursor, cursor + TAIL); cursor += TAIL
        msgs.append({"role": "user", "content": "Tool result (read_file):\n" + chunk + "\n\nTask: " + TASKS[turn % len(TASKS)]})
    body = {"model": "x", "messages": msgs, "max_tokens": GEN, "temperature": 0, "top_k": 1, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    if DEPTH is not None:
        body["speculative.n_max"] = DEPTH
    t0 = time.time(); r = post("/v1/chat/completions", body); t1 = time.time()
    tm, u = r["timings"], r["usage"]
    out = r["choices"][0]["message"].get("content") or ""
    msgs.append({"role": "assistant", "content": out})
    ncomp = u["completion_tokens"]; dn = tm.get("draft_n") or 0; da = tm.get("draft_n_accepted") or 0
    steps = ncomp - da if dn else None
    words = out.split(); shingles = [" ".join(words[i:i + 6]) for i in range(max(0, len(words) - 6))]
    row = {"label": LABEL, "turn": turn, "task": turn % len(TASKS), "prompt_tokens": u["prompt_tokens"], "cache_n": tm.get("cache_n"), "prompt_n": tm.get("prompt_n"),
           "prefill_ms": round(tm["prompt_ms"], 1), "prefill_tps": round(tm["prompt_per_second"], 1) if tm.get("prompt_n", 0) > 64 else None,
           "completion_tokens": ncomp, "decode_tps": round(tm["predicted_per_second"], 2), "decode_ms": round(tm["predicted_ms"], 1),
           "draft_n": dn, "draft_acc": da, "accept_rate": round(da / dn, 3) if dn else None, "steps": steps,
           "tok_per_step": round(ncomp / steps, 2) if steps else None, "step_ms": round(tm["predicted_ms"] / steps, 1) if steps else round(tm["predicted_ms"] / max(ncomp, 1), 1),
           "eff_depth": round(dn / steps, 2) if steps else None, "wall_s": round(t1 - t0, 2), "finish": r["choices"][0]["finish_reason"],
           "loop_flag": bool(shingles and len(set(shingles)) < 0.5 * len(shingles)),
           "out_sha": hashlib.sha1(out.encode()).hexdigest()[:10], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    row.update(tel.window(t0, t1))
    out_rows.append(row)
    open(LEDGER, "a").write(json.dumps(row) + "\n")
    print(json.dumps({k: row[k] for k in ("label", "turn", "prompt_tokens", "cache_n", "prefill_ms", "completion_tokens", "decode_tps", "accept_rate", "tok_per_step", "step_ms", "eff_depth", "power_mean_w", "sm_mhz_mean", "gpu_util_mean", "mem_util_mean", "loop_flag")}), flush=True)
tel.p.terminate()
