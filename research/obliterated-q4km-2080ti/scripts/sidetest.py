#!/usr/bin/env python3
"""Does an unrelated short request (e.g. OpenCode title/summary call) evict the main conversation's cache?
usage: sidetest.py PORT START LABEL LEDGER"""
import json, sys, time, urllib.request, glob, os

PORT, START, LABEL, LEDGER = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
BASE = f"http://127.0.0.1:{PORT}"


def post(path, body, timeout=3600):
    return json.load(urllib.request.urlopen(urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=timeout))


files = sorted(glob.glob(os.path.expanduser("~/llm/llama.cpp/src/*.cpp")))
text = "".join("// FILE: %s\n%s\n" % (os.path.basename(f), open(f, errors="ignore").read()) for f in files)
toks = post("/tokenize", {"content": text})["tokens"]
det = lambda a, b: post("/detokenize", {"tokens": toks[a:b]})["content"]
msgs = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "Context:\n" + det(0, START - 300) + "\n\nTask: describe the main data structures briefly."}]
cursor = START - 300
nothink = {"enable_thinking": False}


def call(m, gen, tag, **kw):
    t0 = time.time()
    r = post("/v1/chat/completions", {"model": "x", "messages": m, "max_tokens": gen, "temperature": 0, "top_k": 1, "cache_prompt": True, "chat_template_kwargs": nothink, **kw})
    tm = r["timings"]
    row = {"label": LABEL, "step": tag, "prompt_tokens": r["usage"]["prompt_tokens"], "cache_n": tm.get("cache_n"), "prompt_n": tm.get("prompt_n"),
           "ttft_ms": round(tm["prompt_ms"], 0), "decode_tps": round(tm["predicted_per_second"], 2), "wall_s": round(time.time() - t0, 1)}
    open(LEDGER, "a").write(json.dumps(row) + "\n"); print(json.dumps(row), flush=True)
    return r["choices"][0]["message"].get("content") or ""


msgs.append({"role": "assistant", "content": call(msgs, 200, "turn0_cold")})
for k in (1, 2):
    msgs.append({"role": "user", "content": "Tool result:\n" + det(cursor, cursor + 1500) + "\n\nTask: summarise in 3 bullets."}); cursor += 1500
    msgs.append({"role": "assistant", "content": call(msgs, 200, f"turn{k}_append")})
call([{"role": "user", "content": "Generate a 3-word title for a conversation about C++ code review."}], 12, "SIDE_title")
msgs.append({"role": "user", "content": "Tool result:\n" + det(cursor, cursor + 1500) + "\n\nTask: summarise in 3 bullets."}); cursor += 1500
msgs.append({"role": "assistant", "content": call(msgs, 200, "turn3_after_side")})
call([{"role": "user", "content": "Generate a 3-word title for a conversation about C++ code review."}], 12, "SIDE_title2")
msgs.append({"role": "user", "content": "Tool result:\n" + det(cursor, cursor + 1500) + "\n\nTask: summarise in 3 bullets."})
call(msgs, 200, "turn4_after_side2")
