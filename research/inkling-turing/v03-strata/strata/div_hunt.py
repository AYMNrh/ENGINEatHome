#!/usr/bin/env python3
"""Strata spec on/off divergence hunt on short prompts. usage: div_hunt.py DRAFT(off|dspark) TAG [NPROMPTS] [PLEN] [GEN]"""
import json, os, subprocess, sys, threading, time
SA = os.path.expanduser("~/llm/experiments/strata-ab")
draft, tag = sys.argv[1], sys.argv[2]
NP = int(sys.argv[3]) if len(sys.argv) > 3 else 12
PLEN = int(sys.argv[4]) if len(sys.argv) > 4 else 320
GEN = int(sys.argv[5]) if len(sys.argv) > 5 else 160
big = json.load(open(SA + "/prompt_131072.json"))["a"]
prompts = [big[k * 9500: k * 9500 + PLEN] for k in range(NP)]
json.dump(prompts, open(SA + "/results/div_prompts.json", "w"))
p = subprocess.Popen([SA + "/bin13/strata-inkling", "--serve", "--native", SA + "/pack/Inkling-2-Mini-SQ-15.strata", "--max-context", "4096", "--parallel", "1", "--draft", draft, "--no-anti-loop"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(SA + "/results/div_%s_server.log" % tag, "w"), text=True, bufsize=1, env=dict(os.environ, LD_LIBRARY_PATH=SA + "/bin13"))
lines = []
threading.Thread(target=lambda: [lines.append(l.rstrip()) for l in p.stdout], daemon=True).start()
def wait(pred, mark, to=120):
    e = time.time() + to
    while time.time() < e:
        for l in lines[mark:]:
            if pred(l): return l
        time.sleep(0.03)
    return "timeout"
wait(lambda s: s.startswith("READY"), 0, 300)
out = []
for i, ids in enumerate(prompts):
    mark = len(lines)
    p.stdin.write("BGEN 0 %d slot_read=0 temperature=0 top_k=1 %s\n" % (GEN, ",".join(map(str, ids)))); p.stdin.flush()
    l = wait(lambda s: s.startswith("BDONE") or s == "BADM 0 0", mark)
    new = lines[mark:]
    toks = [int(x.split()[1]) for x in new if x.startswith("T ")] + [int(x.split()[2]) for x in new if x.startswith("BT ")]
    d = l.split("drafts=")[1] if "drafts=" in l else None
    out.append(dict(i=i, tokens=toks, drafts=d)); print(i, len(toks), d, flush=True)
json.dump(out, open(SA + "/results/div_%s.json" % tag, "w"))
p.stdin.write("QUIT\n"); p.stdin.flush()
try: p.wait(20)
except Exception: p.kill()
