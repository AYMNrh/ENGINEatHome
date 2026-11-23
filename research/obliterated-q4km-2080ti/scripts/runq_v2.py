#!/usr/bin/env python3
"""Serialized experiment queue. usage: runq.py queue.json   (detach with setsid nohup)
Each experiment: {name, server:[args], ctx, runs:[{label,start,turns,gen,depth}]}"""
import json, os, subprocess, sys, time, signal, fcntl, urllib.request

W = os.path.expanduser("~/llm/experiments/obliterated-opt")
BIN = os.path.expanduser("~/llm/llama.cpp/build/bin/llama-server")
MODEL = os.path.expanduser("~/llm/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf")
PORT = "18090"
BASE_ARGS = ["--parallel", "1", "--n-gpu-layers", "99", "--flash-attn", "on", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
             "--batch-size", "512", "--ubatch-size", "128", "--threads", "8", "--fit", "off", "--no-context-shift", "--cache-ram", "0",
             "--ctx-checkpoints", "4", "--no-webui", "--jinja", "--no-warmup"]


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()


def log(m):
    open(f"{W}/runq.log", "a").write(time.strftime("%H:%M:%S ") + m + "\n")


def gate():
    assert sh("systemctl is-active engineathome-api.service") != "active", "gateway still active"
    assert not sh("ps -eo args | grep '[l]lama-server'"), "another llama-server running"
    used = int(sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"))
    assert used < 600, f"GPU not idle {used}"
    assert "175" in sh("nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits"), "power limit not 175 W"


def merge(base, extra):
    """extra flags override same-named flags in base"""
    out, i = list(base), 0
    while i < len(extra):
        f = extra[i]
        has_val = i + 1 < len(extra) and not extra[i + 1].startswith("--")
        if f in out:
            j = out.index(f)
            if has_val and j + 1 < len(out) and not out[j + 1].startswith("--"):
                out[j + 1] = extra[i + 1]
            i += 2 if has_val else 1
            continue
        out += [f] + ([extra[i + 1]] if has_val else [])
        i += 2 if has_val else 1
    return out


def run_exp(e):
    gate()
    args = merge(BASE_ARGS, e.get("server", []))
    cmd = [BIN, "--model", MODEL, "--ctx-size", str(e.get("ctx", 65536)), "--host", "127.0.0.1", "--port", PORT] + args
    json.dump({"name": e["name"], "cmd": cmd, "runs": e["runs"], "time": time.strftime("%FT%T")}, open(f"{W}/configs/{e['name']}.json", "w"), indent=1)
    lf = open(f"{W}/raw/server_{e['name']}.log", "w")
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    log(f"start {e['name']} pid {p.pid}")
    ok = False
    for _ in range(150):
        if p.poll() is not None:
            break
        try:
            if b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read():
                ok = True; break
        except Exception:
            pass
        time.sleep(2)
    try:
        if not ok:
            log(f"FAILED to start {e['name']} (rc={p.poll()}) -- see raw/server_{e['name']}.log")
            open(f"{W}/ledger.jsonl", "a").write(json.dumps({"label": e["name"], "failed_to_start": True, "ts": time.strftime("%FT%T")}) + "\n")
            return
        time.sleep(2)
        log(f"vram after load {sh('nvidia-smi --query-gpu=memory.used --format=csv,noheader')}")
        for r in e["runs"]:
            if r.get("kind") == "side":
                c = ["python3", f"{W}/scripts/sidetest.py", PORT, str(r["start"]), f"{e['name']}:{r['label']}", f"{W}/ledger_side.jsonl"]
            else:
                c = ["python3", f"{W}/scripts/agentsim2.py", PORT, str(r["start"]), str(r["turns"]), str(r["gen"]), f"{e['name']}:{r['label']}",
                     f"{W}/ledger.jsonl", "-" if r.get("depth") is None else str(r["depth"]), r.get("sampling", "greedy")]
            log("run " + " ".join(c[3:]))
            res = subprocess.run(c, capture_output=True, text=True)
            open(f"{W}/raw/{e['name']}_{r['label']}.out", "w").write(res.stdout + res.stderr)
            if res.returncode:
                log(f"run error rc={res.returncode}: {res.stderr[-300:]}")
    finally:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
        lf.close()
        time.sleep(4)
        log(f"stopped {e['name']}; vram {sh('nvidia-smi --query-gpu=memory.used --format=csv,noheader')}")


if __name__ == "__main__":
    lock = open(f"{W}/gpu.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for e in json.load(open(sys.argv[1])):
        try:
            run_exp(e)
        except AssertionError as ex:
            log(f"GATE REFUSED {e['name']}: {ex}")
            break
    log("queue done")
