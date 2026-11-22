#!/usr/bin/env python3
"""Sandboxed coding-agent evaluation harness (v2).
Every task runs in a throw-away docker container: non-root uid 1000, no network, all capabilities dropped, read-only rootfs, memory/CPU/pid/file-size limits,
only the task workspace is mounted. The model gets a real bash (git, sed, pipes, chaining, pytest...). File tools operate on the mounted workspace.
Complete tool-call logs are written per task. Failures are classified, never merged:
  invalid_json / schema (model output), truncated (finish_reason=length), policy_refusal (path escapes the workspace),
  cmd_nonzero (normal development feedback, not a tool error), cmd_timeout, backend_parse_error / transport (server or network), task_timeout, context_exhausted.
Usage: pilot2.py --label L --port P [--model M] [--cred python] [--ctx N] [--tasks a,b] | --validate | --selftest"""
import argparse, http.client, json, os, re, shlex, shutil, socket, subprocess, sys, tempfile, time, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot, pilot_real

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE = "pilot-sbx:1"
WSROOT = os.path.join(HERE, "ws")
SYSTEM = ("You are a coding agent working in a software repository mounted at /work (your current directory). You have a real bash shell in an isolated "
          "sandbox (git, sed, grep, python3, pytest are available; there is no network). The repository has one committed baseline; use git as you like. "
          "Use the tools to investigate, edit and test. Make the minimal correct change, verify with the tests, and finish with a brief plain-text summary (no tool call).")
TOOLS = [
    dict(type="function", function=dict(name="run_command", description="Run a bash command in the sandbox (cwd /work). Pipes, &&, redirects, heredocs, git, sed and python are all allowed. Returns exit code and combined output.", parameters=dict(type="object", properties=dict(command=dict(type="string")), required=["command"]))),
    dict(type="function", function=dict(name="read_file", description="Read a text file (path relative to /work).", parameters=dict(type="object", properties=dict(path=dict(type="string")), required=["path"]))),
    dict(type="function", function=dict(name="write_file", description="Create or overwrite a file (path relative to /work).", parameters=dict(type="object", properties=dict(path=dict(type="string"), content=dict(type="string")), required=["path", "content"]))),
    dict(type="function", function=dict(name="replace_in_file", description="Replace exactly one occurrence of old with new in a file.", parameters=dict(type="object", properties=dict(path=dict(type="string"), old=dict(type="string"), new=dict(type="string")), required=["path", "old", "new"]))),
    dict(type="function", function=dict(name="list_dir", description="List a directory (path relative to /work).", parameters=dict(type="object", properties=dict(path=dict(type="string")), required=["path"]))),
]
REQUIRED = {t["function"]["name"]: t["function"]["parameters"]["required"] for t in TOOLS}


class Refusal(Exception):
    pass


class Sandbox:
    def __init__(self, ws):
        self.ws = ws
        self.name = "psbx_" + uuid.uuid4().hex[:10]
        subprocess.run(["docker", "run", "-d", "--name", self.name, "--network", "none", "--user", "1000:1000", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                        "--memory", "3g", "--memory-swap", "3g", "--cpus", "3", "--pids-limit", "512", "--ulimit", "fsize=200000000", "--ulimit", "nofile=4096",
                        "--read-only", "--tmpfs", "/tmp:rw,size=512m,exec", "--tmpfs", "/home/agent:rw,size=256m,uid=1000,gid=1000", "-e", "HOME=/home/agent",
                        "-e", "PYTHONPATH=/work:/work/src", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "GIT_CONFIG_GLOBAL=/home/agent/.gitconfig",
                        "-v", ws + ":/work:rw", "-w", "/work", IMAGE, "sleep", "infinity"], check=True, capture_output=True)
        self.exec("git config --global user.email agent@sandbox && git config --global user.name agent && git config --global --add safe.directory /work", 20)

    def exec(self, cmd, timeout=120):
        t0 = time.time()
        try:
            r = subprocess.run(["docker", "exec", "-w", "/work", self.name, "timeout", "-k", "5", str(timeout), "bash", "-c", cmd], capture_output=True, text=True, timeout=timeout + 20, errors="replace")
            out = r.stdout + r.stderr
            rc = r.returncode
        except subprocess.TimeoutExpired:
            out, rc = "", 124
        return rc, out, time.time() - t0

    def close(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


def safe(ws, rel):
    p = os.path.realpath(os.path.join(ws, rel or "."))
    if not (p == ws or p.startswith(ws + os.sep)):
        raise Refusal("path outside workspace")
    return p


def clip(s, n=8000):
    return s if len(s) <= n else s[: n // 2] + "\n...[%d chars omitted]...\n" % (len(s) - n) + s[-n // 2:]


def run_tool(sb, name, args):
    """Returns (category, text). category in ok|cmd_nonzero|cmd_timeout|schema|policy_refusal|tool_error"""
    if name not in REQUIRED:
        return "schema", "error: unknown tool %s" % name
    if not isinstance(args, dict) or any(k not in args or not isinstance(args[k], str) for k in REQUIRED[name]):
        return "schema", "error: missing or non-string arguments (required: %s)" % ", ".join(REQUIRED[name])
    ws = sb.ws
    try:
        if name == "run_command":
            rc, out, _ = sb.exec(args["command"], 120)
            if rc == 124:
                return "cmd_timeout", clip("exit=124 (timed out after 120 s)\n" + out)
            return ("ok" if rc == 0 else "cmd_nonzero"), clip("exit=%d\n%s" % (rc, out))
        if name == "list_dir":
            p = safe(ws, args["path"])
            if not os.path.isdir(p):
                return "tool_error", "error: not a directory"
            return "ok", "\n".join(sorted(e + ("/" if os.path.isdir(os.path.join(p, e)) else "") for e in os.listdir(p) if e != "__pycache__"))
        if name == "read_file":
            p = safe(ws, args["path"])
            if not os.path.isfile(p):
                return "tool_error", "error: no such file"
            return "ok", clip(open(p, encoding="utf-8", errors="replace").read(), 20000)
        if name == "write_file":
            p = safe(ws, args["path"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w", encoding="utf-8").write(args["content"])
            return "ok", "wrote %d bytes" % len(args["content"])
        if name == "replace_in_file":
            p = safe(ws, args["path"])
            if not os.path.isfile(p):
                return "tool_error", "error: no such file"
            s = open(p, encoding="utf-8").read()
            if s.count(args["old"]) != 1:
                return "tool_error", "error: old text must occur exactly once (found %d)" % s.count(args["old"])
            open(p, "w", encoding="utf-8").write(s.replace(args["old"], args["new"]))
            return "ok", "replaced"
    except Refusal as e:
        return "policy_refusal", "error: %s" % e
    except Exception as e:
        return "tool_error", "error: %s" % str(e)[:300]


def make_workspace(spec):
    ws = os.path.realpath(tempfile.mkdtemp(prefix="w_", dir=WSROOT))
    pilot_real.git_archive(spec["repo"], spec["commit"] + "^", ws)
    if spec["repo"] == "humanize":
        open(os.path.join(ws, "src/humanize/_version.py"), "w").write("__version__ = '0.0.0'\n")
    g = lambda *a: subprocess.run(["git", "-C", ws, "-c", "user.email=a@b", "-c", "user.name=base", *a], capture_output=True, check=True)
    g("init", "-q"); g("add", "-A"); g("commit", "-q", "-m", "baseline")
    os.chmod(ws, 0o777)
    return ws


def grade(sb, spec):
    for p in spec["hidden_from_commit"]:
        full = os.path.join(sb.ws, p); os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "wb").write(pilot_real.git_show(spec["repo"], spec["commit"], p))
    for p, txt in spec["hidden_extra"].items():
        open(os.path.join(sb.ws, p), "w").write(txt)
    rc, out, _ = sb.exec("python -m pytest -q -p no:cacheprovider --tb=no " + " ".join(shlex.quote(t) for t in spec["targets"]), 240)
    g = lambda w: int(m.group(1)) if (m := re.search(r"(\d+) %s" % w, out)) else 0
    return g("passed"), g("failed") + g("error"), out[-300:]


def run_task(a, token, name, outdir):
    spec = pilot_real.TASKS[name]
    ws = make_workspace(spec)
    sb = Sandbox(ws)
    t_start = time.time(); deadline = t_start + a.task_timeout
    msgs = [dict(role="system", content=SYSTEM), dict(role="user", content=spec["prompt"].replace("python3 -m pytest", "python -m pytest"))]
    base = os.path.join(outdir, "%s_%s" % (a.label, name))
    tlog, tr = open(base + ".toollog.jsonl", "w"), open(base + ".transcript.jsonl", "w")
    cat = {k: 0 for k in ("ok", "cmd_nonzero", "cmd_timeout", "schema", "invalid_json", "policy_refusal", "tool_error", "truncated_calls")}
    turns, status, final, trunc_retries, last_ctx = [], "turn_limit", "", 0, 0
    for turn in range(a.max_turns):
        if last_ctx + a.max_tokens + 2048 > a.ctx:
            status = "context_exhausted"; break
        body = dict(model=a.model, messages=msgs, tools=TOOLS, temperature=0, seed=0, max_tokens=a.max_tokens, stream=True, stream_options=dict(include_usage=True))
        try:
            r = pilot.chat(a.host, a.port, token, body, deadline)
        except TimeoutError:
            status = "task_timeout"; turns.append(dict(turn=turn, aborted="deadline")); break
        except Exception as e:
            msg = str(e)
            status = "backend_parse_error" if "Failed to parse tool call" in msg else "transport_error"
            turns.append(dict(turn=turn, error=msg[:500])); break
        tm, us = r["timings"] or {}, r["usage"] or {}
        pn, cn = tm.get("prompt_n", us.get("prompt_tokens")) or 0, tm.get("cache_n") or 0
        last_ctx = pn + cn + (tm.get("predicted_n") or 0)
        turns.append(dict(turn=turn, ttft_s=r["ttft"], wall_s=r["wall"], prompt_n=pn, cache_n=cn, predicted_n=tm.get("predicted_n", us.get("completion_tokens")),
                          prompt_tps=tm.get("prompt_per_second"), decode_tps=tm.get("predicted_per_second"), draft_n=tm.get("draft_n"), draft_accepted=tm.get("draft_n_accepted"),
                          reasoning_chars=len(r["reasoning"]), n_calls=len(r["calls"]), finish=r["finish"]))
        tr.write(json.dumps(dict(turn=turn, finish=r["finish"], content=r["content"], reasoning=r["reasoning"], calls=r["calls"])) + "\n"); tr.flush()
        am = dict(role="assistant", content=r["content"] or "")
        if r["reasoning"]:
            am["reasoning_content"] = r["reasoning"]
        if r["finish"] == "length":
            # truncated generation: never execute partial tool calls
            cat["truncated_calls"] += len(r["calls"]) or 1
            tlog.write(json.dumps(dict(turn=turn, kind="truncated", finish="length", partial_calls=r["calls"], predicted_n=tm.get("predicted_n"))) + "\n")
            trunc_retries += 1
            if trunc_retries > 2:
                status = "truncated_output"; break
            msgs += [dict(role="assistant", content=r["content"] or "(response cut off at the token limit)"), dict(role="user", content="Your previous response was cut off at the output token limit. Think less and continue with a short, complete tool call or the final answer.")]
            continue
        if r["calls"]:
            for c in r["calls"]:
                c["id"] = c["id"] or "call_" + uuid.uuid4().hex[:8]
            am["tool_calls"] = [dict(id=c["id"], type="function", function=dict(name=c["name"], arguments=c["args"])) for c in r["calls"]]
        msgs.append(am)
        if not r["calls"]:
            final = r["content"]; status = "completed" if r["finish"] == "stop" else "ended_%s" % r["finish"]; break
        for c in r["calls"]:
            t0 = time.time()
            try:
                args = json.loads(c["args"] or "{}"); k, out = run_tool(sb, c["name"], args)
            except json.JSONDecodeError as e:
                k, out = "invalid_json", "error: arguments are not valid JSON (%s)" % e
            cat[k] += 1
            tlog.write(json.dumps(dict(turn=turn, kind="call", id=c["id"], name=c["name"], args=c["args"], category=k, seconds=round(time.time() - t0, 3), result=out[:20000])) + "\n"); tlog.flush()
            msgs.append(dict(role="tool", tool_call_id=c["id"], content=out))
        if time.time() > deadline:
            status = "task_timeout"; break
    wall = time.time() - t_start
    passed, failed, tail = grade(sb, spec)
    sb.close(); tlog.close(); tr.close()
    shutil.rmtree(ws, ignore_errors=True)
    ref = spec["ref_passed"]
    done = [t for t in turns if t.get("prompt_n") is not None and "error" not in t]
    later = done[1:]
    cached = sum(t["cache_n"] for t in later); allp = sum(t["cache_n"] + t["prompt_n"] for t in later)
    calls = sum(cat[k] for k in ("ok", "cmd_nonzero", "cmd_timeout", "schema", "invalid_json", "policy_refusal", "tool_error"))
    model_err = cat["schema"] + cat["invalid_json"] + cat["truncated_calls"]
    dec = [t for t in done if t.get("predicted_n") and t.get("decode_tps")]
    return dict(label=a.label, task=name, functional_pass=bool(failed == 0 and passed >= ref), tests=dict(passed=passed, failed=failed, ref=ref), status=status,
                clean_completion=bool(status == "completed" and failed == 0 and passed >= ref and model_err == 0),
                wall_s=round(wall, 1), turns=len(done), generated_tokens=sum(t.get("predicted_n") or 0 for t in done), reasoning_chars=sum(t.get("reasoning_chars", 0) for t in done),
                tool_calls=calls, tool_categories=cat, model_side_errors=model_err, tool_error_rate=round(model_err / max(1, calls + cat["truncated_calls"]), 3),
                cache=dict(prompt_new=sum(t["prompt_n"] for t in done), cached=sum(t["cache_n"] for t in done), reuse_turns2plus=round(cached / allp, 3) if allp else None,
                           cold_ttft_s=done[0]["ttft_s"] if done else None, median_ttft_s=sorted(t["ttft_s"] for t in done if t.get("ttft_s"))[len(done) // 2] if done else None),
                decode_tps=round(sum(t["predicted_n"] for t in dec) / max(1e-9, sum(t["predicted_n"] / t["decode_tps"] for t in dec)), 1) if dec else None,
                max_ctx_used=max((t["prompt_n"] + t["cache_n"] + (t.get("predicted_n") or 0) for t in done), default=0), final_answer=final[:500], per_turn=turns)


def validate():
    os.makedirs(WSROOT, exist_ok=True); ok = True; refs = {}
    for name, spec in pilot_real.TASKS.items():
        spec["ref_passed"] = 0
        ws = make_workspace(spec); sb = Sandbox(ws)
        p0, f0, _ = grade(sb, spec)
        for f in spec["fix_files"]:
            open(os.path.join(ws, f), "wb").write(pilot_real.git_show(spec["repo"], spec["commit"], f))
        p1, f1, tail = grade(sb, spec)
        rc, out, _ = sb.exec("git status --short | head -5; git log --oneline | head -3; id -u; curl -s -m 3 http://1.1.1.1 || echo no-network; touch /etc/x 2>&1 | head -1; sed -n 1,2p /etc/hostname | wc -l", 30)
        print(name, "parent", (p0, f0), "fixed", (p1, f1), "|", out.replace("\n", " ; ")[:200])
        refs[name] = p1; ok &= f0 > 0 and f1 == 0 and p1 > 0
        sb.close(); shutil.rmtree(ws, ignore_errors=True)
    json.dump(refs, open(os.path.join(HERE, "ref_counts_sbx.json"), "w"))
    print("refs", refs); return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label"); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int); ap.add_argument("--model", default="local"); ap.add_argument("--cred")
    ap.add_argument("--task-timeout", type=float, default=200); ap.add_argument("--max-turns", type=int, default=40); ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--ctx", type=int, default=65536, help="server context; the harness stops with context_exhausted before overflow")
    ap.add_argument("--session-budget", type=float, default=780); ap.add_argument("--tasks", default=",".join(pilot_real.TASKS)); ap.add_argument("--out", default="results_sbx")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    os.makedirs(WSROOT, exist_ok=True)
    if a.validate:
        sys.exit(validate())
    refs = json.load(open(os.path.join(HERE, "ref_counts_sbx.json")))
    for k, v in refs.items():
        pilot_real.TASKS[k]["ref_passed"] = v
    token = json.load(open(os.path.expanduser("~/llm/serving/credentials.json")))[a.cred] if a.cred else None
    os.makedirs(a.out, exist_ok=True)
    t0, results = time.time(), []
    for name in a.tasks.split(","):
        if time.time() - t0 > a.session_budget:
            results.append(dict(label=a.label, task=name, status="skipped_budget")); print("SKIP", name, flush=True); continue
        print("START", name, "at +%ds" % (time.time() - t0), flush=True)
        r = run_task(a, token, name, a.out); results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "per_turn"}), flush=True)
        json.dump(results, open(os.path.join(a.out, "%s_results.json" % a.label), "w"), indent=1)
    print("session_s", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
