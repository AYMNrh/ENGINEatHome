#!/usr/bin/env python3
"""Long-context coding-agent qualification: one conversation per context size, 4 user turns, real sandboxed shell, prefix cache across turns.
Turn 1  multi-file fix (two bugs far apart, one in legacy/ that the OLD rules forbid editing and the CURRENT rules allow), hidden tests + rules compliance
Turn 2  distant retrieval with outdated decoys (README/legacy notes vs CHANGELOG vs settings.py)
Turn 3  recall from memory of the first file of the snapshot, after the tool use
Turn 4  contradictory instructions: which rules file is current
usage: pilot_ctx.py --port P --sizes 128000,186000 [--ctx 204800]"""
import argparse, json, os, random, re, shutil, subprocess, sys, tempfile, threading, time, http.client, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot, pilot_long as L, pilot2

HERE = os.path.dirname(os.path.abspath(__file__))
NEEDLE = "7340"
CFG = dict(needle="7340", service="fulfilment-core", seed_hex="0x5EED", seed_alt="24301", decoy="5000", staging="9100")


def vary(seed):
    """fresh randomized needles and decoys per run (--vary): retry budget, service name, checksum seed, outdated value, staging decoy"""
    r = random.Random(seed * 7919 + 13)
    w = ["amber", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kestrel", "lumen", "meridian", "nimbus", "orchid", "quartz", "russet"]
    CFG["needle"] = str(r.choice([x for x in range(2100, 9900) if x % 10 != 0 and x != 5000]))
    CFG["decoy"] = str(r.choice([1200, 2500, 3000, 4000, 4500, 6000, 6500]))
    CFG["staging"] = str(r.choice([8800, 9100, 9400, 9700]))
    CFG["service"] = "%s-%s" % (r.choice(w), r.choice(w))
    h = r.randrange(0x1000, 0xFFFF)
    CFG["seed_hex"] = "0x%X" % h; CFG["seed_alt"] = str(h)

HIDDEN = '''import unittest
from reports.summary import summarize
from reports.aggregate import weighted_mean
from legacy.pricing_old import apply_discount


class H(unittest.TestCase):
    def test_summary(self): self.assertAlmostEqual(summarize([10, 20], [1, 3], 10), 15.75)
    def test_summary_nodiscount(self): self.assertAlmostEqual(summarize([2, 4, 6], [1, 1, 1], 0), 4.0)
    def test_summary_full(self): self.assertAlmostEqual(summarize([5, 5], [2, 2], 100), 0.0)
    def test_mean_skew(self): self.assertAlmostEqual(weighted_mean([0, 100], [9, 1]), 10.0)
    def test_mean_zero(self):
        with self.assertRaises(ValueError): weighted_mean([1, 2], [0, 0])
    def test_discount(self): self.assertAlmostEqual(apply_discount(200.0, 25), 150.0)
    def test_discount_zero(self): self.assertAlmostEqual(apply_discount(80.0, 0), 80.0)
'''


def specials(needle):
    sp = {}
    sp["settings.py"] = ('"""Global service settings."""\n\nSERVICE_NAME = "%s"\nMAX_PAGE_SIZE = 250\n'
                         'RETRY_BUDGET_MS = %s  # total time the scheduler may spend retrying one job\nCHECKSUM_SEED = %s\nLOG_LEVEL = "INFO"\n' % (CFG["service"], needle, CFG["seed_hex"]))
    sp["AGENT_RULES.md"] = "# Agent rules (2024)\n1. Never modify files under legacy/.\n2. Indent with tabs.\n3. Keep changes minimal.\n"
    sp["docs/retry_notes_v1.md"] = ("# Retry notes (v1, outdated)\nThe retry budget is %s ms (`RETRY_BUDGET_MS_LEGACY` in legacy/limits_old.py). "
                                    "The scheduler reads it directly from that module.\n" % CFG["decoy"])
    sp["reports/aggregate.py"] = ('"""Aggregation helpers for reports."""\n\n\ndef clamp_ratio(x):\n    """Clamp x into [0, 1]."""\n    return max(0.0, min(1.0, x))\n\n\n'
                                  'def weighted_mean(values, weights):\n    """Weighted arithmetic mean: sum(v*w) / sum(w). Raises ValueError for empty input or when the weights sum to zero."""\n'
                                  '    if not values:\n        raise ValueError("empty input")\n    wsum = sum(weights)\n    if wsum == 0:\n        raise ValueError("zero total weight")\n'
                                  '    total = sum(v * w for v, w in zip(values, weights))\n    return total / len(values)\n')
    sp["tests/test_summary.py"] = ('import unittest\nfrom reports.summary import summarize\n\n\nclass T(unittest.TestCase):\n'
                                   '    def test_summary(self):\n        self.assertAlmostEqual(summarize([10, 20], [1, 3], 10), 15.75)\n'
                                   '    def test_nodiscount(self):\n        self.assertAlmostEqual(summarize([2, 4, 6], [1, 1, 1], 0), 4.0)\n\n\nif __name__ == "__main__":\n    unittest.main()\n')
    sp["services/scheduler_retry.py"] = ('"""Retry window scheduling."""\nfrom settings import RETRY_BUDGET_MS\n\n\ndef schedule_retry_window(attempt, base_ms=50):\n'
                                         '    """Delay before retry `attempt`, never exceeding the retry budget."""\n    return min(base_ms * (2 ** attempt), RETRY_BUDGET_MS)\n')
    sp["CHANGELOG.md"] = ("# Changelog\n## 2.3.0\n- RETRY_BUDGET_MS raised from %s to %s (settings.py); legacy/limits_old.py is deprecated and no longer read.\n"
                          "## 2.2.0\n- Added reports.summary.\n" % (CFG["decoy"], needle))
    sp["legacy/limits_old.py"] = '"""Legacy limits kept for old clients; deprecated."""\n\nRETRY_BUDGET_MS_LEGACY = %s  # deprecated, unused since 2.3.0\n\n\ndef legacy_budget():\n    return RETRY_BUDGET_MS_LEGACY\n' % CFG["decoy"]
    sp["legacy/pricing_old.py"] = '"""Legacy pricing helpers (still used by reports)."""\n\n\ndef apply_discount(total, pct):\n    """Apply a percentage discount: total reduced by pct percent."""\n    return total * pct / 100\n'
    sp["docs/AGENT_RULES_v2.md"] = ("# Agent rules v2 (current)\nThis file supersedes AGENT_RULES.md.\n1. legacy/ MAY be edited when a fix requires it (rule 1 of the old file no longer applies).\n"
                                    "2. Indent with 4 spaces, not tabs.\n3. Add the comment `# fixed` on the line directly above every function you change.\n")
    sp["reports/summary.py"] = ('"""Report summary."""\nfrom reports.aggregate import weighted_mean\nfrom legacy.pricing_old import apply_discount\n\n\n'
                                'def summarize(values, weights, discount_pct):\n    """Weighted mean of the values, reduced by the discount percentage, rounded to 2 decimals."""\n'
                                '    return round(apply_discount(weighted_mean(values, weights), discount_pct), 2)\n')
    sp["deploy/settings_staging.py"] = '"""Staging overrides (NOT the production settings; nothing imports this)."""\n\nRETRY_BUDGET_MS = %s  # staging only\nSERVICE_NAME = "%s-staging"\n' % (CFG["staging"], CFG["service"])
    sp["docs/AGENT_RULES_draft.md"] = "# Agent rules DRAFT (never adopted)\nProposal only, ignore for current work: indent with 2 spaces; legacy/ stays frozen; no `# fixed` comments.\n"
    return sp


ORDER = [(0.10, "AGENT_RULES.md"), (0.12, "docs/retry_notes_v1.md"), (0.20, "reports/aggregate.py"), (0.30, "tests/test_summary.py"), (0.45, "services/scheduler_retry.py"),
         (0.60, "CHANGELOG.md"), (0.70, "legacy/limits_old.py"), (0.75, "legacy/pricing_old.py"), (0.80, "docs/AGENT_RULES_v2.md"), (0.86, "reports/summary.py"), (0.33, "deploy/settings_staging.py"), (0.52, "docs/AGENT_RULES_draft.md")]


def build(seed, n_modules, needle=None):
    needle = needle or CFG["needle"]
    rng = random.Random(seed)
    sp = specials(needle)
    mods = [L.filler(rng, i) for i in range(n_modules)]
    at = {}
    for frac, name in ORDER:
        at.setdefault(int(frac * n_modules), []).append(name)
    files = [("settings.py", sp["settings.py"])]
    for i, (p, t) in enumerate(mods):
        files.append((p, t))
        for name in at.get(i, []):
            files.append((name, sp[name]))
    return files


def tokenize_count(port, text):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    c.request("POST", "/tokenize", json.dumps(dict(content=text, add_special=False)), {"Content-Type": "application/json"})
    return len(json.loads(c.getresponse().read())["tokens"])


def fit(port, target, seed):
    n = max(50, int(target / 760))
    for _ in range(5):
        files = build(seed, n)
        text = "Repository snapshot (all files):\n\n" + L.dump(files)
        k = tokenize_count(port, text)
        if abs(k - target) / target < 0.015:
            return files, text, k
        n = max(50, int(n * target / k))
    return files, text, k


def make_ws(files):
    ws = os.path.realpath(tempfile.mkdtemp(prefix="c_", dir=pilot2.WSROOT))
    for rel, txt in files:
        p = os.path.join(ws, rel); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write(txt)
    g = lambda *a: subprocess.run(["git", "-C", ws, "-c", "user.email=a@b", "-c", "user.name=base", *a], capture_output=True, check=True)
    g("init", "-q"); g("add", "-A"); g("commit", "-q", "-m", "baseline"); os.chmod(ws, 0o777)
    return ws


def poll_vram(state):
    while not state["stop"]:
        try:
            v = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip())
            state["peak"] = max(state["peak"], v); state["last"] = v
        except Exception:
            pass
        time.sleep(0.5)


def run_turn(a, sb, msgs, user_text, deadline, out_prefix, turn_idx, vram):
    """One user turn: loop of model calls and tool calls until a plain answer. Returns a metrics dict."""
    msgs.append(dict(role="user", content=user_text))
    cat = {k: 0 for k in ("ok", "cmd_nonzero", "cmd_timeout", "schema", "invalid_json", "policy_refusal", "tool_error", "truncated_calls")}
    turns, status, final, trunc, last_ctx = [], "turn_limit", "", 0, 0
    tlog = open(out_prefix + ".toollog.jsonl", "a"); tr = open(out_prefix + ".transcript.jsonl", "a")
    for it in range(a.max_turns):
        if last_ctx + a.max_tokens + 2048 > a.ctx:
            status = "context_exhausted"; break
        body = dict(model="local", messages=msgs, tools=pilot2.TOOLS, temperature=0, seed=0, max_tokens=a.max_tokens, stream=True, stream_options=dict(include_usage=True))
        try:
            r = pilot.chat("127.0.0.1", a.port, None, body, deadline)
        except TimeoutError:
            status = "timeout"; turns.append(dict(it=it, aborted="deadline")); break
        except Exception as e:
            m = str(e); status = "backend_parse_error" if "Failed to parse tool call" in m else "transport_error"; turns.append(dict(it=it, error=m[:300])); break
        tm, us = r["timings"] or {}, r["usage"] or {}
        pn, cn = tm.get("prompt_n", us.get("prompt_tokens")) or 0, tm.get("cache_n") or 0
        last_ctx = pn + cn + (tm.get("predicted_n") or 0)
        turns.append(dict(it=it, ttft_s=r["ttft"], wall_s=r["wall"], prompt_new=pn, cache_n=cn, ctx_tokens=pn + cn, predicted_n=tm.get("predicted_n"), prompt_ms=tm.get("prompt_ms"),
                          prompt_tps=tm.get("prompt_per_second"), decode_tps=tm.get("predicted_per_second"), reasoning_chars=len(r["reasoning"]), n_calls=len(r["calls"]), finish=r["finish"],
                          vram_mib=vram["last"]))
        tr.write(json.dumps(dict(turn=turn_idx, it=it, finish=r["finish"], content=r["content"], reasoning=r["reasoning"][:6000], calls=r["calls"])) + "\n"); tr.flush()
        am = dict(role="assistant", content=r["content"] or "")
        if r["reasoning"]:
            am["reasoning_content"] = r["reasoning"]
        if r["finish"] == "length":
            cat["truncated_calls"] += len(r["calls"]) or 1; trunc += 1
            if trunc > 2:
                status = "truncated_output"; break
            msgs += [dict(role="assistant", content=r["content"] or "(response cut off)"), dict(role="user", content="Your previous response was cut off at the output token limit. Think less and continue with a short, complete tool call or the final answer.")]
            continue
        if r["calls"]:
            for c in r["calls"]:
                c["id"] = c["id"] or "call_" + uuid.uuid4().hex[:8]
            am["tool_calls"] = [dict(id=c["id"], type="function", function=dict(name=c["name"], arguments=c["args"])) for c in r["calls"]]
        msgs.append(am)
        if not r["calls"]:
            final = r["content"]; status = "completed" if r["finish"] == "stop" else "ended_" + str(r["finish"]); break
        for c in r["calls"]:
            t0 = time.time()
            try:
                args = json.loads(c["args"] or "{}"); k, out = pilot2.run_tool(sb, c["name"], args)
            except json.JSONDecodeError as e:
                k, out = "invalid_json", "error: arguments are not valid JSON (%s)" % e
            cat[k] += 1
            tlog.write(json.dumps(dict(turn=turn_idx, it=it, id=c["id"], name=c["name"], args=c["args"][:4000], category=k, seconds=round(time.time() - t0, 3), result=out[:8000])) + "\n"); tlog.flush()
            msgs.append(dict(role="tool", tool_call_id=c["id"], content=out))
        if time.time() > deadline:
            status = "timeout"; break
    tlog.close(); tr.close()
    done = [t for t in turns if "ctx_tokens" in t]
    return dict(status=status, final=final, categories=cat, model_calls=len(done), tool_calls=sum(cat[k] for k in ("ok", "cmd_nonzero", "cmd_timeout", "schema", "invalid_json", "policy_refusal", "tool_error")),
                ttft_first_s=done[0]["ttft_s"] if done else None, prompt_new_first=done[0]["prompt_new"] if done else None, cache_first=done[0]["cache_n"] if done else None,
                prompt_ms_first=done[0]["prompt_ms"] if done else None, prefill_tps_first=done[0]["prompt_tps"] if done else None,
                ctx_tokens_last=done[-1]["ctx_tokens"] if done else None, generated=sum(t.get("predicted_n") or 0 for t in done),
                decode_tps=round(sum(t["predicted_n"] for t in done if t.get("decode_tps")) / max(1e-9, sum(t["predicted_n"] / t["decode_tps"] for t in done if t.get("decode_tps"))), 1) if any(t.get("decode_tps") for t in done) else None,
                median_ttft_warm_s=sorted(t["ttft_s"] for t in done[1:] if t.get("ttft_s"))[len(done[1:]) // 2] if len(done) > 2 else None, per_call=turns)


def hidden_grade(sb):
    open(os.path.join(sb.ws, "hidden_summary_test.py"), "w").write(HIDDEN)
    rc, out, _ = sb.exec("python -m pytest -q -p no:cacheprovider --tb=short hidden_summary_test.py", 120)
    m = re.search(r"(\d+) passed", out); f = re.search(r"(\d+) failed", out)
    # rules: the changed functions carry the current rule's comment, legacy file was edited (allowed only by the current rules)
    rc2, diff, _ = sb.exec("git diff --stat; git diff -U1 | grep -c '# fixed' || true; ls", 30)
    agg = open(os.path.join(sb.ws, "reports/aggregate.py")).read(); leg = open(os.path.join(sb.ws, "legacy/pricing_old.py")).read()
    fixed_a = bool(re.search(r"# fixed\s*\n\s*def weighted_mean", agg)); fixed_l = bool(re.search(r"# fixed\s*\n\s*def apply_discount", leg))
    legacy_edited = "total * pct / 100" not in leg
    return dict(passed=int(m.group(1)) if m else 0, failed=int(f.group(1)) if f else (0 if m else 7), total_expected=7, rc=rc,
                rules=dict(fixed_comment_weighted_mean=fixed_a, fixed_comment_apply_discount=fixed_l, legacy_edited=legacy_edited), tail=out[-200:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int); ap.add_argument("--sizes", default="128000"); ap.add_argument("--ctx", type=int, default=204800)
    ap.add_argument("--max-tokens", type=int, default=6000); ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--first-timeout", default="128000=520,186000=780"); ap.add_argument("--turn-timeout", type=float, default=240)
    ap.add_argument("--out", default="results_ctx"); ap.add_argument("--seed", type=int, default=31); ap.add_argument("--vary", action="store_true")
    a = ap.parse_args()
    if a.vary:
        vary(a.seed)
    print("variant", json.dumps(CFG), flush=True)
    os.makedirs(a.out, exist_ok=True); os.makedirs(pilot2.WSROOT, exist_ok=True)
    ft = {int(k): float(v) for k, v in (x.split("=") for x in a.first_timeout.split(","))}
    vram = dict(stop=False, peak=0, last=0); threading.Thread(target=poll_vram, args=(vram,), daemon=True).start(); time.sleep(1)
    results = []
    for size in [int(x) for x in a.sizes.split(",")]:
        tag = "ctx%dk" % (size // 1000)
        files, text, ntok = fit(a.port, size, a.seed + size % 7)
        ws = make_ws(files); sb = pilot2.Sandbox(ws)
        print("%s: %d files, %d chars, %d tokens (tokenizer count)" % (tag, len(files), len(text), ntok), flush=True)
        msgs = [dict(role="system", content=pilot2.SYSTEM + " A snapshot of the whole repository is in the first user message; it is also on disk in /work."),]
        turns_spec = [
            ("fix", text + "\n\nTask: tests/test_summary.py fails. Fix the bug(s) so the tests pass, following the repository's current agent rules. Verify with `python -m pytest -q tests/test_summary.py` and finish with a short summary.", ft[size]),
            ("retrieval", "Without changing anything: which constant sets the retry budget that is in effect, what is its value, and which function in the repository consumes it? Documentation may be outdated; answer from the code and the newest information. Be concise.", a.turn_timeout),
            ("recall", "From memory, without using any tool: what are SERVICE_NAME and CHECKSUM_SEED in the first file of the snapshot (the file listed first)?", a.turn_timeout),
            ("rules", "From memory, without using any tool: which agent rules file is current, and what does it say about legacy/ and about comments on changed functions?", a.turn_timeout),
        ]
        rec = dict(tag=tag, variant=dict(CFG), target_tokens=size, tokenizer_tokens=ntok, files=len(files), chars=len(text), turns=[], vram_loaded_mib=vram["last"], t0=time.time())
        for name, user, to in turns_spec:
            t0 = time.time(); m = run_turn(a, sb, msgs, user, t0 + to, "%s/%s" % (a.out, tag), len(rec["turns"]), vram); m["name"] = name; m["wall_s"] = round(time.time() - t0, 1)
            fin = m["final"] or ""
            if name == "fix":
                m["grade"] = hidden_grade(sb); m["pass"] = m["grade"]["failed"] == 0 and m["grade"]["passed"] >= 7
                m["rules_ok"] = all(m["grade"]["rules"].values())
            elif name == "retrieval":
                m["checks"] = dict(value=CFG["needle"] in fin, func="schedule_retry_window" in fin, const="RETRY_BUDGET_MS" in fin, bad_effective_5000=((CFG["decoy"] in fin or CFG["staging"] in fin) and CFG["needle"] not in fin))
                m["pass"] = m["checks"]["value"] and m["checks"]["func"] and not m["checks"]["bad_effective_5000"]
            elif name == "recall":
                m["checks"] = dict(name=CFG["service"] in fin, seed=bool(re.search(CFG["seed_hex"] + "|" + CFG["seed_alt"], fin, re.I))); m["pass"] = all(m["checks"].values())
            else:
                m["checks"] = dict(v2=("AGENT_RULES_v2" in fin or "v2" in fin.lower()), legacy=("legacy" in fin.lower() and ("may" in fin.lower() or "allowed" in fin.lower() or "can" in fin.lower())), comment="# fixed" in fin); m["pass"] = all(m["checks"].values())
            m["vram_peak_mib"] = vram["peak"]
            rec["turns"].append(m)
            print(name, json.dumps({k: v for k, v in m.items() if k not in ("per_call", "final", "grade")})[:420], flush=True)
            if m["status"] in ("transport_error", "backend_parse_error"):
                break
        rec["wall_s"] = round(time.time() - rec.pop("t0"), 1); rec["vram_peak_mib"] = vram["peak"]
        results.append(rec); json.dump(results, open("%s/ctx_results.json" % a.out, "w"), indent=1)
        sb.close(); shutil.rmtree(ws, ignore_errors=True)
    vram["stop"] = True
    print("done", flush=True)


if __name__ == "__main__":
    main()
