#!/usr/bin/env python3
"""Long-context coding pilot: a deterministic synthetic repository is placed in the prompt (and on disk), then one task is asked.
Variants share a common prefix (settings.py first, same files in the same order) so later tasks can reuse the cache.
Usage: pilot_long.py --label L --port P --plan 32:needle,32:bug,64:needle,64:bug [--cred python] [--model M]"""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot

NOUNS = "order invoice ledger shipment carrier pallet depot tariff coupon refund account session profile segment cohort metric bucket window cursor token region quota policy rule".split()
VERBS = "load merge scale rank filter group split encode decode audit refresh resolve compact validate project reduce".split()
ADJ = "active stale pending hot cold primary shadow bulk partial strict lazy".split()


def filler(rng, idx):
    noun = rng.choice(NOUNS)
    sub = rng.choice(["services", "adapters", "models", "jobs", "storage", "api"])
    path = "%s/%s_%d.py" % (sub, noun, idx)
    out = ['"""%s %s helpers (module %d)."""' % (rng.choice(ADJ).title(), noun, idx), "import math", "", "DEFAULT_%s_LIMIT = %d" % (noun.upper(), rng.randint(10, 9999)), ""]
    for k in range(rng.randint(5, 8)):
        verb, adj = rng.choice(VERBS), rng.choice(ADJ)
        name = "%s_%s_%s_%d" % (verb, adj, noun, rng.randint(100, 99999))
        a, b, c = rng.randint(2, 97), rng.randint(2, 97), rng.randint(1, 9)
        kind = rng.randint(0, 3)
        out.append("def %s(items, limit=DEFAULT_%s_LIMIT):" % (name, noun.upper()))
        out.append('    """%s the %s %s records, keeping at most `limit` entries."""' % (verb.title(), adj, noun))
        if kind == 0:
            out += ["    kept = []", "    for item in items:", "        score = (item * %d + %d) %% %d" % (a, b, a + b), "        if score > %d:" % c, "            kept.append(score)", "        if len(kept) >= limit:", "            break", "    return kept", ""]
        elif kind == 1:
            out += ["    total = 0", "    for i, item in enumerate(items):", "        total += (item - %d) * (i + %d)" % (a, c), "    if total < 0:", "        total = -total", "    return min(total, limit * %d)" % b, ""]
        elif kind == 2:
            out += ["    buckets = {}", "    for item in items:", "        buckets.setdefault(item %% %d, []).append(item)" % (c + 2), "    ordered = sorted(buckets.items())", "    return [(k, len(v)) for k, v in ordered][:limit]", ""]
        else:
            out += ["    if not items:", "        return 0.0", "    mean = sum(items) / len(items)", "    var = sum((x - mean) ** 2 for x in items) / len(items)", "    return round(math.sqrt(var) * %d / %d, 3)" % (a, b), ""]
    return path, "\n".join(out) + "\n"


def special(needle):
    settings = ('"""Global service settings."""\n\nSERVICE_NAME = "fulfilment-core"\nMAX_PAGE_SIZE = 250\n'
                'RETRY_BUDGET_MS = %s  # total time the scheduler may spend retrying one job\nCHECKSUM_SEED = 0x5EED\nLOG_LEVEL = "INFO"\n' % needle)
    consumer = ('"""Retry window scheduling."""\nfrom settings import RETRY_BUDGET_MS\n\n\ndef schedule_retry_window(attempt, base_ms=50):\n'
                '    """Delay before retry `attempt`, never exceeding the retry budget."""\n    return min(base_ms * (2 ** attempt), RETRY_BUDGET_MS)\n')
    decoy = ('"""Legacy limits kept for old clients; deprecated."""\n\nRETRY_BUDGET_MS_LEGACY = 5000  # deprecated, unused since v2\n\n\ndef legacy_budget():\n    return RETRY_BUDGET_MS_LEGACY\n')
    agg = ('"""Aggregation helpers for reports."""\n\n\ndef clamp_ratio(x):\n    """Clamp x into [0, 1]."""\n    return max(0.0, min(1.0, x))\n\n\n'
           'def weighted_mean(values, weights):\n    """Weighted arithmetic mean: sum(v*w) / sum(w). Raises ValueError for empty input or when the weights sum to zero."""\n'
           '    if not values:\n        raise ValueError("empty input")\n    wsum = sum(weights)\n    if wsum == 0:\n        raise ValueError("zero total weight")\n'
           '    total = sum(v * w for v, w in zip(values, weights))\n    return total / len(values)\n')
    test = ('import unittest\nfrom reports.aggregate import weighted_mean, clamp_ratio\n\n\nclass T(unittest.TestCase):\n'
            '    def test_equal_weights(self):\n        self.assertAlmostEqual(weighted_mean([1, 2, 3], [1, 1, 1]), 2.0)\n'
            '    def test_unequal_weights(self):\n        self.assertAlmostEqual(weighted_mean([10, 20], [1, 3]), 17.5)\n'
            '    def test_clamp(self):\n        self.assertEqual(clamp_ratio(1.5), 1.0)\n\n\nif __name__ == "__main__":\n    unittest.main()\n')
    return settings, consumer, decoy, agg, test


HIDDEN = '''import unittest
from reports.aggregate import weighted_mean, clamp_ratio


class H(unittest.TestCase):
    def test_equal(self): self.assertAlmostEqual(weighted_mean([1, 2, 3], [1, 1, 1]), 2.0)
    def test_unequal(self): self.assertAlmostEqual(weighted_mean([10, 20], [1, 3]), 17.5)
    def test_single(self): self.assertAlmostEqual(weighted_mean([4], [9]), 4.0)
    def test_skewed(self): self.assertAlmostEqual(weighted_mean([0, 100], [9, 1]), 10.0)
    def test_zero_weight(self):
        with self.assertRaises(ValueError): weighted_mean([1, 2], [0, 0])
    def test_empty(self):
        with self.assertRaises(ValueError): weighted_mean([], [])
    def test_clamp(self): self.assertEqual((clamp_ratio(-1), clamp_ratio(0.3), clamp_ratio(2)), (0.0, 0.3, 1.0))
'''


def build(seed, n_modules, needle):
    """Ordered list of (path, text). Special files are at fixed positions so every variant of a seed shares a prefix."""
    rng = random.Random(seed)
    settings, consumer, decoy, agg, test = special(needle)
    mods = [filler(rng, i) for i in range(n_modules)]
    files = [("settings.py", settings)]
    for i, (p, t) in enumerate(mods):
        files.append((p, t))
        if i == 20:
            files.append(("reports/aggregate.py", agg)); files.append(("tests/test_aggregate.py", test))
        if i == 30:
            files.append(("services/scheduler_retry.py", consumer))
        if i == 35:
            files.append(("legacy/limits_old.py", decoy))
    return files


def dump(files):
    return "".join("=== %s ===\n%s\n" % (p, t) for p, t in files)


def make_task(kind, files):
    pre = "Repository snapshot (all files):\n\n" + dump(files) + "\n"
    if kind == "needle":
        q = ("Without modifying any files: in the first file of the snapshot (settings.py), which constant is the retry budget and what is its value? "
             "Which function in the repository consumes it (give the module path and the function name)? Answer concisely.")
    else:
        q = "tests/test_aggregate.py fails. Find and fix the bug in the repository (do not change the tests), then confirm with `python3 -m unittest tests.test_aggregate`."
    return dict(name=kind, files=dict(files), prompt=pre + q, hidden=HIDDEN, ref={}, grade="long_" + kind)


def grade(task, ws, final_text):
    if task["grade"] == "long_needle":
        t = final_text or ""
        c = dict(value=NEEDLE[0] in t, func="schedule_retry_window" in t, module="scheduler_retry" in t, name="RETRY_BUDGET_MS" in t, decoy_ok="5000" not in t.replace(NEEDLE[0], ""))
        return sum(c.values()), len(c), c
    open(os.path.join(ws, "hidden_grade.py"), "w").write(HIDDEN)
    p, n = pilot.unittest_counts(ws, "hidden_grade")
    return p, n, {}


NEEDLE = ["7340"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label"); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int); ap.add_argument("--model", default="local")
    ap.add_argument("--cred"); ap.add_argument("--plan", default="32:needle,32:bug,64:needle,64:bug"); ap.add_argument("--counts", default="32=47,64=94,128=188")
    ap.add_argument("--timeouts", default="32=150,64=240,128=480"); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--needle", default="7340")
    ap.add_argument("--max-turns", type=int, default=10); ap.add_argument("--max-tokens", type=int, default=5000); ap.add_argument("--session-budget", type=float, default=660)
    ap.add_argument("--out", default="results_long"); ap.add_argument("--tag", default="")
    ap.add_argument("--dump-only", action="store_true")
    a = ap.parse_args()
    NEEDLE[0] = a.needle
    counts = {int(k): int(v) for k, v in (x.split("=") for x in a.counts.split(","))}
    touts = {int(k): float(v) for k, v in (x.split("=") for x in a.timeouts.split(","))}
    if a.dump_only:
        for k, n in counts.items():
            t = dump(build(a.seed, n, a.needle)); print(k, "modules", n, "chars", len(t))
        sys.exit(0)
    pilot.grade = grade
    token = json.load(open(os.path.expanduser("$CREDENTIALS_FILE")))[a.cred] if a.cred else None
    os.makedirs(a.out, exist_ok=True)
    results, t0 = [], time.time()
    for item in a.plan.split(","):
        size, kind = item.split(":")
        size = int(size)
        name = "%s%s_%dk_%s" % (a.label, a.tag, size, kind)
        if time.time() - t0 > a.session_budget:
            results.append(dict(task=name, status="skipped_budget")); print("SKIP", name, flush=True); continue
        task = make_task(kind, build(a.seed, counts[size], a.needle))
        args = argparse.Namespace(label=a.label + a.tag + "_%dk" % size, host=a.host, port=a.port, model=a.model, task_timeout=touts[size], max_turns=a.max_turns, max_tokens=a.max_tokens)
        print("START", name, "chars", len(task["prompt"]), "at +%ds" % (time.time() - t0), flush=True)
        r = pilot.run_task(args, token, task, a.out)
        r.update(task=name, ctx_k=size, prompt_chars=len(task["prompt"]))
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("per_turn", "final_answer")}), flush=True)
        json.dump(results, open(os.path.join(a.out, "%s%s_results.json" % (a.label, a.tag)), "w"), indent=1)
    print("session_s", round(time.time() - t0, 1), flush=True)
