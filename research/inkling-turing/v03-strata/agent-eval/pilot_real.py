#!/usr/bin/env python3
"""Real-repository coding-agent pilot: 4 tasks taken from real upstream bug-fix commits (parent commit as workspace, the commit's tests as hidden grading).
Usage: pilot_real.py --validate | --label L --port P [--cred python] [--model M]"""
import argparse, io, json, os, re, shutil, subprocess, sys, tarfile, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "real")
VENV_BIN = os.path.join(REAL, "venv", "bin")
os.environ["PILOT_BIN"] = VENV_BIN

URL_HIDDEN = '''from boltons.urlutils import OrderedMultiDict as OMD


def test_url_addlist_iterator():
    omd = OMD()
    omd.addlist('a', iter([1, 2, 3]))
    omd.addlist('b', (x for x in [4, 5]))
    assert omd.keys() == ['a', 'b']
    assert omd.getlist('a') == [1, 2, 3]
    assert omd.getlist('b') == [4, 5]
    assert omd.get('a') == 3
    e = OMD()
    e.addlist('a', (x for x in []))
    assert len(e) == 0 and e.keys() == [] and e.get('a') is None
'''

TASKS = {
    "boltons_addlist": dict(
        repo="boltons", commit="a1ea1030ed", fix_files=["boltons/dictutils.py", "boltons/urlutils.py"],
        hidden_from_commit=["tests/test_dictutils.py"], hidden_extra={"tests/test_hidden_url.py": URL_HIDDEN},
        targets=["tests/test_dictutils.py", "tests/test_urlutils.py", "tests/test_hidden_url.py"],
        prompt=("Bug report for the boltons library: OrderedMultiDict.addlist() misbehaves when it is given a one-shot iterator such as iter([1, 2, 3]) or a "
                "generator expression. Afterwards keys() and repr() show the values while getlist() returns [] and get() raises IndexError; an empty generator "
                "leaves a key recorded with no values, so len() disagrees with keys(). The docstring promises that any iterable works. Fix this everywhere "
                "this behaviour is implemented in the repository (the same class may exist in more than one module). Existing tests are in tests/. "
                "Run tests with `python3 -m pytest -q tests/test_dictutils.py tests/test_urlutils.py`.")),
    "boltons_barrel": dict(
        repo="boltons", commit="e00c10a809", fix_files=["boltons/listutils.py"], hidden_from_commit=["tests/test_listutils.py"], hidden_extra={},
        targets=["tests/test_listutils.py"],
        prompt=("Bug report for the boltons library: boltons.listutils.BarrelList (a list built from several sublists) does not behave like a plain list once it "
                "holds more than one sublist. Reported symptoms: bl[len(bl)] returns an element instead of raising IndexError; insert(len(bl), x), or an index "
                "beyond either end, puts the item in the wrong place instead of clamping like list.insert; and pop() can raise IndexError on a non-empty "
                "BarrelList after indexed pops emptied the last sublist. Fix the implementation in boltons/listutils.py so it matches list semantics. "
                "Tests live in tests/test_listutils.py (run `python3 -m pytest -q tests/test_listutils.py`).")),
    "cachetools_setitem": dict(
        repo="cachetools", commit="39b31bc9b6", fix_files=["src/cachetools/__init__.py"], hidden_from_commit=["tests/__init__.py"], hidden_extra={},
        targets=["tests"],
        prompt=("Bug report for cachetools: when a Cache is created with getsizeof and an existing key's value grows, __setitem__ evicts an unrelated entry "
                "even though the updated cache would still fit within maxsize. Example: cache = Cache(maxsize=10, getsizeof=lambda x: x); cache[3] = 3; "
                "cache[4] = 4; cache[4] = 7 -> the entry for key 3 is evicted although total size 3 + 7 = 10 fits. Replacing with the same or a smaller size "
                "must also keep the accounting correct. Fix this in src/cachetools/__init__.py without breaking the cache subclasses. "
                "Run tests with `python3 -m pytest -q tests`.")),
    "humanize_tzaware": dict(
        repo="humanize", commit="a47a89ea91", fix_files=["src/humanize/time.py"], hidden_from_commit=["tests/test_time.py"], hidden_extra={},
        targets=["tests/test_time.py"],
        prompt=("Bug report for humanize: naturalday() and naturaldate() compare a timezone-aware datetime against the machine's local date, so for values "
                "expressed in another timezone they can say 'today'/'tomorrow'/'yesterday' wrongly. 'Today' should be evaluated in the value's own timezone. "
                "Naive datetimes and plain dates must keep their current behaviour. Fix both functions in src/humanize/time.py. "
                "Tests: `python3 -m pytest -q tests/test_time.py`.")),
}


def git_archive(repo, rev, dest):
    data = subprocess.run(["git", "-C", os.path.join(REAL, repo), "archive", rev], capture_output=True, check=True).stdout
    tarfile.open(fileobj=io.BytesIO(data)).extractall(dest)


def git_show(repo, rev, path):
    return subprocess.run(["git", "-C", os.path.join(REAL, repo), "show", "%s:%s" % (rev, path)], capture_output=True, check=True).stdout


def make_setup(t):
    def setup(ws):
        git_archive(t["repo"], t["commit"] + "^", ws)
        if t["repo"] == "humanize":  # version module is generated at build time upstream
            open(os.path.join(ws, "src/humanize/_version.py"), "w").write("__version__ = '0.0.0'" + chr(10))
    return setup


def pytest_counts(ws, targets):
    env = dict(PATH=VENV_BIN + ":" + os.environ.get("PATH", ""), PYTHONPATH=ws + ":" + ws + "/src", PYTHONDONTWRITEBYTECODE="1", HOME=ws)
    r = subprocess.run([os.path.join(VENV_BIN, "python"), "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x" if False else "--tb=no", *targets], cwd=ws, capture_output=True, text=True, env=env, timeout=240)
    out = r.stdout + r.stderr
    g = lambda w: int(m.group(1)) if (m := re.search(r"(\d+) %s" % w, out)) else 0
    return g("passed"), g("failed") + g("error"), out[-400:]


def grade_real(task, ws, final_text):
    t = task["spec"]
    for p in t["hidden_from_commit"]:
        full = os.path.join(ws, p); os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "wb").write(git_show(t["repo"], t["commit"], p))
    for p, txt in t["hidden_extra"].items():
        open(os.path.join(ws, p), "w").write(txt)
    passed, failed, tail = pytest_counts(ws, t["targets"])
    ref = t["ref_passed"]
    ok = failed == 0 and passed >= ref
    return (ref if ok else passed), ref, dict(failed=failed, passed=passed, ref_passed=ref, tail=tail[-200:])


def build_task(name):
    t = TASKS[name]
    return dict(name=name, spec=t, files={}, prompt=t["prompt"], hidden=None, ref={}, grade="real", setup=make_setup(t))


def validate():
    ok = True
    for name, t in TASKS.items():
        ws = os.path.realpath(tempfile.mkdtemp(prefix="val_"))
        t["ref_passed"] = 0
        make_setup(t)(ws)
        # hidden tests on the parent: must fail
        task = build_task(name)
        _, _, c0 = grade_real(task, ws, "")
        for f in t["fix_files"]:
            open(os.path.join(ws, f), "wb").write(git_show(t["repo"], t["commit"], f))
        _, _, c1 = grade_real(task, ws, "")
        print(name, "parent:", {k: c0[k] for k in ("passed", "failed")}, "fixed:", {k: c1[k] for k in ("passed", "failed")}, "| files", sum(len(f) for _, _, f in os.walk(ws)))
        ok &= c0["failed"] > 0 and c1["failed"] == 0 and c1["passed"] > 0
        print("   ref_passed =", c1["passed"], "OK" if c0["failed"] > 0 and c1["failed"] == 0 else "BAD", c1["tail"][-90:].replace("\n", " "))
        shutil.rmtree(ws)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label"); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int); ap.add_argument("--model", default="local")
    ap.add_argument("--cred"); ap.add_argument("--task-timeout", type=float, default=180); ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=6000); ap.add_argument("--session-budget", type=float, default=520)
    ap.add_argument("--tasks", default=",".join(TASKS)); ap.add_argument("--out", default="results_real"); ap.add_argument("--validate", action="store_true")
    ap.add_argument("--ref", default="ref_counts.json")
    a = ap.parse_args()
    if a.validate:
        sys.exit(validate())
    refs = json.load(open(os.path.join(HERE, a.ref)))
    for k, v in refs.items():
        TASKS[k]["ref_passed"] = v
    pilot.grade = grade_real
    token = json.load(open(os.path.expanduser("$CREDENTIALS_FILE")))[a.cred] if a.cred else None
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time(); results = []
    for name in a.tasks.split(","):
        if time.time() - t0 > a.session_budget:
            results.append(dict(task=name, label=a.label, status="skipped_budget")); print("SKIP", name, flush=True); continue
        print("START", name, "at +%ds" % (time.time() - t0), flush=True)
        r = pilot.run_task(a, token, build_task(name), a.out)
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("per_turn",)}), flush=True)
        json.dump(results, open(os.path.join(a.out, "%s_results.json" % a.label), "w"), indent=1)
    print("session_s", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
