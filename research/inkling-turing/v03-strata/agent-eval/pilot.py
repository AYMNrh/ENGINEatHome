#!/usr/bin/env python3
"""Small coding-agent pilot: four short-context tasks, one tool schema, strict timeouts, identical for every model.
Usage: pilot.py --label L --port P [--cred python] [--task-timeout 135] [--tasks bugfix,feature,multifile,explore]
       pilot.py --selftest      (CPU only: validates workspaces and graders with reference solutions)"""
import argparse, http.client, json, os, re, shlex, shutil, socket, subprocess, sys, tempfile, time, uuid

# ----------------------------------------------------------------------------------------------------------------- tasks
def T_bugfix():
    files = {
        "pager.py": '''def paginate(items, page, per_page):
    """Return the items on a 1-based page. Page 1 is the first page; pages past the end give []."""
    start = page * per_page
    return items[start:start + per_page]


def page_count(total, per_page):
    """Number of pages needed for `total` items (rounded up). 0 items need 0 pages."""
    return total // per_page
''',
        "test_pager.py": '''import unittest
from pager import paginate, page_count


class T(unittest.TestCase):
    def test_first_page(self):
        self.assertEqual(paginate(list(range(10)), 1, 4), [0, 1, 2, 3])

    def test_last_partial_page(self):
        self.assertEqual(paginate(list(range(10)), 3, 4), [8, 9])

    def test_count_rounds_up(self):
        self.assertEqual(page_count(10, 4), 3)


if __name__ == "__main__":
    unittest.main()
''',
    }
    hidden = '''import unittest
from pager import paginate, page_count


class H(unittest.TestCase):
    def test_first(self): self.assertEqual(paginate(list("abcdefg"), 1, 3), ["a", "b", "c"])
    def test_second(self): self.assertEqual(paginate(list("abcdefg"), 2, 3), ["d", "e", "f"])
    def test_last(self): self.assertEqual(paginate(list("abcdefg"), 3, 3), ["g"])
    def test_past_end(self): self.assertEqual(paginate(list("abcdefg"), 4, 3), [])
    def test_count_exact(self): self.assertEqual(page_count(9, 3), 3)
    def test_count_round_up(self): self.assertEqual(page_count(10, 3), 4)
    def test_count_zero(self): self.assertEqual(page_count(0, 5), 0)
    def test_count_one(self): self.assertEqual(page_count(1, 5), 1)
'''
    prompt = "The tests in test_pager.py fail. Find and fix the bugs in pager.py (do not change the tests), then confirm the tests pass."
    ref = {"pager.py": files["pager.py"].replace("page * per_page", "(page - 1) * per_page").replace("total // per_page", "-(-total // per_page)")}
    return dict(name="bugfix", files=files, hidden=hidden, prompt=prompt, ref=ref, grade=None)


def T_feature():
    files = {
        "inventory.py": '''class Inventory:
    """Tracks stock per SKU."""

    def __init__(self):
        self._qty = {}
        self._price = {}

    def add(self, sku, qty, price):
        self._qty[sku] = self._qty.get(sku, 0) + qty
        self._price[sku] = price

    def remove(self, sku, qty):
        if self._qty.get(sku, 0) < qty:
            raise ValueError("insufficient stock")
        self._qty[sku] -= qty

    def quantity(self, sku):
        return self._qty.get(sku, 0)

    def total_value(self):
        return sum(self._qty[s] * self._price[s] for s in self._qty)
''',
        "test_inventory.py": '''import unittest
from inventory import Inventory


class T(unittest.TestCase):
    def test_add_remove(self):
        inv = Inventory(); inv.add("a", 5, 2.0); inv.remove("a", 2)
        self.assertEqual(inv.quantity("a"), 3)

    def test_value(self):
        inv = Inventory(); inv.add("a", 5, 2.0); inv.add("b", 1, 10.0)
        self.assertEqual(inv.total_value(), 20.0)


if __name__ == "__main__":
    unittest.main()
''',
    }
    hidden = '''import unittest
from inventory import Inventory


def make():
    inv = Inventory(); inv.add("b", 2, 1.0); inv.add("a", 1, 1.0); inv.add("c", 9, 1.0); inv.add("d", 5, 1.0)
    return inv


class H(unittest.TestCase):
    def test_existing(self):
        inv = make(); inv.remove("c", 4); self.assertEqual(inv.quantity("c"), 5)
    def test_low_stock_sorted(self): self.assertEqual(make().low_stock(5), ["a", "b"])
    def test_low_stock_strict(self): self.assertNotIn("d", make().low_stock(5))
    def test_low_stock_empty(self): self.assertEqual(make().low_stock(0), [])
    def test_restock(self):
        inv = make(); inv.restock("a", 10); self.assertEqual(inv.quantity("a"), 11)
    def test_restock_unknown(self):
        with self.assertRaises(KeyError): make().restock("zzz", 1)
    def test_restock_keeps_price(self):
        inv = make(); inv.restock("a", 1); self.assertEqual(inv.total_value(), 2*1.0 + 2*1.0 + 9*1.0 + 5*1.0)
'''
    prompt = ("Add two methods to the Inventory class in inventory.py: low_stock(threshold) returns the SKUs whose quantity is strictly below "
              "threshold, sorted alphabetically; restock(sku, qty) adds qty to an existing SKU and raises KeyError for an unknown SKU "
              "(it must not change the price). Add tests for both to test_inventory.py and make sure all tests pass.")
    ref = {"inventory.py": files["inventory.py"] + '''
    def low_stock(self, threshold):
        return sorted(s for s, q in self._qty.items() if q < threshold)

    def restock(self, sku, qty):
        if sku not in self._qty:
            raise KeyError(sku)
        self._qty[sku] += qty
'''}
    return dict(name="feature", files=files, hidden=hidden, prompt=prompt, ref=ref, grade=None)


def T_multifile():
    files = {
        "shop/__init__.py": "",
        "shop/pricing.py": '''def calc_total(items):
    """items: list of (unit_price, qty). Returns the total rounded to 2 decimals."""
    return round(sum(p * q for p, q in items), 2)
''',
        "shop/cart.py": '''from shop.pricing import calc_total


class Cart:
    def __init__(self):
        self.items = []

    def add(self, price, qty=1):
        self.items.append((price, qty))

    def checkout(self):
        return calc_total(self.items)
''',
        "shop/report.py": '''from shop.pricing import calc_total


def summary(cart):
    return "%d lines, total %.2f" % (len(cart.items), calc_total(cart.items))
''',
        "shop/cli.py": '''from shop import pricing
from shop.cart import Cart


def main(argv):
    cart = Cart()
    for arg in argv:
        price, qty = arg.split("x")
        cart.add(float(price), int(qty))
    print(pricing.calc_total(cart.items))
    return cart.checkout()
''',
        "test_shop.py": '''import unittest
from shop.cart import Cart
from shop.pricing import calc_total
from shop.report import summary


class T(unittest.TestCase):
    def test_total(self): self.assertEqual(calc_total([(1.5, 2), (2.0, 1)]), 5.0)
    def test_cart(self):
        c = Cart(); c.add(10.0, 2); self.assertEqual(c.checkout(), 20.0)
    def test_report(self):
        c = Cart(); c.add(10.0, 2); self.assertEqual(summary(c), "1 lines, total 20.00")


if __name__ == "__main__":
    unittest.main()
''',
    }
    hidden = '''import unittest, subprocess, io, contextlib
from shop.cart import Cart
from shop.pricing import compute_total
from shop.report import summary
from shop import cli


class H(unittest.TestCase):
    def test_compute(self): self.assertEqual(compute_total([(1.5, 2), (2.0, 1)]), 5.0)
    def test_compute_tax(self): self.assertEqual(compute_total([(10.0, 2)], tax_rate=0.25), 25.0)
    def test_rounding(self): self.assertEqual(compute_total([(0.1, 3)], tax_rate=0.1), 0.33)
    def test_old_name_gone(self):
        import shop.pricing as p
        self.assertFalse(hasattr(p, "calc_total"))
    def test_cart_tax(self):
        c = Cart(tax_rate=0.2); c.add(10.0, 2); self.assertEqual(c.checkout(), 24.0)
    def test_cart_default(self):
        c = Cart(); c.add(10.0, 2); self.assertEqual(c.checkout(), 20.0)
    def test_report_tax(self):
        c = Cart(tax_rate=0.5); c.add(2.0, 2); self.assertEqual(summary(c), "1 lines, total 6.00")
    def test_cli(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = cli.main(["2.0x3"])
        self.assertEqual(r, 6.0); self.assertEqual(buf.getvalue().strip(), "6.0")
    def test_no_stale_refs(self):
        out = subprocess.run(["grep", "-rn", "calc_total", "shop"], capture_output=True, text=True).stdout
        self.assertEqual(out.strip(), "")
'''
    prompt = ("Refactor the shop package: rename pricing.calc_total to compute_total and give it an optional keyword argument tax_rate (default 0.0) "
              "so the result is round(sum(p*q) * (1 + tax_rate), 2). Cart must accept an optional tax_rate constructor argument (default 0.0) "
              "and apply it when checking out; the report summary must show the taxed total. Update every caller and the existing tests "
              "(no references to calc_total may remain), then run the tests.")
    ref = {
        "shop/pricing.py": '''def compute_total(items, tax_rate=0.0):
    return round(sum(p * q for p, q in items) * (1 + tax_rate), 2)
''',
        "shop/cart.py": '''from shop.pricing import compute_total


class Cart:
    def __init__(self, tax_rate=0.0):
        self.items = []
        self.tax_rate = tax_rate

    def add(self, price, qty=1):
        self.items.append((price, qty))

    def checkout(self):
        return compute_total(self.items, tax_rate=self.tax_rate)
''',
        "shop/report.py": '''from shop.pricing import compute_total


def summary(cart):
    return "%d lines, total %.2f" % (len(cart.items), compute_total(cart.items, tax_rate=cart.tax_rate))
''',
        "shop/cli.py": '''from shop import pricing
from shop.cart import Cart


def main(argv):
    cart = Cart()
    for arg in argv:
        price, qty = arg.split("x")
        cart.add(float(price), int(qty))
    print(pricing.compute_total(cart.items))
    return cart.checkout()
''',
        "test_shop.py": files["test_shop.py"].replace("calc_total", "compute_total"),
    }
    return dict(name="multifile", files=files, hidden=hidden, prompt=prompt, ref=ref, grade=None)


def T_explore():
    files = {
        "README.md": "# netclient\nSmall HTTP client. Docs may be stale: the retry numbers below are from v0.1 (3 retries, 1s base delay).\n",
        "netclient/__init__.py": "",
        "netclient/config.py": '''DEFAULTS = {
    "retry_statuses": (429, 500, 502, 503),
    "max_retries": 4,
    "base_delay": 0.5,
    "backoff_factor": 2,
    "max_delay": 5,
    "timeout": 10,
}
''',
        "netclient/profiles.py": '''from netclient.config import DEFAULTS

PROFILES = {
    "batch": dict(max_retries=6, max_delay=30),
    "interactive": dict(max_retries=3, base_delay=0.25, backoff_factor=3, max_delay=2),
}
DEFAULT_PROFILE = "interactive"


def resolve(profile=None):
    cfg = dict(DEFAULTS)
    cfg.update(PROFILES[profile or DEFAULT_PROFILE])
    return cfg
''',
        "netclient/retry.py": '''def delay_for(cfg, attempt):
    """attempt is 0-based: delay slept before retry number attempt+1."""
    return min(cfg["max_delay"], cfg["base_delay"] * cfg["backoff_factor"] ** attempt)


def should_retry(cfg, status, attempt):
    return status in cfg["retry_statuses"] and attempt < cfg["max_retries"]
''',
        "netclient/client.py": '''from netclient.profiles import resolve
from netclient.retry import delay_for, should_retry


class Client:
    def __init__(self, profile=None):
        self.cfg = resolve(profile)

    def request(self, send, url):
        attempt = 0
        while True:
            status = send(url)
            if not should_retry(self.cfg, status, attempt):
                return status
            attempt += 1
''',
        "netclient/log.py": "def log(msg):\n    print(msg)\n",
        "netclient/util.py": "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
    }
    prompt = ("Using only read-only inspection (do not modify files): for a Client created with the default profile and no overrides, "
              "which HTTP status codes are retried, what are the delays (in seconds) slept before each retry, and what is the total worst-case "
              "waiting time in seconds? Cite the files that decide each value. Answer concisely.")
    return dict(name="explore", files=files, hidden=None, prompt=prompt, ref={}, grade="explore")


TASKS = {f.__name__[2:]: f for f in (T_bugfix, T_feature, T_multifile, T_explore)}
SYSTEM = ("You are a coding agent working in a small Python repository. Use the provided tools to inspect and edit files and run commands in the workspace. "
          "Make the minimal correct change, verify with the tests when relevant, and finish with a brief plain-text summary (no tool call) when done.")

# ----------------------------------------------------------------------------------------------------------------- tools
TOOLS = [
    dict(type="function", function=dict(name="list_dir", description="List files in a directory of the workspace.", parameters=dict(type="object", properties=dict(path=dict(type="string")), required=["path"]))),
    dict(type="function", function=dict(name="read_file", description="Read a text file.", parameters=dict(type="object", properties=dict(path=dict(type="string")), required=["path"]))),
    dict(type="function", function=dict(name="write_file", description="Create or overwrite a file.", parameters=dict(type="object", properties=dict(path=dict(type="string"), content=dict(type="string")), required=["path", "content"]))),
    dict(type="function", function=dict(name="replace_in_file", description="Replace one exact occurrence of old with new in a file.", parameters=dict(type="object", properties=dict(path=dict(type="string"), old=dict(type="string"), new=dict(type="string")), required=["path", "old", "new"]))),
    dict(type="function", function=dict(name="grep", description="Search for a regex in the workspace, returns file:line:text.", parameters=dict(type="object", properties=dict(pattern=dict(type="string"), path=dict(type="string")), required=["pattern"]))),
    dict(type="function", function=dict(name="run_command", description="Run a command (python3, ls, cat, grep, head, tail, wc, find only; no shell operators) in the workspace, e.g. 'python3 -m unittest -v'.", parameters=dict(type="object", properties=dict(command=dict(type="string")), required=["command"]))),
]
REQUIRED = {t["function"]["name"]: t["function"]["parameters"]["required"] for t in TOOLS}


class Bad(Exception):
    """A malformed or disallowed tool call (counted as a tool-call error)."""


def safe(ws, rel):
    p = os.path.realpath(os.path.join(ws, rel or "."))
    if not (p == ws or p.startswith(ws + os.sep)):
        raise Bad("path outside workspace")
    return p


def run_tool(ws, name, args):
    if name not in REQUIRED:
        raise Bad("unknown tool " + name)
    if not isinstance(args, dict) or any(k not in args for k in REQUIRED[name]):
        raise Bad("missing arguments")
    if name == "list_dir":
        p = safe(ws, args["path"])
        if not os.path.isdir(p):
            raise Bad("not a directory")
        return "\n".join(sorted(e + ("/" if os.path.isdir(os.path.join(p, e)) else "") for e in os.listdir(p) if e != "__pycache__"))
    if name == "read_file":
        p = safe(ws, args["path"])
        if not os.path.isfile(p):
            raise Bad("no such file")
        return open(p, encoding="utf-8").read()[:12000]
    if name == "write_file":
        p = safe(ws, args["path"])
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w", encoding="utf-8").write(args["content"])
        return "wrote %d bytes" % len(args["content"])
    if name == "replace_in_file":
        p = safe(ws, args["path"])
        if not os.path.isfile(p):
            raise Bad("no such file")
        s = open(p, encoding="utf-8").read()
        if s.count(args["old"]) != 1:
            raise Bad("old text must occur exactly once (found %d)" % s.count(args["old"]))
        open(p, "w", encoding="utf-8").write(s.replace(args["old"], args["new"]))
        return "replaced"
    if name == "grep":
        base = safe(ws, args.get("path") or ".")
        r = subprocess.run(["grep", "-rnE", "--include=*", "--exclude-dir=__pycache__", "-e", args["pattern"], base], capture_output=True, text=True, timeout=15)
        return (r.stdout.replace(ws + os.sep, ""))[:6000] or "(no matches)"
    if name == "run_command":
        cmd = args["command"]
        if re.search(r"[;&|><`$\n]", cmd):
            raise Bad("shell operators are not allowed")
        parts = shlex.split(cmd)
        if not parts or parts[0] not in ("python3", "python", "ls", "cat", "grep", "head", "tail", "wc", "find"):
            raise Bad("command not allowed")
        if any(a.startswith("/") or ".." in a for a in parts[1:]) or "-exec" in parts or ("-c" in parts and parts[0].startswith("python")):
            raise Bad("argument not allowed")
        env = dict(PATH=os.environ.get("PATH", "/usr/bin:/bin"), PYTHONDONTWRITEBYTECODE="1", HOME=ws)
        if os.environ.get("PILOT_BIN"):  # real-repo mode: venv python with pytest, repo importable
            env["PATH"] = os.environ["PILOT_BIN"] + ":" + env["PATH"]
            env["PYTHONPATH"] = ws + ":" + ws + "/src"
        r = subprocess.run(parts, cwd=ws, capture_output=True, text=True, timeout=30, env=env)
        return ("exit=%d\n" % r.returncode + (r.stdout + r.stderr))[-5000:]


# ----------------------------------------------------------------------------------------------------------------- grading
def unittest_counts(ws, module):
    r = subprocess.run(["python3", "-m", "unittest", module], cwd=ws, capture_output=True, text=True, timeout=60, env=dict(PATH=os.environ.get("PATH", ""), PYTHONDONTWRITEBYTECODE="1"))
    out = r.stderr + r.stdout
    m = re.search(r"Ran (\d+) test", out)
    n = int(m.group(1)) if m else 0
    bad = sum(int(x) for x in re.findall(r"(?:failures|errors)=(\d+)", out))
    if not m:
        bad = n = max(n, 1)
    return n - bad, n


def grade(task, ws, final_text):
    if task["grade"] == "explore":
        t = final_text or ""
        checks = dict(statuses=all(s in t for s in ("429", "500", "502", "503")), d1="0.25" in t, d2="0.75" in t,
                      d3=bool(re.search(r"(?<![\d.])2(\.0)?(?![\d.])", t)), total=bool(re.search(r"(?<![\d.])3(\.0)?\s*(s|sec|second)", t)) or "3.0" in t,
                      stale=not re.search(r"\b4 retries\b|1s base", t))
        return sum(checks.values()), len(checks), checks
    open(os.path.join(ws, "hidden_grade.py"), "w").write(task["hidden"])
    p, n = unittest_counts(ws, "hidden_grade")
    return p, n, {}


# ----------------------------------------------------------------------------------------------------------------- model client
def chat(host, port, token, body, deadline):
    conn = http.client.HTTPConnection(host, port, timeout=max(1, deadline - time.time()))
    t0 = time.time()
    hdr = {"Content-Type": "application/json"}
    if token:
        hdr["Authorization"] = "Bearer " + token
    conn.request("POST", "/v1/chat/completions", json.dumps(body), hdr)
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError("http %d %s" % (resp.status, resp.read()[:300]))
    content, reasoning, calls, timings, usage, finish, ttft = "", "", {}, None, None, None, None
    while True:
        if time.time() > deadline:
            conn.close()
            raise TimeoutError()
        if conn.sock:
            conn.sock.settimeout(max(0.5, deadline - time.time()))
        try:
            line = resp.readline()
        except socket.timeout:
            conn.close()
            raise TimeoutError()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        ev = json.loads(data)
        if ev.get("timings"):
            timings = ev["timings"]
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                ttft = ttft if ttft is not None else time.time() - t0
            content += d.get("content") or ""
            reasoning += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), dict(id=None, name="", args=""))
                c["id"] = tc.get("id") or c["id"]
                f = tc.get("function") or {}
                c["name"] += f.get("name") or ""
                c["args"] += f.get("arguments") or ""
            finish = ch.get("finish_reason") or finish
    conn.close()
    return dict(content=content, reasoning=reasoning, calls=[calls[k] for k in sorted(calls)], timings=timings, usage=usage, finish=finish, ttft=ttft, wall=time.time() - t0)


def run_task(args, token, task, outdir):
    ws = tempfile.mkdtemp(prefix="pilot_%s_" % task["name"])
    ws = os.path.realpath(ws)
    if task.get("setup"):
        task["setup"](ws)
    else:
        for rel, txt in task["files"].items():
            os.makedirs(os.path.dirname(os.path.join(ws, rel)) or ws, exist_ok=True)
            open(os.path.join(ws, rel), "w").write(txt)
    msgs = [dict(role="system", content=SYSTEM), dict(role="user", content=task["prompt"])]
    t_start = time.time()
    deadline = t_start + args.task_timeout
    turns, stats = [], dict(calls=0, bad_calls=0, tool_exec_errors=0, writes=0, fail_tests_after_write=0)
    status, final = "turn_limit", ""
    wrote = False
    transcript = open(os.path.join(outdir, "%s_%s.transcript.jsonl" % (args.label, task["name"])), "w")
    for turn in range(args.max_turns):
        body = dict(model=args.model, messages=msgs, tools=TOOLS, temperature=0, seed=0, max_tokens=args.max_tokens, stream=True, stream_options=dict(include_usage=True))
        try:
            r = chat(args.host, args.port, token, body, deadline)
        except TimeoutError:
            status = "timeout"
            turns.append(dict(turn=turn, aborted="timeout"))
            break
        except Exception as e:
            status = "error: %s" % str(e)[:200]
            break
        tm = r["timings"] or {}
        us = r["usage"] or {}
        turns.append(dict(turn=turn, ttft_s=r["ttft"], wall_s=r["wall"], prompt_n=tm.get("prompt_n", us.get("prompt_tokens")), cache_n=tm.get("cache_n"),
                          predicted_n=tm.get("predicted_n", us.get("completion_tokens")), prompt_tps=tm.get("prompt_per_second"), decode_tps=tm.get("predicted_per_second"),
                          draft_n=tm.get("draft_n"), draft_accepted=tm.get("draft_n_accepted"), reasoning_chars=len(r["reasoning"]), content_chars=len(r["content"]),
                          n_calls=len(r["calls"]), finish=r["finish"]))
        transcript.write(json.dumps(dict(turn=turn, content=r["content"], reasoning=r["reasoning"][:4000], calls=r["calls"])) + "\n")
        am = dict(role="assistant", content=r["content"] or "")
        if r["reasoning"]:
            am["reasoning_content"] = r["reasoning"]
        if r["calls"]:
            for c in r["calls"]:
                c["id"] = c["id"] or "call_" + uuid.uuid4().hex[:8]
            am["tool_calls"] = [dict(id=c["id"], type="function", function=dict(name=c["name"], arguments=c["args"])) for c in r["calls"]]
        msgs.append(am)
        if not r["calls"]:
            final = r["content"]
            status = "answered" if r["finish"] == "stop" else "truncated(%s)" % r["finish"]
            break
        for c in r["calls"]:
            stats["calls"] += 1
            try:
                a = json.loads(c["args"] or "{}")
                out = run_tool(ws, c["name"], a)
                if c["name"] in ("write_file", "replace_in_file"):
                    stats["writes"] += 1
                    wrote = True
                if c["name"] == "run_command" and wrote and "unittest" in a.get("command", "") and not out.startswith("exit=0"):
                    stats["fail_tests_after_write"] += 1
            except Bad as e:
                stats["bad_calls"] += 1
                out = "error: %s" % e
            except json.JSONDecodeError:
                stats["bad_calls"] += 1
                out = "error: arguments are not valid JSON"
            except Exception as e:
                stats["tool_exec_errors"] += 1
                out = "error: %s" % str(e)[:300]
            msgs.append(dict(role="tool", tool_call_id=c["id"], content=out))
        if time.time() > deadline:
            status = "timeout"
            break
    wall = time.time() - t_start
    transcript.close()
    p, n, checks = grade(task, ws, final)
    passed = n > 0 and p == n
    tm_turns = [t for t in turns if "prompt_n" in t and t.get("prompt_n") is not None]
    later = [t for t in tm_turns[1:] if t.get("cache_n") is not None]
    reuse = (sum(t["cache_n"] for t in later) / max(1, sum(t["cache_n"] + t["prompt_n"] for t in later))) if later else None
    dec = [t for t in tm_turns if t.get("predicted_n") and t.get("decode_tps")]
    res = dict(label=args.label, task=task["name"], status=status, tests_passed=p, tests_total=n, passed=passed,
               first_attempt=bool(passed and stats["fail_tests_after_write"] == 0 and status == "answered"), checks=checks, wall_s=round(wall, 1), turns=len(turns),
               completion_tokens=sum(t.get("predicted_n") or 0 for t in turns), prompt_tokens_total=sum(t.get("prompt_n") or 0 for t in turns),
               cached_tokens_total=sum(t.get("cache_n") or 0 for t in turns), prefix_reuse_turns2plus=None if reuse is None else round(reuse, 3),
               ttft_first_s=turns[0].get("ttft_s") if turns else None, ttft_median_s=sorted(t["ttft_s"] for t in turns if t.get("ttft_s") is not None)[len([1 for t in turns if t.get("ttft_s") is not None]) // 2] if any(t.get("ttft_s") for t in turns) else None,
               decode_tps_mean=round(sum(t["predicted_n"] * 1.0 for t in dec) / max(1e-9, sum(t["predicted_n"] / t["decode_tps"] for t in dec)), 1) if dec else None,
               prefill_tps_max=max([t["prompt_tps"] for t in tm_turns if t.get("prompt_tps") and (t.get("prompt_n") or 0) >= 100] or [0]) or None,
               reasoning_chars=sum(t.get("reasoning_chars", 0) for t in turns), tool=stats, per_turn=turns, final_answer=final[:600])
    shutil.rmtree(ws, ignore_errors=True)
    return res


def selftest():
    ok = True
    for name, f in TASKS.items():
        t = f()
        ws = os.path.realpath(tempfile.mkdtemp(prefix="selftest_"))
        for rel, txt in t["files"].items():
            os.makedirs(os.path.dirname(os.path.join(ws, rel)) or ws, exist_ok=True)
            open(os.path.join(ws, rel), "w").write(txt)
        answer = "Retried statuses: 429, 500, 502, 503 (config.py). Delays 0.25s, 0.75s, 2.0s (profiles.py interactive, retry.py). Total 3.0 seconds." if name == "explore" else ""
        before = grade(t, ws, "")[:2] if name != "explore" else grade(t, ws, "unknown")[:2]
        for rel, txt in t["ref"].items():
            open(os.path.join(ws, rel), "w").write(txt)
        after = grade(t, ws, answer)[:2]
        good = after[0] == after[1] and before[0] < before[1]
        ok &= good
        print(name, "before", before, "after-reference", after, "OK" if good else "BAD")
        shutil.rmtree(ws)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label"); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int)
    ap.add_argument("--model", default="local"); ap.add_argument("--cred"); ap.add_argument("--task-timeout", type=float, default=135)
    ap.add_argument("--max-turns", type=int, default=14); ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--tasks", default="bugfix,feature,multifile,explore"); ap.add_argument("--out", default="results"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--session-budget", type=float, default=660, help="stop starting tasks after this many seconds")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    token = None
    if args.cred:
        token = json.load(open(os.path.expanduser("$CREDENTIALS_FILE")))[args.cred]
    os.makedirs(args.out, exist_ok=True)
    # warm-up: tiny request, timed separately (includes a cold model load when the backend was unloaded)
    t0 = time.time()
    try:
        chat(args.host, args.port, token, dict(model=args.model, messages=[dict(role="user", content="Say OK.")], temperature=0, max_tokens=8, stream=True, stream_options=dict(include_usage=True)), time.time() + 300)
        warm = round(time.time() - t0, 1)
    except Exception as e:
        warm = "failed: %s" % e
    print("warmup_s", warm, flush=True)
    session_t0 = time.time()
    results = []
    for name in args.tasks.split(","):
        if time.time() - session_t0 > args.session_budget:
            results.append(dict(label=args.label, task=name, status="skipped_budget"))
            continue
        print("task", name, flush=True)
        r = run_task(args, token, TASKS[name](), args.out)
        r["warmup_s"] = warm
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("per_turn", "final_answer")}), flush=True)
        json.dump(results, open(os.path.join(args.out, "%s_results.json" % args.label), "w"), indent=1)
    json.dump(results, open(os.path.join(args.out, "%s_results.json" % args.label), "w"), indent=1)
    print("session_s", round(time.time() - session_t0, 1))


if __name__ == "__main__":
    main()
