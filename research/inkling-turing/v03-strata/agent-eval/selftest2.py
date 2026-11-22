import json, threading, http.server, argparse, os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot2, pilot_real
def sse(chunks):
    return "".join("data: %s\n\n" % json.dumps(c) for c in chunks) + "data: [DONE]\n\n"
def call(name, args, finish="tool_calls"):
    return [dict(choices=[dict(delta=dict(tool_calls=[dict(index=0, id="c1", function=dict(name=name, arguments=args))]))]),
            dict(choices=[dict(delta={}, finish_reason=finish)]), dict(choices=[], timings=dict(prompt_n=100, cache_n=50, predicted_n=20, prompt_per_second=500, predicted_per_second=40), usage=dict(prompt_tokens=100, completion_tokens=20))]
SCRIPT = [
    call("run_command", json.dumps(dict(command="git status --short | head -3; ls | head -3 && python -c 'import urllib.request as u; u.urlopen(\"http://1.1.1.1\", timeout=3)' 2>&1 | tail -1"))),
    call("run_command", '{"command": "echo hi'),
    call("run_command", '{"command": "echo trunc', finish="length"),
    call("nonexistent_tool", "{}"),
    call("read_file", json.dumps(dict(path="../../etc/passwd"))),
    call("run_command", json.dumps(dict(command="exit 3"))),
    [dict(choices=[dict(delta=dict(content="Done."))]), dict(choices=[dict(delta={}, finish_reason="stop")]), dict(choices=[], timings=dict(prompt_n=10, cache_n=900, predicted_n=5), usage={})],
]
state = {"i": 0}
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = sse(SCRIPT[min(state["i"], len(SCRIPT) - 1)]); state["i"] += 1
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers(); self.wfile.write(body.encode())
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 18999), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
pilot_real.TASKS["boltons_barrel"]["ref_passed"] = 8
a = argparse.Namespace(label="mock", host="127.0.0.1", port=18999, model="m", task_timeout=120, max_turns=20, max_tokens=6000, ctx=65536)
os.makedirs("results_mock", exist_ok=True)
r = pilot2.run_task(a, None, "boltons_barrel", "results_mock")
print(r["per_turn"][:1]); print(r["status"], r["tool_categories"], "model_side_errors", r["model_side_errors"], "func", r["functional_pass"], "clean", r["clean_completion"])
for l in open("results_mock/mock_boltons_barrel.toollog.jsonl"):
    d = json.loads(l); print(d["turn"], d["kind"], d.get("category"), (d.get("result") or "")[:100].replace("\n", " | "))
