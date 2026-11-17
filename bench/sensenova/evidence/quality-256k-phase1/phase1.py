"""Phase 1 only. Six exact-match lookups in one near-filled synthetic archive."""
import hashlib
import json
import os
import random
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = Path.home() / "models/sensenova-runtime"
CONTEXT = 262144
TARGET = 261632
SEED = 42
DEPTHS = (5, 20, 40, 60, 80, 95)
PORT = 18931
OUTPUT_LIMIT = 64


def save(name, value):
    target = OUT / name
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(target)


def request(path, data=None, timeout=900):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", body,
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def api(path, data=None, timeout=900):
    with request(path, data, timeout) as res:
        return json.load(res)


def tokenize(text):
    return api("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]


def reference(rng):
    return "-".join("".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=4)) for _ in range(3))


def record(rng, ident, state, source, alias, code):
    return (f"Record {ident}. Facility {source}. Shipment class {alias}. Entry status {state}. "
            f"Release reference {code}. Inspection {rng.choice(('routine', 'manual', 'deferred'))}. "
            f"Batch {rng.randrange(1000, 9999)} was logged by the receiving desk.\n")


def fixtures():
    rng = random.Random(SEED)
    sites = ("Cedar", "Juniper", "Harbor", "Mesa", "Northgate", "Riverbend")
    classes = ("relay", "gasket", "panel", "sensor", "motor", "switch")
    scaffold = "SHAREDARCHIVEPLACEHOLDER"
    cases = []
    common_head = None
    for i, depth in enumerate(DEPTHS):
        ident = f"S{7731 + i * 127}"
        site, alias = sites[i], classes[i]
        expected = reference(rng)
        question = (f"What is the release reference in the APPROVED record for shipment {ident} "
                    f"at {site}, class {alias}? Copy the reference exactly. Reply with only the reference, "
                    "without punctuation, markup, or explanation.")
        user = f"Shipment release archive\n\n{scaffold}\n\nEnd of archive.\n\n{question}"
        prompt = api("/apply-template", {"messages": [
            {"role": "system", "content": "Answer using the supplied shipment archive. Only APPROVED entries are authoritative."},
            {"role": "user", "content": user}
        ], "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        if prompt.count(scaffold) != 1:
            raise RuntimeError("Chat template did not preserve archive placeholder")
        head, tail = prompt.split(scaffold)
        if common_head is not None and head != common_head:
            raise RuntimeError("Chat template changed the common prefix")
        common_head = head
        cases.append({"depth_percent": depth, "shipment": ident, "question": question,
                      "expected": expected, "tail_text": tail, "tail_ids": tokenize(tail),
                      "approved": record(rng, ident, "APPROVED", site, alias, expected),
                      "draft": record(rng, ident, "DRAFT", site, alias, reference(rng)),
                      "revoked": record(rng, ident, "REVOKED", site, alias, reference(rng))})
    head_ids = tokenize(common_head)
    filler = tokenize("".join(record(rng, f"B{i:06}", rng.choice(("APPROVED", "DRAFT", "REVOKED")),
                                    rng.choice(sites), rng.choice(classes), reference(rng))
                              for i in range(1, 6601)))
    shared_size = TARGET - max(len(case["tail_ids"]) for case in cases)
    ids = head_ids[:]
    cursor = 0

    def fill(size):
        nonlocal cursor
        count = size - len(ids)
        if count < 0 or cursor + count > len(filler):
            raise RuntimeError("Unable to place fixture at requested position")
        ids.extend(filler[cursor:cursor + count])
        cursor += count

    for case in cases:
        draft = tokenize("\n" + case["draft"])
        approved = tokenize("\n" + case["approved"])
        revoked = tokenize("\n" + case["revoked"])
        position = round(TARGET * case["depth_percent"] / 100)
        fill(position - len(draft))
        ids.extend(draft)
        case["needle_token_offset"] = len(ids)
        ids.extend(approved + revoked)
    fill(shared_size)
    save("shared-prefix-tokens.json", ids)
    rendered = api("/detokenize", {"tokens": ids})["content"]
    for case in cases:
        if rendered.count(case["expected"]) != 1:
            raise RuntimeError("Expected reference is not unique in the archive")
        case["target_input_tokens"] = len(ids) + len(case["tail_ids"])
        case["actual_depth_percent"] = 100 * case["needle_token_offset"] / case["target_input_tokens"]
        flat = json.dumps(ids + case["tail_ids"], separators=(",", ":")).encode()
        case["input_id_sha256"] = hashlib.sha256(flat).hexdigest()
    (OUT / "shared-prefix.txt").write_text(rendered, encoding="utf-8")
    save("fixtures.json", cases)
    return ids, cases


def complete(ids, cache_prompt):
    start = time.monotonic()
    pieces, output_ids = [], []
    ttft, final = None, None
    payload = {"prompt": ids, "cache_prompt": cache_prompt, "stream": True,
               "n_predict": OUTPUT_LIMIT, "temperature": 0, "seed": SEED,
               "return_tokens": True, "ignore_eos": False}
    with request("/completion", payload) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            item = json.loads(line[6:])
            if "error" in item:
                raise RuntimeError(item["error"])
            new = item.get("tokens", [])
            piece = item.get("content", "")
            if ttft is None and (new or piece):
                ttft = time.monotonic() - start
            output_ids.extend(new)
            pieces.append(piece)
            if item.get("stop"):
                final = item
    if final is None:
        raise RuntimeError("No terminal completion")
    timings = final["timings"]
    fresh, cached = timings["prompt_n"], timings["cache_n"]
    if fresh + cached != len(ids) or final.get("truncated"):
        raise RuntimeError(f"Context integrity failure: {timings}")
    return {"mode": "cached" if cached else "cold", "input_tokens": fresh + cached,
            "fresh_tokens": fresh, "cached_tokens": cached,
            "context_utilization_percent": 100 * (fresh + cached) / CONTEXT,
            "output_tokens": timings["predicted_n"], "output_token_ids": output_ids,
            "output": "".join(pieces), "stop_type": final.get("stop_type"),
            "ttft_s": ttft, "wall_s": time.monotonic() - start,
            "prefill_tok_s": timings["prompt_per_second"],
            "decode_tok_s": timings["predicted_per_second"], "native_timings": timings}


def aggregate(rows):
    groups = {}
    for mode in ("cold", "cached"):
        selected = [row for row in rows if row["mode"] == mode]
        if selected:
            groups[mode] = {"requests": len(selected), "correct": sum(row["correct"] for row in selected)}
            for key in ("ttft_s", "prefill_tok_s", "decode_tok_s", "input_tokens", "fresh_tokens", "cached_tokens"):
                values = [row[key] for row in selected if row[key] is not None]
                groups[mode]["median_" + key] = statistics.median(values) if values else None
    return {"completed": len(rows), "correct": sum(row["correct"] for row in rows), "by_cache_mode": groups}


def main():
    binary = ROOT / "build-cu133/bin/llama-server"
    model = ROOT / "SenseNova-Loop-9B-Q4_K_M.gguf"
    command = [str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(PORT),
               "--device", "CUDA0", "--fit", "off", "-lv", "5", "-ngl", "99", "-np", "1",
               "-c", str(CONTEXT), "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
               "-b", "2048", "-ub", "1024", "--no-warmup"]
    install = ROOT / "build-cu133/bin/libggml-cuda.so.0"
    snapshot = ROOT / "256k-baseline-lib/libggml-cuda.so.0"
    sha = hashlib.file_digest(install.open("rb"), "sha256").hexdigest()
    if sha != hashlib.file_digest(snapshot.open("rb"), "sha256").hexdigest():
        raise RuntimeError("Installed CUDA runtime differs from the validated snapshot")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-13.3/targets/x86_64-linux/lib:/usr/lib/wsl/lib"
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", PORT)) == 0:
            raise RuntimeError("Benchmark port is occupied")
    summary = {"phase": 1, "category": "synthetic archive exact retrieval",
               "seed": SEED, "temperature": 0, "context_capacity": CONTEXT,
               "sha256_cuda_library": sha, "runtime_command": command, "tests": []}
    with (OUT / "engine.log").open("x") as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline and server.poll() is None:
                try:
                    if api("/health", timeout=2).get("status") == "ok":
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError("Server did not become ready. See engine.log")
            text = (OUT / "engine.log").read_text(errors="replace")
            for marker in ("offloaded 57/57 layers to GPU", "n_ctx_slot = 262144", "K (q8_0)", "V (q8_0)"):
                if marker not in text:
                    raise RuntimeError("Runtime configuration check failed: " + marker)
            shared, cases = fixtures()
            print("Fixtures verified, first cold request starting.", flush=True)
            for i, case in enumerate(cases):
                row = complete(shared + case["tail_ids"], cache_prompt=i > 0)
                row.update({key: case[key] for key in ("depth_percent", "actual_depth_percent",
                           "needle_token_offset", "expected", "question", "input_id_sha256")})
                row["correct"] = row["output"].strip() == row["expected"]
                summary["tests"].append(row)
                summary["summary"] = aggregate(summary["tests"])
                save("results.json", summary)
                print(row["depth_percent"], row["mode"], row["correct"], repr(row["output"]),
                      round(row["ttft_s"], 3) if row["ttft_s"] is not None else None, flush=True)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *args: sys.exit(143))
    main()
