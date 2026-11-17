"""Matched cold SenseNova speed runs with request warmup and token-ID comparison."""
import argparse
import hashlib
import json
import re
import socket
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


def fetch(path, payload=None, timeout=900):
    url = "http://127.0.0.1:18931" + path
    req = urllib.request.Request(url, None if payload is None else json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def tokenize(prompt):
    with fetch("/tokenize", {"content": prompt, "add_special": False}) as response:
        return json.load(response)["tokens"]


def completion(prompt, count):
    payload = {"prompt": prompt, "stream": True, "n_predict": count, "cache_prompt": False,
               "ignore_eos": True, "temperature": 0, "seed": 42, "return_tokens": True}
    start = time.monotonic()
    ttft, final, tokens = None, None, []
    with fetch("/completion", payload) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            item = json.loads(line[6:])
            if "error" in item:
                raise RuntimeError(item["error"])
            new_tokens = item.get("tokens", [])
            if new_tokens and ttft is None:
                ttft = time.monotonic() - start
            tokens.extend(new_tokens)
            if item.get("stop"):
                final = item
    if final is None:
        raise RuntimeError("Completion stream ended without terminal timings")
    timings = final["timings"]
    if (timings["prompt_n"], timings["predicted_n"], timings["cache_n"]) != (len(prompt), count, 0):
        raise RuntimeError(f"Unmatched token counts: {timings}")
    if len(tokens) != count or final.get("truncated") or ttft is None:
        raise RuntimeError("Missing output IDs or prompt truncation")
    return {"prefill_tps": timings["prompt_per_second"], "decode_tps": timings["predicted_per_second"],
            "ttft_s": ttft, "wall_s": time.monotonic() - start, "output_ids": tokens}


def watch_gpu(stop, samples):
    while not stop.is_set():
        line = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits"], text=True, timeout=5)
        samples.append(int(line.splitlines()[0]))
        stop.wait(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("context", type=int)
    parser.add_argument("out", type=Path)
    parser.add_argument("--ubatch", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--input", type=int)
    parser.add_argument("--output", type=int, default=256)
    parser.add_argument("--kv", default="q8_0")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--nsys", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    runtime = Path.home() / "models/sensenova-runtime"
    engine = runtime / "build-cu133/bin/llama-server"
    model = runtime / "SenseNova-Loop-9B-Q4_K_M.gguf"
    command = [str(engine), "-m", str(model), "--host", "127.0.0.1", "--port", "18931",
               "--device", "CUDA0", "--fit", "off", "-lv", "5", "-ngl", "99", "-np", "1",
               "-c", str(args.context), "-fa", "on", "-ctk", args.kv, "-ctv", args.kv,
               "-b", "2048", "-ub", str(args.ubatch), "--no-warmup"]
    if args.nsys:
        command = [str(args.nsys), "profile", "--trace=cuda,nvtx", "--sample=none",
                   "--cpuctxsw=none", "--cuda-graph-trace=node",
                   "-o", str(args.out / "trace")] + command
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 18931)) == 0:
            raise RuntimeError("Benchmark port is occupied")
    with (args.out / "engine.log").open("x") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        stop, samples = threading.Event(), []
        monitor = threading.Thread(target=watch_gpu, args=(stop, samples), daemon=True)
        monitor.start()
        try:
            deadline = time.monotonic() + 180
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Server startup failed. See engine.log.")
                if time.monotonic() > deadline:
                    raise TimeoutError("Server readiness timed out. See engine.log.")
                try:
                    with fetch("/health", timeout=2) as res:
                        if json.load(res).get("status") == "ok":
                            break
                except OSError:
                    time.sleep(1)
            state = (args.out / "engine.log").read_text(errors="replace")
            if "offloaded 57/57 layers to GPU" not in state or not re.search(r"n_layer\s+= 56\b", state):
                raise RuntimeError("Model visit count or layer placement mismatch")
            if not re.search(r"n_ctx_slot = " + str(args.context) + r"\b", state):
                raise RuntimeError("Context size mismatch")
            size = args.input or args.context - 512
            filler = "The checkout service reads the inventory before creating a shipment. Each item has a SKU, quantity, and enabled flag.\n"
            tail = tokenize("\nList 100 distinct checkout test cases, one per line.\n")
            ids = tokenize(filler * (size // 20 + 1))[:size-len(tail)] + tail
            if len(ids) != size:
                raise RuntimeError("Short prompt")
            prompt_sha = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            reference = json.loads(args.reference.read_text()) if args.reference else None
            if reference and (reference["prompt_sha256"], len(reference["runs"][0]["output_ids"])) != (prompt_sha, args.output):
                raise RuntimeError("Reference prompt or generation cap differs")
            completion(tokenize("Warm up the CUDA kernels and decoder.\n" * 300)[:2048], 32)
            rows = []
            for i in range(args.reps):
                row = completion(ids, args.output)
                if reference:
                    expected = reference["runs"][i % len(reference["runs"])]["output_ids"]
                    different = [idx + 1 for idx, (a, b) in enumerate(zip(expected, row["output_ids"])) if a != b]
                    row["mismatched_tokens"] = len(different)
                    row["first_mismatch_token"] = different[0] if different else None
                rows.append(row)
                medians = {k: statistics.median(run[k] for run in rows)
                           for k in ("prefill_tps", "decode_tps", "ttft_s", "wall_s")}
                result = {"context": args.context, "input_tokens": size, "prompt_sha256": prompt_sha,
                          "command": command, "runs": rows, "median": medians,
                          "peak_device_vram_mib": max(samples) if samples else None}
                (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
                print(args.out.name, "rep", i+1, {k: v for k, v in row.items() if k != "output_ids"},
                      "peak MiB", result["peak_device_vram_mib"], flush=True)
        finally:
            stop.set()
            if args.nsys:
                import os
                import signal
                os.killpg(server.pid, signal.SIGINT)
            else:
                server.terminate()
            try:
                server.wait(timeout=90 if args.nsys else 15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
