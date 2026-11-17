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


def completion(prompt, count, cached=False, probabilities=False):
    payload = {"prompt": prompt, "stream": True, "n_predict": count, "cache_prompt": cached, "n_probs": 2 if probabilities else 0,
               "ignore_eos": True, "temperature": 0, "seed": 42, "return_tokens": True}
    start = time.monotonic()
    ttft, final, tokens = None, None, []
    top_two = []
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
            for item_probs in item.get("completion_probabilities", []):
                options = item_probs["top_logprobs"]
                top_two.append({"token": item_probs["id"], "options": options,
                                "margin": options[0]["logprob"] - options[1]["logprob"]})
            if item.get("stop"):
                final = item
    if final is None:
        raise RuntimeError("Completion stream ended without terminal timings")
    timings = final["timings"]
    reuse = timings["cache_n"]
    fresh = timings["prompt_n"]
    if timings["predicted_n"] != count or fresh + reuse != len(prompt):
        raise RuntimeError(f"Unmatched token counts: {timings}")
    if (not cached and reuse != 0) or (cached and reuse == 0):
        raise RuntimeError(f"Unexpected cache reuse: {timings}")
    if len(tokens) != count or final.get("truncated") or ttft is None:
        raise RuntimeError("Missing output IDs or prompt truncation")
    if probabilities and len(top_two) != count:
        raise RuntimeError(f"Missing top-two logprobs: {len(top_two)} of {count}")
    return {"prefill_tps": timings["prompt_per_second"], "decode_tps": timings["predicted_per_second"],
            "ttft_s": ttft, "wall_s": time.monotonic() - start, "output_ids": tokens,
            "cached_tokens": reuse, "fresh_tokens": fresh, "top_two": top_two}


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
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--input", type=int)
    parser.add_argument("--output", type=int, default=256)
    parser.add_argument("--kv", default="q8_0")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--nsys", type=Path)
    parser.add_argument("--profiler-prefix", type=Path)
    parser.add_argument("--frozen-prompt", type=Path, required=True)
    parser.add_argument("--no-probs", action="store_true")
    parser.add_argument("--cold-only", action="store_true")
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
    if args.profiler_prefix:
        command = json.loads(args.profiler_prefix.read_text()) + command
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
            if args.frozen_prompt.is_file():
                if json.loads(args.frozen_prompt.read_text()) != ids:
                    raise RuntimeError("Input IDs differ from frozen fixture")
            else:
                args.frozen_prompt.write_text(json.dumps(ids, separators=(",", ":")))
            completion(tokenize("Warm up the CUDA kernels and decoder.\n" * 300)[:2048], 32)
            rows = []
            modes = ("cold",) if args.cold_only else ("cold", "cached")
            for i in range(args.reps):
                for mode in modes:
                    sample_start = len(samples)
                    row = completion(ids, args.output, cached=mode == "cached", probabilities=not args.no_probs)
                    row.update(mode=mode, rep=i+1, peak_vram_mib=max(samples[sample_start:], default=None))
                    if reference:
                        expected = reference["runs"][0]["output_ids"]
                        different = [idx + 1 for idx, (a, b) in enumerate(zip(expected, row["output_ids"])) if a != b]
                        row["mismatched_tokens"] = len(different)
                        row["first_mismatch_token"] = different[0] if different else None
                    rows.append(row)
                    medians = {m: {k: statistics.median(run[k] for run in rows if run["mode"] == m)
                                  for k in ("prefill_tps", "decode_tps", "ttft_s", "wall_s")}
                               for m in modes if any(run["mode"] == m for run in rows)}
                    result = {"context": args.context, "input_tokens": size, "prompt_sha256": prompt_sha,
                              "command": command, "runs": rows, "median": medians,
                              "peak_device_vram_mib": max(samples, default=None)}
                    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
                    print(args.out.name, mode, i+1, {k: v for k, v in row.items() if k not in ("output_ids", "top_two")},
                          flush=True)
        finally:
            stop.set()
            if args.nsys or args.profiler_prefix:
                import os
                import signal
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGINT)
            else:
                server.terminate()
            try:
                server.wait(timeout=90 if args.nsys or args.profiler_prefix else 15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
