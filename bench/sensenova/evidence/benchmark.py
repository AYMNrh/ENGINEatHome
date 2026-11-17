"""Run one cold SenseNova Loop 9B speed probe on the local CUDA server."""
import json
import re
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parent
engine = Path.home() / "models/sensenova-runtime/build-cu133/bin/llama-server"
model = engine.parents[2] / "SenseNova-Loop-9B-Q4_K_M.gguf"
base = "http://127.0.0.1:18931"
command = [str(engine), "-m", str(model), "--host", "127.0.0.1", "--port", "18931",
           "--device", "CUDA0", "--fit", "off", "-lv", "5", "-ngl", "99", "-c", "65536", "-np", "1", "-fa", "on", "-ctk", "q8_0",
           "-ctv", "q8_0", "-b", "2048", "-ub", "512", "--no-warmup"]


def request(path, data=None, timeout=900):
    if data is None:
        target = base + path
    else:
        target = urllib.request.Request(base + path, json.dumps(data).encode(),
                                        {"Content-Type": "application/json"})
    return urllib.request.urlopen(target, timeout=timeout)


def sample_gpu(stop, samples):
    while not stop.is_set():
        used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits"], text=True, timeout=5)
        samples.append(int(used.splitlines()[0]))
        stop.wait(1)


with socket.socket() as probe:
    if probe.connect_ex(("127.0.0.1", 18931)) == 0:
        raise RuntimeError("Benchmark port is occupied")

with (root / "engine.log").open("x") as log:
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    stop, samples = threading.Event(), []
    monitor = threading.Thread(target=sample_gpu, args=(stop, samples), daemon=True)
    monitor.start()
    try:
        deadline = time.monotonic() + 180
        while True:
            if server.poll() is not None:
                raise RuntimeError("Server exited during setup. See engine.log.")
            if time.monotonic() > deadline:
                raise TimeoutError("Server startup timed out. See engine.log.")
            try:
                with request("/health", timeout=2) as response:
                    if json.load(response).get("status") == "ok":
                        break
            except OSError:
                time.sleep(1)
        state = (root / "engine.log").read_text(errors="replace")
        placement = re.search(r"offloaded (\d+)/(\d+) layers to GPU", state)
        if not placement or placement[1] != placement[2]:
            raise RuntimeError("Full GPU layer placement was not confirmed. See engine.log.")
        filler = "The checkout service reads the inventory before creating a shipment. Each item has a SKU, quantity, and enabled flag.\n"
        text = filler * 6000
        with request("/tokenize", {"content": text, "add_special": False}) as response:
            tokens = json.load(response)["tokens"]
        with request("/tokenize", {"content": "\nList 100 distinct checkout test cases, one per line.\n", "add_special": False}) as response:
            task = json.load(response)["tokens"]
        prompt = tokens[:65024-len(task)] + task
        assert len(prompt) == 65024
        print("Running 65,024 fresh input tokens and 256 generated tokens.", flush=True)
        started = time.monotonic()
        ttft = None
        content = []
        final = None
        with request("/completion", {"prompt": prompt, "stream": True, "n_predict": 256,
                                      "cache_prompt": False, "ignore_eos": True, "temperature": 0,
                                      "seed": 42}) as response:
            for line in response:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                item = json.loads(line[6:])
                if "error" in item:
                    raise RuntimeError(item["error"])
                if item.get("content"):
                    if ttft is None:
                        ttft = time.monotonic() - started
                    content.append(item["content"])
                if item.get("stop"):
                    final = item
        elapsed = time.monotonic() - started
        if final is None:
            raise RuntimeError("Completion stream ended without benchmark timings.")
        timings = final["timings"]
        if timings["prompt_n"] != len(prompt) or timings["predicted_n"] != 256 or final.get("truncated"):
            raise RuntimeError(f"Incomplete benchmark: {timings}")
        result = {"command": command, "timings": timings, "ttft_seconds": ttft,
                  "elapsed_seconds": elapsed, "peak_device_vram_mib": max(samples) if samples else None,
                  "offloaded_model_layers": int(placement[1]), "output_sample": "".join(content)[:400]}
        (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        stop.set()
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

