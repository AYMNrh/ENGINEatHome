"""SenseNova Phase 2 multi-hop reasoning benchmark."""
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
import datetime as dt
import threading
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = Path.home() / "models/sensenova-runtime"
CONTEXT = 262144
TARGET = 261632
SEED = 42
PORT = 18931


def save(name, value):
    target = OUT / name
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(target)


OUTPUT_LIMIT = 128


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


def archive_cases():
    temporal = [
        ("T01 deployment record", "Signed production record for Orion. Activation occurred on 2031-05-12 at 09:00 UTC. The previous dry run was 2031-05-10 at 09:00 UTC."),
        ("T02 signoff register", "Orion policy N17 was signed on 2031-05-01. N21 was signed on 2031-05-11. N24 was signed on 2031-05-13. A sales note from May 9 announced N24 as final, but it is not a signature."),
        ("T03 retention table", "For Orion, N17 requires 18 hours, N21 requires 30 hours, and N24 requires 48 hours. An obsolete copy labeled N21 says 24 hours. This signed table replaces it."),
        ("T04 operations bulletin", "Select the latest signed Orion policy as of the production activation. Apply its retention interval starting at activation. Routing after retention uses lane Mica for intervals below 24 hours or Sable for intervals of at least 24 hours. The rehearsal used lane Slate."),
    ]
    signed = {"N17": "2031-05-01", "N21": "2031-05-11", "N24": "2031-05-13"}
    activation = dt.datetime.fromisoformat("2031-05-12T09:00:00+00:00")
    policy = max((key for key, date in signed.items() if date <= activation.date().isoformat()),
                 key=signed.__getitem__)
    hours = {"N17": 18, "N21": 30, "N24": 48}[policy]
    temporal_answer = {"policy": policy, "release_utc": (activation + dt.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%MZ"),
                       "lane": "Mica" if hours < 24 else "Sable"}

    dependency = [
        ("D01 release manifest", "Atlas production builds must select the newest compatible Relay version in major 3. An archived Atlas 6 manifest permits Relay 2; a preview Atlas 8 manifest permits Relay 4."),
        ("D02 Relay catalog", "Relay 3.2 requires Bridge >=2.0 <3.0. Relay 3.5 requires Bridge >=2.2 <3.0. Relay 3.8 requires Bridge >=2.4 <3.0. A deprecated memo claims Relay 3.8 works with Bridge 2.3; the catalog governs."),
        ("D03 Bridge catalog", "Bridge 2.1 requires driver ABI 6, Bridge 2.3 requires ABI 7, Bridge 2.4 requires ABI 8, and Bridge 3.0 requires ABI 7. Bridge 2.3 ships with codec Lumen and Bridge 2.4 ships with codec Pollen. For each Relay candidate prefer the highest compatible Bridge version."),
        ("D04 host attestation", "The signed Atlas production host attestation is ABI 7. The staging host runs ABI 8; an old inventory lists ABI 6 for production."),
    ]
    relay = [(32, 20), (35, 22), (38, 24)]
    bridges = [(21, 6, "Quartz"), (23, 7, "Lumen"), (24, 8, "Pollen"), (30, 7, "Flint")]
    r, b, codec = max((r, b, codec) for r, minimum in relay for b, abi, codec in bridges
                      if minimum <= b < 30 and abi == 7)
    dependency_answer = {"relay": f"{r//10}.{r%10}", "bridge": f"{b//10}.{b%10}", "codec": codec}

    overrides = [
        ("O01 live base specification", "Beryl production base values are workers=4, timeout_seconds=30, endpoint=jade. An obsolete baseline lists workers=12 and endpoint=topaz."),
        ("O02 deployment overlay", "The live region overlay for west is workers=8 and endpoint=onyx. The live region east overlay sets endpoint=pearl. Beryl tenant Kite is deployed in west."),
        ("O03 tenant ledger", "The current Kite override is endpoint=sable and timeout_seconds=45. Kite's discarded draft set workers=16. A different tenant, Finch, sets timeout_seconds=90."),
        ("O04 override contract and incident", "Apply base then region then active tenant then active emergency. Emergency is allowed to change timeout_seconds only. At 2031-06-10T15:30Z, incident E7 is active from 15:00 through 16:00 and proposes timeout_seconds=20, workers=2, endpoint=quartz. The earlier incident E6 expired at 14:00."),
    ]
    override_answer = {"workers": 4, "timeout_seconds": 30, "endpoint": "jade"}
    override_answer.update(workers=8, endpoint="onyx")
    override_answer.update(endpoint="sable", timeout_seconds=45)
    override_answer.update(timeout_seconds=20)

    synthesis = [
        ("S01 destination registry", "The current alias registry maps destination D-417 to depot Alder. Its superseded export maps D-417 to Birch. Depot Alder uses dispatch region North."),
        ("S02 inventory return", "Alder has 17 grade-A seal sets and 11 grade-B seal sets on hand. Alder has 4 grade-A sets already reserved. Birch has 29 grade-A sets and 2 reserved. A preliminary Alder inventory reported 20 grade-A sets before reconciliation."),
        ("S03 clearance certificates", "Site-level quality certificates allow grade A at Alder, and grade B at Birch. An expired Alder waiver allowed grade B. The in-force certificates supersede all waiver exports."),
        ("S04 dispatch agreement", "Use the inventory at the depot resolved from the current alias registry. Subtract reserved sets of the authorized grade. Region North uses carrier Tern below 15 available sets and carrier Siskin at 15 or more. Region South uses carrier Lark. A draft North agreement used a threshold of 12."),
    ]
    depot, region = {"D-417": ("Alder", "North")}["D-417"]
    grade = {"Alder": "A", "Birch": "B"}[depot]
    stock, reservations = {("Alder", "A"): (17, 4)}[depot, grade]
    available = stock - reservations
    synthesis_answer = {"depot": depot, "authorized_grade": grade, "available_sets": available,
                        "carrier": "Tern" if region == "North" and available < 15 else "Siskin"}

    return [
        ("temporal", temporal, temporal_answer,
         "For Orion production, determine the applicable policy ID, release timestamp in UTC as YYYY-MM-DDTHH:MMZ, and lane.",
         "Join production activation with the signatures, signed retention table, and release-routing rule."),
        ("dependency", dependency, dependency_answer,
         "For Atlas production, resolve Relay version, Bridge version, and the selected Bridge codec. Encode versions as strings.",
         "Combine allowed major, Relay constraints, Bridge catalog and production ABI."),
        ("overrides", overrides, override_answer,
         "For Beryl tenant Kite at 2031-06-10T15:30Z, give the effective workers, timeout_seconds, and endpoint.",
         "Apply west region and Kite's live record in precedence order, then honor E7 only for its allowed key."),
        ("synthesis", synthesis, synthesis_answer,
         "For destination D-417, determine depot, authorized_grade, available_sets, and carrier.",
         "Resolve location and region, select the valid grade, subtract reserved inventory, and apply the active carrier threshold."),
    ]


def fixtures():
    rng = random.Random(SEED)
    archive = archive_cases()
    cases = []
    scaffold = "SHAREDARCHIVEPLACEHOLDER"
    common_head = None
    for name, docs, expected, question, derivation in archive:
        question += " Return one JSON object only with keys " + ", ".join(expected) + "."
        user = f"Operations reference archive\n\n{scaffold}\n\nEnd of archive.\n\n{question}"
        prompt = api("/apply-template", {"messages": [
            {"role": "system", "content": "Resolve the question from the archive. Follow effective dates, scope, signed revisions and authority."},
            {"role": "user", "content": user}
        ], "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        if prompt.count(scaffold) != 1:
            raise RuntimeError("Chat template did not preserve archive placeholder")
        head, tail = prompt.split(scaffold)
        if common_head is not None and head != common_head:
            raise RuntimeError("Chat template changed the common prefix")
        common_head = head
        cases.append({"category": name, "question": question, "expected": expected,
                      "derivation": derivation, "evidence": [{"title": label, "text": text}
                      for label, text in docs], "tail_text": tail, "tail_ids": tokenize(tail)})
    ids = tokenize(common_head)
    filler = tokenize("".join(record(rng, f"B{i:06}", rng.choice(("APPROVED", "DRAFT", "REVOKED")),
                                    rng.choice(("Cedar", "Juniper", "Harbor", "Mesa")),
                                    rng.choice(("relay", "gasket", "panel", "sensor")), reference(rng))
                              for i in range(1, 6601)))
    cursor = 0

    def fill(size):
        nonlocal cursor
        count = size - len(ids)
        if count < 0 or cursor + count > len(filler):
            raise RuntimeError("Unable to place evidence")
        ids.extend(filler[cursor:cursor + count])
        cursor += count

    for i in range(4):
        for j, case in enumerate(cases):
            section = case["evidence"][i]
            position = round(TARGET * (4 + i * 24 + j * 6) / 100)
            fill(position)
            section["token_offset"] = len(ids)
            ids.extend(tokenize("\n\n" + section["title"] + "\n" + section["text"] + "\n\n"))
    fill(TARGET - max(len(case["tail_ids"]) for case in cases))
    rendered = api("/detokenize", {"tokens": ids})["content"]
    for case in cases:
        for section in case["evidence"]:
            if rendered.count(section["title"]) != 1 or rendered.count(section["text"]) != 1:
                raise RuntimeError("Evidence section missing or duplicated")
        case["target_input_tokens"] = len(ids) + len(case["tail_ids"])
        for section in case["evidence"]:
            section["depth_percent"] = 100 * section["token_offset"] / case["target_input_tokens"]
        flat = json.dumps(ids + case["tail_ids"], separators=(",", ":")).encode()
        case["input_id_sha256"] = hashlib.sha256(flat).hexdigest()
    (OUT / "shared-prefix.txt").write_text(rendered, encoding="utf-8")
    save("shared-prefix-tokens.json", ids)
    save("fixtures.json", cases)
    return ids, cases


def grade(output, expected):
    def unique_keys(pairs):
        values = {}
        for key, value in pairs:
            if key in values:
                raise ValueError(f"Duplicate JSON key {key}")
            values[key] = value
        return values

    try:
        observed = json.loads(output, object_pairs_hook=unique_keys)
    except ValueError as exc:
        return {"correct": False, "observed": None, "errors": [f"Invalid JSON: {exc}"]}
    if not isinstance(observed, dict):
        return {"correct": False, "observed": observed, "errors": ["Answer is not a JSON object"]}
    errors = [f"{key}: expected {value!r}, observed {observed.get(key)!r}"
              for key, value in expected.items()
              if type(observed.get(key)) is not type(value) or observed.get(key) != value]
    errors.extend(f"Unexpected key {key}" for key in observed.keys() - expected.keys())
    return {"correct": not errors, "observed": observed, "errors": errors}


class Monitor:
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                               "--format=csv,noheader,nounits"], text=True, timeout=4)
                self.samples.append((time.monotonic(), int(raw.splitlines()[0])))
            except (ValueError, IndexError, OSError, subprocess.SubprocessError) as exc:
                print("GPU telemetry unavailable:", exc, flush=True)
                return
            self.stop.wait(0.25)


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
    summary = {"phase": 2, "category": "synthetic archive multi-hop reasoning",
               "seed": SEED, "temperature": 0, "context_capacity": CONTEXT,
               "sha256_cuda_library": sha, "runtime_command": command, "tests": []}
    with (OUT / "engine.log").open("x") as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        monitor = Monitor()
        monitor.thread.start()
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
            if not monitor.samples:
                print("Waiting for first GPU sample.", flush=True)
            for i, case in enumerate(cases):
                begin = time.monotonic()
                row = complete(shared + case["tail_ids"], cache_prompt=i > 0)
                end = time.monotonic()
                samples = [value for t, value in monitor.samples if begin <= t <= end]
                row["peak_device_vram_mib"] = max(samples, default=None)
                row["vram_samples"] = len(samples)
                row.update({key: case[key] for key in ("category", "evidence",
                           "derivation", "expected", "question", "input_id_sha256")})
                row.update(grade(row["output"], row["expected"]))
                summary["tests"].append(row)
                summary["summary"] = aggregate(summary["tests"])
                save("results.json", summary)
                print(row["category"], row["mode"], row["correct"], repr(row["output"]),
                      round(row["ttft_s"], 3) if row["ttft_s"] is not None else None, flush=True)
        finally:
            monitor.stop.set()
            monitor.thread.join(timeout=5)
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
