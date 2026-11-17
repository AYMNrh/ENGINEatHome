set -e
r=/mnt/f/MODELS/SenseNova-Loop-9B/optimization-256k
reference=/mnt/f/MODELS/SenseNova-Loop-9B/optimization/baseline-262144/results.json
export LD_LIBRARY_PATH="$HOME/models/sensenova-runtime/256k-j64-lib:/usr/local/cuda-13.3/targets/x86_64-linux/lib:/usr/lib/wsl/lib"
export GGML_CUDA_FUSED_Q4K_J64=1
python3 - "$r" <<'PY'
import json, subprocess, sys, time
from pathlib import Path
out = Path(sys.argv[1])
deadline = time.monotonic() + 1100
while time.monotonic() < deadline:
    try:
        raw = json.loads((out / "matched-baseline/results.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        raw = {}
    if len(raw.get("runs", [])) == 6 and subprocess.run(
            ["pgrep", "-x", "llama-server"], stdout=subprocess.DEVNULL).returncode == 1:
        if any(run["mismatched_tokens"] for run in raw["runs"]):
            raise SystemExit("Baseline mismatch, inspect cached/frozen results before proceeding")
        break
    time.sleep(4)
else:
    raise SystemExit("Matched baseline did not finish")
PY
python3 -u "$r/benchmark.py" 262144 "$r/mmq-j64" --frozen-prompt "$r/frozen-prompt.json" --reference "$reference"
nsys=/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
python3 -u "$r/benchmark.py" 262144 "$r/nsys-mmq-j64" --frozen-prompt "$r/frozen-prompt.json" --reps 1 --no-probs --cold-only --reference "$reference" --nsys "$nsys"
"$nsys" stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv --output "$r/nsys-mmq-j64/summary" "$r/nsys-mmq-j64/trace.nsys-rep"
