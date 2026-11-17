set -e
r=/mnt/f/MODELS/SenseNova-Loop-9B/optimization
rt=~/models/sensenova-runtime
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/targets/x86_64-linux/lib:/usr/lib/wsl/lib
normal_lib=$LD_LIBRARY_PATH
old_lib="$rt/pre-sinkhorn-cuda:$normal_lib"
python3 - "$r" <<'PY'
import json, subprocess, sys, time
from pathlib import Path
root = Path(sys.argv[1])
deadline = time.monotonic() + 1400
while time.monotonic() < deadline:
    try:
        report = json.loads((root / "baseline-262144/results.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        report = {}
    if len(report.get("runs", [])) == 2 and subprocess.run(
            ["pgrep", "-x", "llama-server"], stdout=subprocess.DEVNULL).returncode == 1:
        break
    time.sleep(3)
else:
    raise SystemExit("Baseline did not finish. Candidate stages were not launched.")
PY
mv -T "$r/baseline-65536" "$r/baseline-65536-build-overlap"
LD_LIBRARY_PATH="$old_lib" python3 -u "$r/benchmark.py" 65536 "$r/baseline-65536"
nsys=/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
python3 -u "$r/benchmark.py" 8192 "$r/profile-sinkhorn" --input 4096 --output 64 --reps 1 --nsys "$nsys" --reference "$r/profile-baseline/results.json"
"$nsys" stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv --output "$r/profile-sinkhorn/summary" "$r/profile-sinkhorn/trace.nsys-rep"
python3 -u "$r/benchmark.py" 65536 "$r/sinkhorn-65536" --reference "$r/baseline-65536/results.json"
variant=$(python3 - "$r" <<'PY'
import json, sys
from pathlib import Path
r = Path(sys.argv[1])
base = json.loads((r / "baseline-65536/results.json").read_text())
cand = json.loads((r / "sinkhorn-65536/results.json").read_text())
same = all(row["mismatched_tokens"] == 0 for row in cand["runs"])
# Both comparisons use the same prompt, KV and microbatch. Proceed with the faster equal-output path.
print("sinkhorn" if same and cand["median"]["wall_s"] < base["median"]["wall_s"] else "baseline")
PY
)
printf '%s\n' "$variant" > "$r/sweep-routing.txt"
if [ "$variant" = baseline ]; then export LD_LIBRARY_PATH="$old_lib"; fi
for batch in 2048 512 256; do
    python3 -u "$r/benchmark.py" 65536 "$r/ubatch-$batch" --ubatch "$batch" --reference "$r/baseline-65536/results.json"
done
batch=$(python3 - "$r" "$variant" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
variant = sys.argv[2]
runs = {}
for n in (256, 512, 1024, 2048):
    name = f"{variant}-65536" if n == 1024 else f"ubatch-{n}"
    runs[n] = json.loads((root / name / "results.json").read_text())
equal = [n for n, report in runs.items()
         if all(row.get("mismatched_tokens", 0) == 0 for row in report["runs"])]
print(min(equal, key=lambda n: runs[n]["median"]["wall_s"]))
PY
)
printf 'routing=%s\nubatch=%s\n' "$variant" "$batch" | tee "$r/selected.txt"
if [ "$variant" = baseline ] && [ "$batch" = 1024 ]; then exit 0; fi
for c in 131072 262144; do
    if ! python3 -u "$r/benchmark.py" "$c" "$r/final-$c" --ubatch "$batch" --reference "$r/baseline-$c/results.json"; then
        if ! grep -iqE 'out of memory|failed to allocate|alloc.*failed' "$r/final-$c/engine.log"; then exit 1; fi
        if [ "$batch" = 1024 ]; then exit 1; fi
        python3 -u "$r/benchmark.py" "$c" "$r/final-$c-ub1024" --ubatch 1024 --reference "$r/baseline-$c/results.json"
    fi
done
