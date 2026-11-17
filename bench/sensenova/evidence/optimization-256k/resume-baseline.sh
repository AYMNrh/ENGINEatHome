set -e
r=/mnt/f/MODELS/SenseNova-Loop-9B/optimization-256k
ref=/mnt/f/MODELS/SenseNova-Loop-9B/optimization/baseline-262144/results.json
export LD_LIBRARY_PATH="$HOME/models/sensenova-runtime/256k-baseline-lib:/usr/local/cuda-13.3/targets/x86_64-linux/lib:/usr/lib/wsl/lib"
if pgrep -x llama-server; then exit 2; fi
python3 -u "$r/benchmark.py" 262144 "$r/matched-baseline-resume" --frozen-prompt "$r/frozen-prompt.json" --reference "$ref" --reps 1
python3 - "$r" <<'PY'
import json,sys,statistics
from pathlib import Path
p=Path(sys.argv[1]); old=p/"matched-baseline/results.json"; new=p/"matched-baseline-resume/results.json"
base=json.loads(old.read_text()); more=json.loads(new.read_text())
assert len(base["runs"])==4 and len(more["runs"])==2 and base["prompt_sha256"]==more["prompt_sha256"]
for row in more["runs"]:
    row["rep"]=3
    row["server_restart_for_rep"]=True
base["runs"].extend(more["runs"])
base["peak_device_vram_mib"]=max(base["peak_device_vram_mib"],more["peak_device_vram_mib"])
for mode, metrics in base["median"].items():
    for key in metrics:
        metrics[key]=statistics.median(x[key] for x in base["runs"] if x["mode"]==mode)
tmp=old.with_suffix(".tmp")
tmp.write_text(json.dumps(base,indent=2)+"\n")
tmp.replace(old)
PY
