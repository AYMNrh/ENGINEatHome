set -e
r=/mnt/f/MODELS/SenseNova-Loop-9B/optimization-256k
python3 - "$r" <<'PY'
import json, subprocess, sys, time
from pathlib import Path
out = Path(sys.argv[1])
deadline = time.monotonic() + 1000
while time.monotonic() < deadline:
    try:
        trace = json.loads((out / "nsys-mmq-j64/results.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        trace = {}
    if len(trace.get("runs", [])) == 1 and subprocess.run(
            ["pgrep", "-x", "llama-server"], stdout=subprocess.DEVNULL).returncode == 1:
        break
    time.sleep(4)
else:
    raise SystemExit("MMQ trace never reached a completed request")
PY
bash "$r/resume-mmq.sh"
