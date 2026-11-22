cd ~/llm/experiments/strata-ab
T0=$(date +%s); el() { echo $(( $(date +%s) - T0 )); }
PY=./venv/bin/python
echo "M mem probes $(el)"
timeout 400 python3 mem_probe.py 262144:256:q8_0 262144:128:q8_0 262144:64:q8_0 262144:256:q4_0 262144:128:q4_0 131072:512:q8_0 65536:512:q8_0
echo "D div hunt off $(el)"; timeout 240 $PY div_hunt.py off off1 12 320 160
echo "D div hunt on $(el)"; timeout 240 $PY div_hunt.py dspark on1 12 320 160
echo "D div hunt on again $(el)"; timeout 240 $PY div_hunt.py dspark on2 12 320 160
echo "K kv q8 $(el)"; timeout 200 $PY kv_acc.py q8_0
echo "K kv q4 $(el)"; timeout 200 $PY kv_acc.py q4_0
echo "end $(el)"
