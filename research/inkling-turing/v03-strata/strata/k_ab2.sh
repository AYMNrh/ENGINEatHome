cd ~/llm/experiments/strata-ab
T0=$(date +%s); el() { echo $(( $(date +%s) - T0 )); }
PY=./venv/bin/python
echo "Q1 strata off $(el)"; timeout 300 $PY ab.py strata --draft off --sizes 8192,32768 --ctx 40960 --tag off_32k --timeout 240
echo "Q2 strata dspark $(el)"; timeout 300 $PY ab.py strata --draft dspark --sizes 8192,32768 --ctx 40960 --tag on_32k --timeout 240
echo "Q3 strata dspark 128K fp8 kv $(el)"; timeout 420 $PY ab.py strata --draft dspark --sizes 131072 --ctx 135168 --kv fp8 --tag on_128k_fp8 --timeout 400
echo "end $(el)"
