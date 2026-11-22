cd ~/llm/experiments/strata-ab
T0=$(date +%s); el() { echo $(( $(date +%s) - T0 )); }
PY=./venv/bin/python
echo "R1 strata dspark $(el)"; timeout 280 $PY ab.py strata --draft dspark --sizes 8192,32768 --ctx 40960 --tag on_32k --timeout 200
echo "R2 strata off $(el)"; timeout 280 $PY ab.py strata --draft off --sizes 8192,32768 --ctx 40960 --tag off_32k --timeout 200
echo "end $(el)"
