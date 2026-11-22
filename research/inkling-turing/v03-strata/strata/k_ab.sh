cd ~/llm/experiments/strata-ab
T0=$(date +%s); el() { echo $(( $(date +%s) - T0 )); }
PY=./venv/bin/python
echo "P1 strata off $(el)"; timeout 420 $PY ab.py strata --draft off --sizes 8192,32768 --ctx 40960 --tag off_32k
echo "P2 strata dspark $(el)"; timeout 420 $PY ab.py strata --draft dspark --sizes 8192,32768 --ctx 40960 --tag on_32k
echo "P3 ours $(el)"; timeout 420 $PY ab.py ours --sizes 8192,32768 --ctx 40960 --ub 512 --tag 32k
echo "after phase 3: $(el)s"
if [ $(el) -lt 780 ]; then echo "P4 strata dspark 128K $(el)"; timeout 480 $PY ab.py strata --draft dspark --sizes 131072 --ctx 135168 --tag on_128k --timeout 450; fi
if [ $(el) -lt 1200 ]; then echo "P5 ours 128K $(el)"; timeout 480 $PY ab.py ours --sizes 131072 --ctx 139264 --ub 256 --tag 128k --timeout 450; fi
echo "end $(el)"
