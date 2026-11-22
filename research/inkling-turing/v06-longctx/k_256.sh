R=$HOME/llm/experiments/inkling-kernels
B=$R/bin-rp2
M=$R/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf
OUT=results_256
cd $R/pilot; rm -rf $OUT; mkdir -p $OUT
T00=$(date +%s)
echo "bin 7d3e5b1 (bin-rp2) ctx 262144 ub 256 q8_0/q8_0 np 1; hard stop 1500 s"
# VRAM watchdog: log every 2 s, stop the evaluation safely if the card gets within 230 MiB of full
( while sleep 2; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); echo "$(date +%s) $u" >> $OUT/vram.log
    if [ "$u" -gt 22300 ]; then echo "OOM_GUARD at $u MiB" >> $OUT/guard.txt; pkill -f pilot_ctx.py; fi; done ) &
WD=$!
T0=$(date +%s.%N)
LD_LIBRARY_PATH=$B $B/llama-server -m $M -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 262144 -ub 256 -np 1 --jinja --port 18092 --no-warmup -lv 4 > $OUT/server_ub256.log 2>&1 &
SP=$!
OK=0
for i in $(seq 1 240); do curl -sf localhost:18092/health >/dev/null && { OK=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
if [ $OK = 1 ]; then
  echo ub=256 load_s $(python3 -c "print(round($(date +%s.%N)-$T0,1))") vram_loaded_mib $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  LEFT=$(( 1500 - ($(date +%s) - T00) ))
  timeout $LEFT python3 -u pilot_ctx.py --port 18092 --sizes 244000,128000 --ctx 262144 --vary --seed 4242 --max-tokens 4000 --first-timeout 244000=1100,128000=420 --turn-timeout 90 --out $OUT
  echo "pilot rc=$?"
else
  echo "ub=256 failed to load (server exited or timeout)"
fi
echo "vram peak (watchdog) $(sort -k2 -n $OUT/vram.log | tail -1)"
{ echo "re-plan lines: $(grep -c sched_graph_realloc $OUT/server_ub256.log)"; grep -E "sched_reserve|compute buffer|KV buffer|graph: nodes" $OUT/server_ub256.log | head -12; echo "errors:"; grep -iE "error|abort|illegal|out of memory|cudaMalloc failed|failed to allocate" $OUT/server_ub256.log | head -5; echo "cuda graph lines: $(grep -ciE 'cuda graph|graph_update|CUDA_GRAPH' $OUT/server_ub256.log)"; } > $OUT/server_summary.txt 2>&1
kill $SP 2>/dev/null; wait $SP 2>/dev/null
# ubatch comparison: can ub 512 even load at 262144? (load probe only, no evaluation)
if [ $(( $(date +%s) - T00 )) -lt 1350 ]; then
  sleep 3
  LD_LIBRARY_PATH=$B $B/llama-server -m $M -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 262144 -ub 512 -np 1 --jinja --port 18093 --no-warmup > $OUT/server_ub512_probe.log 2>&1 &
  P2=$!; OK2=0
  for i in $(seq 1 120); do curl -sf localhost:18093/health >/dev/null && { OK2=1; break; }; kill -0 $P2 2>/dev/null || break; sleep 1; done
  echo "ub512 probe: loaded=$OK2 vram_mib $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
  grep -iE "out of memory|failed to allocate|cudaMalloc" $OUT/server_ub512_probe.log | head -2
  kill $P2 2>/dev/null; wait $P2 2>/dev/null
fi
kill $WD 2>/dev/null
echo "total_s $(( $(date +%s) - T00 ))"
echo K256_DONE
