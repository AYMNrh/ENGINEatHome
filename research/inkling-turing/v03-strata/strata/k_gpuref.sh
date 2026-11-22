cd ~/llm/experiments/strata-ab
export LD_LIBRARY_PATH=$PWD/bin13
T0=$(date +%s)
for c in 0 1 3 4 5 7 8 10; do
  [ -f results/case$c.ids ] || continue
  timeout 150 ./bin13/strata-inkling --dump-ref results/ref_gpu_$c.bin --prompt-ids $(cat results/case$c.ids) --native pack/Inkling-2-Mini-SQ-15.strata --max-context 1024 > results/ref_gpu_$c.log 2>&1
  echo gpu case $c rc=$? t=$(( $(date +%s) - T0 ))
done
