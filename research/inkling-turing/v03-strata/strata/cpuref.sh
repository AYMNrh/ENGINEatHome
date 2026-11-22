cd ~/llm/experiments/strata-ab
export LD_LIBRARY_PATH=$PWD/bin13
S=$(date +%s)
timeout 1500 systemd-run --user --scope -q -p MemoryMax=21G -p MemorySwapMax=0 ./bin13/strata-inkling --dump-ref results/ref_cpu_$1.bin --prompt-ids $(cat results/case$1.ids) --native pack/Inkling-2-Mini-SQ-15.strata --cpu --threads 10 > results/ref_cpu_$1.log 2>&1
echo done $1 rc=$? secs=$(( $(date +%s) - S )) >> results/cpuref_status.txt
