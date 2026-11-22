R=~/llm/experiments/inkling-kernels
B=$R/bin-base
echo "=== acceptance reference: fork baseline, 3 reps, 0K and 8K"
REPS=3 bash $R/k_bench.sh $B base3 0,8192
echo "=== fork baseline, 1 rep, 32K (reference for the b12568 comparison)"
bash $R/k_bench.sh $B base_32k 32768
echo "=== final build default path (must equal baseline), 1 rep"
bash $R/k_bench.sh $R/llama.cpp/build/bin final_default 0,8192
