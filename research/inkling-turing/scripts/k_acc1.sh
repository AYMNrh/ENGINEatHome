R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
TAG=acc1
echo "=== decode equivalence (policy 3 default)"
python3 $R/k_decode_check.py run $N $R/results/decode_$TAG.json
python3 $R/k_decode_check.py cmp $R/results/decode_ref.json $R/results/decode_$TAG.json
echo "=== MUL_MAT_ID tests"
r=$(LD_LIBRARY_PATH=$N $N/test-backend-ops test -b CUDA0 -o MUL_MAT_ID 2>&1 | grep -E "tests passed" | tail -1); echo "MUL_MAT_ID: $r"
echo "=== acceptance: 3 reps 0K + 8K (final build), same-session control (all off) 3 reps"
REPS=3 bash $R/k_bench.sh $N $TAG 0,8192
REPS=3 GGML_CUDA_INK_T_DISABLE=15 GGML_CUDA_INK_T_MOE1=0 bash $R/k_bench.sh $N ${TAG}_off 0,8192
echo "=== 32K, 1 rep (final build)"
bash $R/k_bench.sh $N ${TAG}_32k 32768
