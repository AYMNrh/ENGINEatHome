R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
TAG=f4
echo "=== backend tests (default policy = down-bank types)"
r=$(LD_LIBRARY_PATH=$N $N/test-backend-ops test -b CUDA0 -o MUL_MAT_ID 2>&1 | grep -E "tests passed" | tail -1); echo "MUL_MAT_ID: $r"
r=$(GGML_CUDA_INK_T_MOE1=2 LD_LIBRARY_PATH=$N $N/test-backend-ops test -b CUDA0 -o MUL_MAT_ID 2>&1 | grep -E "tests passed" | tail -1); echo "MUL_MAT_ID (all types): $r"
echo "=== decode equivalence (default policy)"
python3 $R/k_decode_check.py run $N $R/results/decode_$TAG.json
python3 $R/k_decode_check.py cmp $R/results/decode_ref.json $R/results/decode_$TAG.json
echo "=== screen 0K: policy 0 / 1 / 2"
GGML_CUDA_INK_T_MOE1=0 bash $R/k_bench.sh $N ${TAG}_p0 0
GGML_CUDA_INK_T_MOE1=1 bash $R/k_bench.sh $N ${TAG}_p1 0
GGML_CUDA_INK_T_MOE1=2 bash $R/k_bench.sh $N ${TAG}_p2 0
