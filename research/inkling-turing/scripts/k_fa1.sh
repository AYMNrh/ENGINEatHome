R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
TAG=${1:-fa1}
echo "=== backend tests"
r=$(LD_LIBRARY_PATH=$N $N/test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT 2>&1 | grep -E "tests passed" | tail -1); echo "FLASH_ATTN_EXT: $r"
echo "=== decode equivalence"
python3 $R/k_decode_check.py run $N $R/results/decode_$TAG.json
python3 $R/k_decode_check.py cmp $R/results/decode_ref.json $R/results/decode_$TAG.json
echo "=== screen 0K + 8K, interleaved control/candidate/control"
GGML_CUDA_INK_T_DISABLE=128 bash $R/k_bench.sh $N ${TAG}_c1 0,8192
bash $R/k_bench.sh $N ${TAG} 0,8192
GGML_CUDA_INK_T_DISABLE=128 bash $R/k_bench.sh $N ${TAG}_c2 0,8192
