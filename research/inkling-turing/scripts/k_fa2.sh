R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
TAG=fa2
echo "=== decode equivalence, mode 2 (expected: not bit-exact; check drift size)"
python3 $R/k_decode_check.py run $N $R/results/decode_$TAG.json GGML_CUDA_INK_T_FAGQA=2
python3 $R/k_decode_check.py cmp $R/results/decode_ref.json $R/results/decode_$TAG.json
r=$(GGML_CUDA_INK_T_FAGQA=2 LD_LIBRARY_PATH=$N $N/test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT 2>&1 | grep -E "tests passed" | tail -1); echo "FLASH_ATTN_EXT (mode 2): $r"
echo "=== screen 0K + 8K: off / mode 2 / off"
bash $R/k_bench.sh $N ${TAG}_c1 0,8192
GGML_CUDA_INK_T_FAGQA=2 bash $R/k_bench.sh $N ${TAG} 0,8192
bash $R/k_bench.sh $N ${TAG}_c2 0,8192
