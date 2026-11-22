R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
for mode in 0 2; do
  for tol in 5e-4 5e-5 1e-5 5e-6 1e-6; do
    r=$(GGML_CUDA_INK_T_FAGQA=$mode INK_FA_NMSE=$tol LD_LIBRARY_PATH=$N $N/test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT -p "ink=1" 2>&1 | grep -E "tests passed" | tail -1)
    echo "mode $mode tol $tol: $r"
  done
done
