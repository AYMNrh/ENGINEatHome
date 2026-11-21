R=~/llm/experiments/inkling-kernels
B=$R/bin-base
echo "=== control (base, 0K)";                       bash $R/k_bench.sh $B p_ctl 0
echo "=== P1 GGML_CUDA_GRAPH_OPT=1";                 GGML_CUDA_GRAPH_OPT=1 bash $R/k_bench.sh $B p_graphopt 0
echo "=== P2 LLAMA_ATTN_ROT_DISABLE=1";              LLAMA_ATTN_ROT_DISABLE=1 bash $R/k_bench.sh $B p_norot 0
echo "=== P3 GGML_CUDA_GRAPH_CHUNK_SIZE=0";          GGML_CUDA_GRAPH_CHUNK_SIZE=0 bash $R/k_bench.sh $B p_chunk0 0
