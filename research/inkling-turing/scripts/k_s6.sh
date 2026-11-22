R=~/llm/experiments/inkling-kernels
B=$R/bin-base
GGML_CUDA_GRAPH_OPT=1 bash $R/k_prof.sh $B gopt_tg_d0 tg 0
