R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
TAG=fa3
echo "=== long-context decode equivalence (~7K-token prompts): default vs GQA mode 2"
LONG_CHECK=1 python3 $R/k_decode_check.py run $N $R/results/decode_long_ref.json
LONG_CHECK=1 python3 $R/k_decode_check.py run $N $R/results/decode_long_$TAG.json GGML_CUDA_INK_T_FAGQA=2
python3 $R/k_decode_check.py cmp $R/results/decode_long_ref.json $R/results/decode_long_$TAG.json
echo "=== screen 8K: off / mode 2 / off"
bash $R/k_bench.sh $N ${TAG}_c1 8192
GGML_CUDA_INK_T_FAGQA=2 bash $R/k_bench.sh $N ${TAG} 8192
bash $R/k_bench.sh $N ${TAG}_c2 8192
echo "=== 32K: off / mode 2 / off (tg only)"
bash $R/k_bench.sh $N ${TAG}_32c1 32768 -p 0
GGML_CUDA_INK_T_FAGQA=2 bash $R/k_bench.sh $N ${TAG}_32 32768 -p 0
bash $R/k_bench.sh $N ${TAG}_32c2 32768 -p 0
