R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
echo "=== long-context decode equivalence (~7K-token prompts): default vs GQA mode 2"
LONG_CHECK=1 python3 $R/k_decode_check.py run $N $R/results/decode_long_ref.json
LONG_CHECK=1 python3 $R/k_decode_check.py run $N $R/results/decode_long_fa4.json GGML_CUDA_INK_T_FAGQA=2
python3 $R/k_decode_check.py cmp $R/results/decode_long_ref.json $R/results/decode_long_fa4.json
echo "=== reference drift: default vs IQ opt-in on the same long prompts (known benign reduction-order change)"
LONG_CHECK=1 python3 $R/k_decode_check.py run $N $R/results/decode_long_iq.json GGML_CUDA_INK_T_MOE1=3
python3 $R/k_decode_check.py cmp $R/results/decode_long_ref.json $R/results/decode_long_iq.json
