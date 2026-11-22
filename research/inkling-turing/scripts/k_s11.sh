R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
echo "=== hardware microbench (175 W cap as in production)"
nvidia-smi --query-gpu=power.limit,clocks.max.sm,clocks.max.mem --format=csv,noheader
$R/hwbench | tee $R/results/hwbench.txt
echo "=== decode check: default must be bit-exact"
python3 $R/k_decode_check.py run $N $R/results/decode_s11.json
python3 $R/k_decode_check.py cmp $R/results/decode_ref.json $R/results/decode_s11.json
echo "=== screen 0K: default vs IQ opt-in (interleaved C-T-C)"
bash $R/k_bench.sh $N s11_c1 0
GGML_CUDA_INK_T_MOE1=3 bash $R/k_bench.sh $N s11_iq 0
bash $R/k_bench.sh $N s11_c2 0
