R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
B=$R/bin-base
TAG=acc2
echo "=== decode equivalence of the default build"
python3 $R/k_decode_check.py run $N $R/results/decode_$TAG.json
python3 $R/k_decode_check.py cmp $R/results/decode_ref.json $R/results/decode_$TAG.json
echo "=== acceptance, 3 reps each, interleaved: pure fork / new (default) / new with all fusions off / new"
REPS=3 bash $R/k_bench.sh $B ${TAG}_fork 0,8192
REPS=3 bash $R/k_bench.sh $N ${TAG}_new_a 0,8192
REPS=3 GGML_CUDA_INK_T_DISABLE=127 bash $R/k_bench.sh $N ${TAG}_off 0,8192
REPS=3 bash $R/k_bench.sh $N ${TAG}_new_b 0,8192
echo "=== opt-in IQ routing, 1 rep"
GGML_CUDA_INK_T_MOE1=3 bash $R/k_bench.sh $N ${TAG}_iq 0,8192
echo "=== 32K, 1 rep: new default"
bash $R/k_bench.sh $N ${TAG}_32k 32768
