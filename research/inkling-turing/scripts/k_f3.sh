R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
TAG=${1:-f3}
echo "=== backend tests"
for o in MUL_MAT MUL_MAT_ID; do r=$(LD_LIBRARY_PATH=$N $N/test-backend-ops test -b CUDA0 -o $o 2>&1 | grep -E "tests passed" | tail -1); echo "$o: $r"; done
echo "=== decode equivalence"
python3 $R/k_decode_check.py run $N $R/results/decode_$TAG.json
python3 $R/k_decode_check.py cmp $R/results/decode_ref.json $R/results/decode_$TAG.json
echo "=== screen 0K + 8K"
bash $R/k_bench.sh $N $TAG 0,8192
echo "=== control in the same session: all new fusions off (bitmask 15)"
GGML_CUDA_INK_T_DISABLE=15 bash $R/k_bench.sh $N ${TAG}_off 0,8192
echo "=== reprofile decode"
bash $R/k_prof.sh $N ${TAG}_tg_d0 tg 0
python3 $R/k_buckets.py $R/prof/${TAG}_tg_d0.kern.csv | head -24
