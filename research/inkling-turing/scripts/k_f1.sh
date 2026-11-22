R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
B=$R/bin-base
M=$R/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf
TAG=${1:-f1}
echo "=== graph dump: fused QKV_POST runs per decode graph"
GGML_CUDA_DUMP_GRAPH=$R/prof/graph_$TAG.txt LD_LIBRARY_PATH=$N $N/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 0 -n 4 -r 1 -o md > /dev/null 2>&1
echo "kernel nodes (K+FUSE): $(grep -cE '^ *[0-9]+ (K|FUSE) ' $R/prof/graph_$TAG.txt)   (base: $(grep -cE '^ *[0-9]+ (K|FUSE) ' $R/prof/graph_dump.txt))"
echo "=== backend tests"
for o in INKLING2_CLIP INKLING2_RELBIAS SET_ROWS RMS_NORM SUM_ROWS PAD MUL_MAT; do
  r=$(LD_LIBRARY_PATH=$N $N/test-backend-ops test -b CUDA0 -o $o 2>&1 | grep -E "tests passed" | tail -1); echo "$o: $r"; done
echo "=== decode equivalence"
[ -f $R/results/decode_ref.json ] || python3 $R/k_decode_check.py run $B $R/results/decode_ref.json
python3 $R/k_decode_check.py run $N $R/results/decode_$TAG.json
python3 $R/k_decode_check.py cmp $R/results/decode_ref.json $R/results/decode_$TAG.json
echo "=== screen 0K"
bash $R/k_bench.sh $N $TAG 0
