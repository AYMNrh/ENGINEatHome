R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
M=$R/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf
TAG=${1:-f6}
GGML_CUDA_DUMP_GRAPH=$R/prof/graph_$TAG.txt LD_LIBRARY_PATH=$N $N/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 0 -n 4 -r 1 -o md > /dev/null 2>&1
echo "kernel nodes (K+FUSE): $(grep -cE '^ *[0-9]+ (K|FUSE) ' $R/prof/graph_$TAG.txt)"
for o in INKLING2_CTRL; do r=$(LD_LIBRARY_PATH=$N $N/test-backend-ops test -b CUDA0 -o $o 2>&1 | grep -E "tests passed" | tail -1); echo "$o: $r"; done
echo "=== decode equivalence (bit-exact default)"
python3 $R/k_decode_check.py run $N $R/results/decode_$TAG.json
python3 $R/k_decode_check.py cmp $R/results/decode_ref.json $R/results/decode_$TAG.json
echo "=== screen 0K + 8K, interleaved: control (fusion off) / candidate / control"
GGML_CUDA_INK_T_DISABLE=16 bash $R/k_bench.sh $N ${TAG}_c1 0,8192
bash $R/k_bench.sh $N ${TAG} 0,8192
GGML_CUDA_INK_T_DISABLE=16 bash $R/k_bench.sh $N ${TAG}_c2 0,8192
