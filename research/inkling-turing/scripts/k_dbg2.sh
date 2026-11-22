R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
M=$R/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf
GGML_CUDA_DUMP_GRAPH=$R/prof/graph_ctrl.txt GGML_CUDA_INK_T_DEBUG=1 LD_LIBRARY_PATH=$N $N/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 0 -n 2 -r 1 -o md > $R/prof/dbg2.txt 2>&1
grep "ink_t" $R/prof/dbg2.txt | grep -v "line 2[34][0-9][0-9] (node [0-9]* node_\|ffn_moe\|Rcur\|Qcur" | sed "s/node [0-9]* //;s/(node_[0-9]*)//" | sort | uniq -c | sort -rn | head
echo "--- fused CTRL_T runs by length (nodes) in the decode graph"
awk '/FUSE RMS_NORM/ {n=$1; c=0; f=1; next} f && $2=="+" {c++; next} f {print c; f=0}' $R/prof/graph_ctrl.txt | sort -n | uniq -c | sort -rn | head -8
