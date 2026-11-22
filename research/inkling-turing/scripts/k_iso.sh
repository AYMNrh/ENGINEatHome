R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
M=$R/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf
for d in 32 63; do
  echo "=== GGML_CUDA_INK_T_DISABLE=$d"
  GGML_CUDA_INK_T_DISABLE=$d python3 $R/k_decode_check.py run $N $R/results/decode_iso$d.json GGML_CUDA_INK_T_DISABLE=$d > /dev/null
  python3 $R/k_decode_check.py cmp $R/results/decode_ref.json $R/results/decode_iso$d.json | tail -1
done
GGML_CUDA_DUMP_GRAPH=$R/prof/graph_pair.txt GGML_CUDA_INK_T_DEBUG=1 LD_LIBRARY_PATH=$N $N/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 0 -n 2 -r 1 -o md > $R/prof/dbg3.txt 2>&1
grep "ink_t" $R/prof/dbg3.txt | grep -v "line 23[89][0-9] \|line 2[34][0-9][0-9] (node [0-9]* \(ffn_moe\|Rcur\|Qcur\|node_\)" | sed "s/node [0-9]* //;s/(node_[0-9]*)//;s/([^)]*)//" | sort | uniq -c | sort -rn | head -6
echo "--- fused-run lengths now vs before"
for g in graph_f6c graph_pair; do echo $g; grep -E "^ *[0-9]+ (FUSE|K) " $R/prof/$g.txt 2>/dev/null | awk '{print $2, $3}' | sort | uniq -c | sort -rn | head -14; done
