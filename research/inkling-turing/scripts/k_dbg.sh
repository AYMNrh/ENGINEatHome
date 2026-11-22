R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
M=$R/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf
GGML_CUDA_INK_T_DEBUG=1 LD_LIBRARY_PATH=$N $N/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 0 -n 2 -r 1 -o md > $R/prof/dbg.txt 2>&1
grep "ink_t" $R/prof/dbg.txt | sed "s/node [0-9]* //;s/(node [0-9]* /(/" | sort | uniq -c | sort -rn | head -12
grep -c "ink_t" $R/prof/dbg.txt
