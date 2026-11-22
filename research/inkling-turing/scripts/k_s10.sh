R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
M=$R/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf
rm -f $R/prof/graph_dump.txt
GGML_CUDA_DUMP_GRAPH=$R/prof/graph_dump.txt LD_LIBRARY_PATH=$N $N/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 0 -n 4 -r 1 -o md 2>&1 | grep tg
head -2 $R/prof/graph_dump.txt; wc -l $R/prof/graph_dump.txt
