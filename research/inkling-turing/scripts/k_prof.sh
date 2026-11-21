# usage: k_prof.sh <bindir> <label> <pp|tg> <depth>
BIN=$1; LABEL=$2; MODE=$3; DEPTH=${4:-0}
R=~/llm/experiments/inkling-kernels
M=${MODEL:-$HOME/llm/experiments/inkling-kernels/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf}
mkdir -p $R/prof
export LD_LIBRARY_PATH=$BIN
if [ "$MODE" = pp ]; then ARGS="-p 512 -n 0"; else ARGS="-p 0 -n 128"; fi
nsys profile --cuda-graph-trace=node -t cuda -o $R/prof/$LABEL --force-overwrite true --stats=false \
  $BIN/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 $ARGS -d $DEPTH -r 1 -o md > $R/prof/$LABEL.bench.txt 2>&1
[ -f $R/prof/$LABEL.nsys-rep ] || /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i $R/prof/$LABEL.qdstrm -o $R/prof/$LABEL.nsys-rep > /dev/null 2>&1
nsys stats -r cuda_gpu_kern_sum --format csv $R/prof/$LABEL.nsys-rep > $R/prof/$LABEL.kern.csv 2>/dev/null
grep -a '^|' $R/prof/$LABEL.bench.txt | tail -1
