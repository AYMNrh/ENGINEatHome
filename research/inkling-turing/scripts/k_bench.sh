# usage: k_bench.sh <bindir> <label> <depths comma list> [extra llama-bench args]
BIN=$1; LABEL=$2; DEPTHS=$3; shift 3
R=~/llm/experiments/inkling-kernels
M=${MODEL:-$HOME/llm/experiments/inkling-kernels/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf}
mkdir -p $R/results
export LD_LIBRARY_PATH=$BIN
$BIN/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 512 -n 128 -d $DEPTHS -r ${REPS:-1} -o json "$@" > $R/results/$LABEL.json 2> $R/results/$LABEL.err
python3 - "$R/results/$LABEL.json" "$LABEL" <<'PY'
import json,sys
raw=open(sys.argv[1]).read()
rows=json.loads(raw[raw.index('['):raw.rindex(']')+1])
out=[]
for r in rows:
    t='pp512' if r.get('n_prompt') else 'tg128'
    out.append(f"{t}@d{r.get('n_depth',0)}={r['avg_ts']:.2f}")
print(sys.argv[2], ' '.join(out))
open(sys.argv[1].replace('.json','.txt'),'w').write(' '.join(out)+'\n')
PY
nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu --format=csv,noheader
