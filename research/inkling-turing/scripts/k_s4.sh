R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
echo "=== correctness: MUL_MAT_ID + MUL_MAT (new binary)"
for o in MUL_MAT_ID MUL_MAT; do
  LD_LIBRARY_PATH=$N $N/test-backend-ops test -b CUDA0 -o $o > $R/results/e2_$o.txt 2>&1
  echo "$o: $(grep -c 'OK' $R/results/e2_$o.txt) OK lines, $(grep -c 'FAIL' $R/results/e2_$o.txt) FAIL lines; $(grep -E 'tests passed|backends passed' $R/results/e2_$o.txt | tail -2 | tr '\n' ' ')"
  grep 'FAIL' $R/results/e2_$o.txt | head -5
done
echo "=== screening bench: new path"
bash $R/k_bench.sh $N e2 0,8192
echo "=== ncu: new IQ3_XXS decode kernel"
source <(sed -n '1,/^run() /p' $R/k_ncu.sh | sed '$d')
run() { L=$1; MODE=$2; RE=$3; SK=$4; if [ "$MODE" = tg ]; then ARGS="-p 0 -n 6"; else ARGS="-p 512 -n 0"; fi
  LD_LIBRARY_PATH=$B sudo -n env LD_LIBRARY_PATH=$B GGML_CUDA_DISABLE_GRAPHS=1 ncu --kernel-name-base demangled -k "regex:$RE" -s $SK -c 2 --metrics $MET --csv --page raw $B/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 $ARGS -d 0 -r 1 -o md > $R/prof/ncu_${TAG}_$L.csv 2> $R/prof/ncu_${TAG}_$L.err; echo "$L rc=$?"; }
B=$N; TAG=e2
run dec_iq3xxs tg 'mul_mat_vec_q_iq_warp' 10
