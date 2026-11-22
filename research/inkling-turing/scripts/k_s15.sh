# MMQ 2-blocks/SM tile (I=64, 128 thr): correctness + interleaved prefill vs bin-j
R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
C=$R/bin-j
M=$R/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf
cd $R
LD_LIBRARY_PATH=$N $N/test-backend-ops -o MUL_MAT_ID -b CUDA0 > results/s15_tbo_mmid.txt 2>&1; echo "MUL_MAT_ID: $(grep -c "OK" results/s15_tbo_mmid.txt) OK, $(grep -c FAIL results/s15_tbo_mmid.txt) FAIL; $(tail -2 results/s15_tbo_mmid.txt | tr "\n" " ")"
pp() { # label bindir env...
  L=$1; B=$2; shift 2
  env LD_LIBRARY_PATH=$B "$@" $B/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 512 -n 0 -d 0 -r 3 -o json 2>/dev/null > $R/results/s15_$L.json
  python3 -c "import json,sys;r=json.loads(open(sys.argv[1]).read());print(sys.argv[2], round(r[0][\"avg_ts\"],1), round(r[0][\"stddev_ts\"],1))" $R/results/s15_$L.json $L
}
pp ctl_J32a $C X=1
pp new_J32a $N X=1
pp ctl_J48a $C GGML_CUDA_INK_T_MMQJ=48
pp new_J48a $N GGML_CUDA_INK_T_MMQJ=48
pp new_J64a $N GGML_CUDA_INK_T_MMQJ=64
pp new_J40a $N GGML_CUDA_INK_T_MMQJ=40
pp ctl_J32b $C X=1
pp new_J32b $N X=1
pp new_J48b $N GGML_CUDA_INK_T_MMQJ=48
LONG_CHECK=1 python3 k_decode_check.py run $C results/s15_long_ctl.json
LONG_CHECK=1 python3 k_decode_check.py run $N results/s15_long_new.json
python3 k_decode_check.py cmp results/s15_long_ctl.json results/s15_long_new.json
