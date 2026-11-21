# GQA decode attention mode 1 (stock split, bit-exact by construction) at depth, bin-i2f, same binary
R=~/llm/experiments/inkling-kernels
N=$R/bin-i2f
M=$R/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf
cd $R
LONG_CHECK=1 python3 k_decode_check.py run $N results/s21_long_off.json
LONG_CHECK=1 python3 k_decode_check.py run $N results/s21_long_m1.json GGML_CUDA_INK_T_FAGQA=1
python3 k_decode_check.py cmp results/s21_long_off.json results/s21_long_m1.json
tg() { L=$1; D=$2; shift 2
  env LD_LIBRARY_PATH=$N "$@" $N/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 0 -n 128 -d $D -r 2 -o json 2>/dev/null > $R/results/s21_$L.json
  python3 -c "import json,sys;r=json.loads(open(sys.argv[1]).read());print(sys.argv[2], round(r[0][\"avg_ts\"],2), round(r[0][\"stddev_ts\"],2))" $R/results/s21_$L.json $L
}
tg off_8k  8192 X=1
tg m1_8k   8192 GGML_CUDA_INK_T_FAGQA=1
tg off_32k 32768 X=1
tg m1_32k  32768 GGML_CUDA_INK_T_FAGQA=1
tg m2_32k  32768 GGML_CUDA_INK_T_FAGQA=2
tg off_32kb 32768 X=1
