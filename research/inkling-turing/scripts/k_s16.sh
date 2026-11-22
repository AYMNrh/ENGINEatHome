# IQ loader v2 (sign arithmetic + IQ3_XXS smem grid): correctness, interleaved prefill vs bin-j, ncu
R=~/llm/experiments/inkling-kernels
N=$R/bin-iqA
C=$R/bin-j
M=$R/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf
cd $R
LD_LIBRARY_PATH=$N $N/test-backend-ops -o MUL_MAT_ID -b CUDA0 > results/s16_tbo_mmid.txt 2>&1; echo "MUL_MAT_ID: $(grep -c "OK" results/s16_tbo_mmid.txt) OK, $(grep -c FAIL results/s16_tbo_mmid.txt) FAIL"
LD_LIBRARY_PATH=$N $N/test-backend-ops -o MUL_MAT -b CUDA0 -p "iq3_xxs|iq2_s" > results/s16_tbo_mm.txt 2>&1; echo "MUL_MAT iq: $(grep -c "OK" results/s16_tbo_mm.txt) OK, $(grep -c FAIL results/s16_tbo_mm.txt) FAIL"
pp() { L=$1; B=$2; shift 2
  env LD_LIBRARY_PATH=$B "$@" $B/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 512 -n 0 -d 0 -r 3 -o json 2>/dev/null > $R/results/s16_$L.json
  python3 -c "import json,sys;r=json.loads(open(sys.argv[1]).read());print(sys.argv[2], round(r[0][\"avg_ts\"],1), round(r[0][\"stddev_ts\"],1))" $R/results/s16_$L.json $L
}
pp ctl_J32a $C X=1
pp new_J32a $N X=1
pp ctl_J48a $C GGML_CUDA_INK_T_MMQJ=48
pp new_J48a $N GGML_CUDA_INK_T_MMQJ=48
pp ctl_J32b $C X=1
pp new_J32b $N X=1
pp new_J48b $N GGML_CUDA_INK_T_MMQJ=48
pp ctl_J48b $C GGML_CUDA_INK_T_MMQJ=48
LONG_CHECK=1 python3 k_decode_check.py run $N results/s16_long_new.json
python3 k_decode_check.py cmp results/s15_long_ctl.json results/s16_long_new.json
source <(sed -n "1,/^run() /p" $R/k_ncu.sh | sed "\$d")
for T in 18 22; do
  sudo -n env LD_LIBRARY_PATH=$N GGML_CUDA_DISABLE_GRAPHS=1 ncu --kernel-name-base demangled -k "regex:mul_mat_q<\(ggml_type\)$T, \(int\)32" -s 4 -c 2 --metrics $MET --csv --page raw $N/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 512 -n 0 -d 0 -r 1 -o md > $R/prof/ncu_iqA_$T.csv 2>/dev/null
  python3 $R/k_ncusum.py "$R/prof/ncu_iqA_$T.csv" | cut -c1-900
done
