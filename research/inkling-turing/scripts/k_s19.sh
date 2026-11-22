# fast exact I2F in MMQ epilogues (bin-i2f) vs bin-iqB: correctness + interleaved prefill + ncu
R=~/llm/experiments/inkling-kernels
N=$R/bin-i2f; C=$R/bin-iqB
M=$R/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf
cd $R
LD_LIBRARY_PATH=$N $N/test-backend-ops -o MUL_MAT_ID -b CUDA0 > results/s19_tbo_mmid.txt 2>&1; echo "MUL_MAT_ID: $(grep -c "OK" results/s19_tbo_mmid.txt) OK, $(grep -c FAIL results/s19_tbo_mmid.txt) FAIL"
LD_LIBRARY_PATH=$N $N/test-backend-ops -o MUL_MAT -b CUDA0 > results/s19_tbo_mm.txt 2>&1; echo "MUL_MAT: $(grep -c "OK" results/s19_tbo_mm.txt) OK, $(grep -c FAIL results/s19_tbo_mm.txt) FAIL"
pp() { L=$1; B_=$2; D=$3; shift 3
  env LD_LIBRARY_PATH=$B_ "$@" $B_/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 512 -n 0 -d $D -r 3 -o json 2>/dev/null > $R/results/s19_$L.json
  python3 -c "import json,sys;r=json.loads(open(sys.argv[1]).read());print(sys.argv[2], round(r[0][\"avg_ts\"],1), round(r[0][\"stddev_ts\"],1))" $R/results/s19_$L.json $L
}
pp ctl_a $C 0 X=1
pp new_a $N 0 X=1
pp ctl_b $C 0 X=1
pp new_b $N 0 X=1
pp ctl_8k $C 8192 X=1
pp new_8k $N 8192 X=1
LONG_CHECK=1 python3 k_decode_check.py run $N results/s19_long_new.json
python3 k_decode_check.py cmp results/s15_long_ctl.json results/s19_long_new.json
source <(sed -n "1,/^run() /p" $R/k_ncu.sh | sed "\$d")
for T in 40 18 22; do
sudo -n env LD_LIBRARY_PATH=$N GGML_CUDA_DISABLE_GRAPHS=1 ncu --kernel-name-base demangled -k "regex:mul_mat_q<\(ggml_type\)$T, \(int\)48" -s 3 -c 1 --metrics gpu__time_duration.sum,smsp__inst_executed.sum,sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active --csv --page raw $N/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 512 -n 0 -d 0 -r 1 -o md 2>/dev/null | tail -1 | awk -F\",\" -v t=$T "{print \"t\" t \" ncu: \" \$(NF-2) \" ns, inst \" \$(NF-1) \", imma% \" \$NF}"
done
