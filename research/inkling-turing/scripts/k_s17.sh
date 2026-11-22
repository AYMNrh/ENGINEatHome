# IQ2_S smem grid + J=1.5x policy (bin-iqB) vs bin-iqA vs bin-j; 0K + 8K prefill; long check; ncu iq2s
R=~/llm/experiments/inkling-kernels
A=$R/bin-iqA; B=$R/bin-iqB; C=$R/bin-j
M=$R/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf
cd $R
LD_LIBRARY_PATH=$B $B/test-backend-ops -o MUL_MAT_ID -b CUDA0 > results/s17_tbo_mmid.txt 2>&1; echo "MUL_MAT_ID: $(grep -c "OK" results/s17_tbo_mmid.txt) OK, $(grep -c FAIL results/s17_tbo_mmid.txt) FAIL"
pp() { L=$1; B_=$2; D=$3; shift 3
  env LD_LIBRARY_PATH=$B_ "$@" $B_/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 512 -n 0 -d $D -r 3 -o json 2>/dev/null > $R/results/s17_$L.json
  python3 -c "import json,sys;r=json.loads(open(sys.argv[1]).read());print(sys.argv[2], round(r[0][\"avg_ts\"],1), round(r[0][\"stddev_ts\"],1))" $R/results/s17_$L.json $L
}
pp A_J48a  $A 0 GGML_CUDA_INK_T_MMQJ=48
pp B_defa  $B 0 X=1
pp A_J32a  $A 0 X=1
pp B_J32a  $B 0 GGML_CUDA_INK_T_MMQJ=32
pp C_defa  $C 0 X=1
pp B_defb  $B 0 X=1
pp A_J48b  $A 0 GGML_CUDA_INK_T_MMQJ=48
pp B_J32b  $B 0 GGML_CUDA_INK_T_MMQJ=32
pp C_d8k   $C 8192 X=1
pp B_d8k   $B 8192 X=1
pp C_d8kb  $C 8192 X=1
LONG_CHECK=1 python3 k_decode_check.py run $B results/s17_long_B.json
python3 k_decode_check.py cmp results/s15_long_ctl.json results/s17_long_B.json
source <(sed -n "1,/^run() /p" $R/k_ncu.sh | sed "\$d")
sudo -n env LD_LIBRARY_PATH=$B GGML_CUDA_DISABLE_GRAPHS=1 ncu --kernel-name-base demangled -k "regex:mul_mat_q<\(ggml_type\)(22|18|40), \(int\)48" -s 6 -c 6 --metrics $MET --csv --page raw $B/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 512 -n 0 -d 0 -r 1 -o md > $R/prof/ncu_iqB.csv 2>/dev/null
python3 - <<"PY"
import csv,io
t=open("~/llm/experiments/inkling-kernels/prof/ncu_iqB.csv",errors="ignore").read(); t=t[t.find("\"ID\""):]
rows=list(csv.reader(io.StringIO(t))); h=rows[0]
for d in rows[2:]:
    g=lambda k: d[h.index(k)]
    print(g("Kernel Name")[:40], "us=",g("gpu__time_duration.sum"), "inst=",g("smsp__inst_executed.sum"), "imma%=",g("sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active"), "lg=",g("smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct"), "mio=",g("smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct"), "lsb=",g("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"), "bank=",g("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"), "regs=",g("launch__registers_per_thread"))
PY
