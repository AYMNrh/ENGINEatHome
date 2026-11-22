# NVFP4-only 2 blocks/SM (bin-nv2) vs bin-i2f: interleaved prefill + ncu duration of the NVFP4 kernel
R=~/llm/experiments/inkling-kernels
N=$R/bin-nv2; C=$R/bin-i2f
M=$R/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf
cd $R
pp() { L=$1; B_=$2; D=$3; shift 3
  env LD_LIBRARY_PATH=$B_ "$@" $B_/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 512 -n 0 -d $D -r 3 -o json 2>/dev/null > $R/results/s20_$L.json
  python3 -c "import json,sys;r=json.loads(open(sys.argv[1]).read());print(sys.argv[2], round(r[0][\"avg_ts\"],1), round(r[0][\"stddev_ts\"],1))" $R/results/s20_$L.json $L
}
pp ctl_a $C 0 X=1
pp new_a $N 0 X=1
pp ctl_b $C 0 X=1
pp new_b $N 0 X=1
for B_ in $C $N; do
sudo -n env LD_LIBRARY_PATH=$B_ GGML_CUDA_DISABLE_GRAPHS=1 ncu --kernel-name-base demangled -k "regex:mul_mat_q<\(ggml_type\)40, \(int\)48" -s 3 -c 1 --metrics gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct --csv --page raw $B_/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 512 -n 0 -d 0 -r 1 -o md 2>/dev/null > $R/prof/s20_nv.csv
python3 -c "
import csv,io,sys;t=open(sys.argv[1]).read();t=t[t.find(chr(34)+\"ID\"):];r=list(csv.reader(io.StringIO(t)));h=r[0];d=r[-1]
print(sys.argv[2], {k.split(\".\")[0]:d[h.index(k)] for k in h if k.startswith((\"gpu__\",\"sm__\",\"smsp__\"))})" $R/prof/s20_nv.csv $(basename $B_)
done
