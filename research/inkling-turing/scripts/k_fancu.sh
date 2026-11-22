R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
source <(sed -n '1,/^run() /p' $R/k_ncu.sh | sed '$d')
B=$N
# 6th vec launch of the first decoded token = layer 5, full attention over 8K
sudo -n env LD_LIBRARY_PATH=$B GGML_CUDA_DISABLE_GRAPHS=1 ncu --kernel-name-base demangled -k "regex:flash_attn_ext_vec" -s 53 -c 1 --metrics $MET,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sector_hit_rate.pct --csv --page raw $B/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 0 -n 2 -d 8192 -r 1 -o md > $R/prof/ncu_fa8k.csv 2>/dev/null
python3 $R/k_ncusum.py "$R/prof/ncu_fa8k.csv" | cut -c1-760
python3 - <<'PY'
import csv, io
t = open('~/llm/experiments/inkling-kernels/prof/ncu_fa8k.csv', errors='ignore').read(); t = t[t.find('"ID"'):]
r = list(csv.reader(io.StringIO(t))); h, d = r[0], r[-1]
for k in ('lts__t_sectors_srcunit_tex_op_read.sum', 'lts__t_sector_hit_rate.pct', 'launch__grid_size', 'launch__block_size', 'dram__bytes_read.sum', 'gpu__time_duration.sum'):
    print(k, d[h.index(k)] if k in h else 'n/a')
PY
