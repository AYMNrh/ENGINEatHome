R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
bash $R/k_prof.sh $N ctrl2_tg_d0 tg 0 > /dev/null
python3 $R/k_buckets.py $R/prof/ctrl2_tg_d0.kern.csv | grep -E "ctrl|all:"
source <(sed -n '1,/^run() /p' $R/k_ncu.sh | sed '$d')
B=$N
sudo -n env LD_LIBRARY_PATH=$B GGML_CUDA_DISABLE_GRAPHS=1 ncu --kernel-name-base demangled -k "regex:inkling2t_ctrl_f32" -s 6 -c 2 --metrics $MET,gpu__time_duration.sum --csv --page raw $B/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 0 -n 4 -d 0 -r 1 -o md > $R/prof/ncu_ctrl_t.csv 2>/dev/null
python3 $R/k_ncusum.py "$R/prof/ncu_ctrl_t.csv" | cut -c1-700
