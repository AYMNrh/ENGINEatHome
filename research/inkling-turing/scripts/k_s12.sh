R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
bash $R/k_prof.sh $N ctrl_tg_d0 tg 0 > /dev/null
python3 $R/k_buckets.py $R/prof/ctrl_tg_d0.kern.csv | grep -E "ctrl|all:"
