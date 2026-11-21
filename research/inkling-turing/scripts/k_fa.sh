R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
for d in 8192 32768; do
  bash $R/k_prof.sh $N fa_tg_d$d tg $d > /dev/null
  echo "=== depth $d (decode kernels only; prefill uses other kernels)"
  python3 $R/k_kernsum.py $R/prof/fa_tg_d$d.kern.csv 40 | grep -E "flash_attn|total kernel" | cut -c1-150
done
