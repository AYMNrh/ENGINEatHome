# prefill MoE tile width screen: same binary, env override of J (interleaved)
R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
M=$R/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf
pp() { # label env...
  L=$1; shift
  env LD_LIBRARY_PATH=$N "$@" $N/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 512 -n 0 -d 0 -r 3 -o json 2>/dev/null > $R/results/s14_$L.json
  python3 -c "import json,sys;r=json.loads(open(sys.argv[1]).read());print(sys.argv[2], round(r[0][\"avg_ts\"],1), round(r[0][\"stddev_ts\"],1))" $R/results/s14_$L.json $L
}
pp J32a X=1
pp J64a GGML_CUDA_INK_T_MMQJ=64
pp J48a GGML_CUDA_INK_T_MMQJ=48
pp J40a GGML_CUDA_INK_T_MMQJ=40
pp J32b X=1
pp J64b GGML_CUDA_INK_T_MMQJ=64
pp J128a GGML_CUDA_INK_T_MMQJ=128
pp J24a GGML_CUDA_INK_T_MMQJ=24
pp J32c X=1
