R=~/llm/experiments/inkling-kernels
N=$R/llama.cpp/build/bin
TAG=f5
echo "=== decode equivalence (policy 2 default)"
python3 $R/k_decode_check.py run $N $R/results/decode_$TAG.json
python3 $R/k_decode_check.py cmp $R/results/decode_ref.json $R/results/decode_$TAG.json
echo "=== screen 0K + 8K"
bash $R/k_bench.sh $N $TAG 0,8192
echo "=== ncu: NVFP4 down bank, stock path vs MoE path; IQ3_XXS gate/up on the MoE path"
source <(sed -n '1,/^run() /p' $R/k_ncu.sh | sed '$d')
B=$N
runx() { L=$1; RE=$2; SK=$3; shift 3
  sudo -n env LD_LIBRARY_PATH=$B GGML_CUDA_DISABLE_GRAPHS=1 "$@" ncu --kernel-name-base demangled -k "regex:$RE" -s $SK -c 2 --metrics $MET --csv --page raw $B/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 0 -n 6 -d 0 -r 1 -o md > $R/prof/ncu_${TAG}_$L.csv 2> $R/prof/ncu_${TAG}_$L.err; echo "$L rc=$?"; }
runx nvfp4_stock 'mul_mat_vec_q<\(ggml_type\)40, \(int\)1' 30 GGML_CUDA_INK_T_MOE1=0
runx nvfp4_moe   'mul_mat_vec_q_moe<\(ggml_type\)40' 30 GGML_CUDA_INK_T_MOE1=2
runx iq3xxs_moe  'mul_mat_vec_q_moe<\(ggml_type\)18' 20 GGML_CUDA_INK_T_MOE1=2
python3 $R/k_ncusum.py "$R/prof/ncu_${TAG}_*.csv" | cut -c1-600
