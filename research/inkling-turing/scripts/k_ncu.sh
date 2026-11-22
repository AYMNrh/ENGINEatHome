# usage: k_ncu.sh <bindir> <tag>   (decode + prefill kernels of interest)
B=$1; TAG=$2
R=~/llm/experiments/inkling-kernels
M=$R/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf
mkdir -p $R/prof
MET="gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__grid_size,launch__block_size,launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block_static,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__warp_issue_stalled_selected_per_warp_active.pct,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,smsp__inst_executed.sum,l1tex__t_sector_hit_rate.pct"
run() { # label mode regex skip
  L=$1; MODE=$2; RE=$3; SK=$4
  if [ "$MODE" = tg ]; then ARGS="-p 0 -n 6"; else ARGS="-p 512 -n 0"; fi
  GGML_CUDA_DISABLE_GRAPHS=1 LD_LIBRARY_PATH=$B sudo -n env LD_LIBRARY_PATH=$B GGML_CUDA_DISABLE_GRAPHS=1 ncu --kernel-name-base demangled -k "regex:$RE" -s $SK -c 2 --metrics $MET --csv --page raw \
    $B/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 $ARGS -d 0 -r 1 -o md > $R/prof/ncu_${TAG}_$L.csv 2> $R/prof/ncu_${TAG}_$L.err
  echo "$L rc=$? lines=$(wc -l < $R/prof/ncu_${TAG}_$L.csv)"
}
run dec_iq3xxs  tg 'mul_mat_vec_q<\(ggml_type\)18, \(int\)1' 30
run dec_iq2s    tg 'mul_mat_vec_q<\(ggml_type\)22, \(int\)1' 30
run dec_nvfp4   tg 'mul_mat_vec_q<\(ggml_type\)40, \(int\)1' 30
run dec_q8_0    tg 'mul_mat_vec_q<\(ggml_type\)8, \(int\)1'  60
run pre_nvfp4   pp 'mul_mat_q<\(ggml_type\)40, \(int\)32' 4
run pre_iq3xxs  pp 'mul_mat_q<\(ggml_type\)18, \(int\)32' 4
run pre_iq2s    pp 'mul_mat_q<\(ggml_type\)22, \(int\)32' 4
