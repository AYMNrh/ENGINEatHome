R=~/llm/experiments/inkling-kernels
B=$R/bin-base
echo "=== baseline bench (fork, unmodified)"
bash $R/k_bench.sh $B base 0,8192
echo "=== test-backend-ops INKLING2 (fork correctness on sm_75)"
for o in INKLING2_CLIP INKLING2_CONV INKLING2_SITU INKLING2_MIX INKLING2_CTRL INKLING2_RELBIAS INKLING2_CONV_STATE INKLING2_CONV_DYN; do LD_LIBRARY_PATH=$B $B/test-backend-ops test -b CUDA0 -o $o 2>&1 | grep -E 'tests passed|OK|FAIL|not supported' | tail -2; done
echo "=== nsys tg128 d0"
bash $R/k_prof.sh $B base_tg_d0 tg 0
echo "=== nsys pp512 d0"
bash $R/k_prof.sh $B base_pp_d0 pp 0
