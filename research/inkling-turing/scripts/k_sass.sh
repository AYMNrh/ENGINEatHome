cd ~/llm/experiments/inkling-kernels
L=llama.cpp/build/bin/libggml-cuda.so
# list cubins/functions matching, then dump only that function
cuobjdump -sass -fun _Z20mul_mat_vec_q_iq_warpIL9ggml_type18EEvPKvS2_PKiPfiiO5uint3jjjjS5_S5_jjj $L > prof/sass_iq_v2.txt 2>&1
wc -l prof/sass_iq_v2.txt
