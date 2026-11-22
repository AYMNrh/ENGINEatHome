p = '~/llm/experiments/inkling-kernels/llama.cpp/ggml/src/ggml-cuda/mmvq.cu'
s = open(p).read()
old = '''    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel'''
new = '''    // turing-iq3-opt: single-token MUL_MAT_ID through the warp-per-row MoE kernel. The generic ncols_dst == 1 kernel
    // ends every row block with a shared-memory reduction; for the NVFP4 down banks of Inkling 2 Nsight Compute shows
    // 46% of stall cycles at that barrier. GGML_CUDA_INK_T_MOE1: 0 = stock, 1 = down-bank types (default), 2 = all types.
    static const int moe1 = [] {
        const char * env = getenv("GGML_CUDA_INK_T_MOE1");
        return env ? atoi(env) : 1;
    }();
    const bool moe1_type = moe1 == 2 ||
        (moe1 == 1 && (type == GGML_TYPE_NVFP4 || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q5_0));

    if (has_ids && (ncols_dst > 1 || moe1_type)) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, 'w').write(s)
print('patched moe1')
