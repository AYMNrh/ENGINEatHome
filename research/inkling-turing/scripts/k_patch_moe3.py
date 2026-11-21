p = '/home/marvin/llm/experiments/inkling-kernels/llama.cpp/ggml/src/ggml-cuda/mmvq.cu'
s = open(p).read()
a = s.index('    // turing-iq3-opt: single-token MUL_MAT_ID through the warp-per-row MoE kernel.')
b = s.index('    if (has_ids && (ncols_dst > 1 || moe1_type)) {')
s = s[:a] + '''    // turing-iq3-opt: single-token MUL_MAT_ID through the warp-per-row MoE kernel (no shared-memory reduction).
    // Measured on an RTX 2080 Ti with Inkling 2 Mini: the IQ gate/up banks run 19% faster this way (IQ3_XXS 102 -> 82 us,
    // Nsight Compute), the NVFP4 down banks do not (44.5 -> 47.7 us: without the barrier they are latency bound, a
    // K = 1152 row is ~650 bytes). GGML_CUDA_INK_T_MOE1: 0 = stock, 1 = down-bank types, 2 = all types,
    // 3 = i-quant types (default).
    static const int moe1 = [] {
        const char * env = getenv("GGML_CUDA_INK_T_MOE1");
        return env ? atoi(env) : 3;
    }();
    const bool is_iq = type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S ||
                       type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S;
    const bool moe1_type = moe1 == 2 || (moe1 == 3 && is_iq) ||
        (moe1 == 1 && (type == GGML_TYPE_NVFP4 || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q5_0));

''' + s[b:]
open(p, 'w').write(s)
print('policy 3 default')
