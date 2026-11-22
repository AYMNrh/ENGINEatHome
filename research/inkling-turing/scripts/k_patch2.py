p = '~/llm/experiments/inkling-kernels/llama.cpp/ggml/src/ggml-cuda/mmvq.cu'
s = open(p).read()
a = s.index("    if constexpr (type == GGML_TYPE_IQ3_XXS) {\n        // Turing-oriented")
b = s.index("    const int device = ggml_cuda_get_device();", a)
hook = """    if constexpr (type == GGML_TYPE_IQ3_XXS) {
        // Turing-oriented persistent warp-per-row decode kernel (mmvq-iq.cuh); GGML_CUDA_MMVQ_IQ_OLD=1 restores the stock kernel.
        // Handles one shared activation vector (nchannels_y == 1, one sample), the MUL_MAT_ID expert case.
        const bool has_fusion_iq = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                                   fusion.x_scale != nullptr || fusion.gate_scale != nullptr || fusion.shared_up != nullptr;
        if (ncols_dst == 1 && !has_fusion_iq && ids != nullptr && nchannels_y == 1 && nsamples_dst == 1 && nsamples_x == 1 &&
                ncols_x % 256 == 0 && ggml_cuda_mmvq_iq_new_enabled()) {
            mul_mat_vec_q_iq_warp_launch<type>(vx, vy, ids, dst, ncols_x, nrows_x, stride_row_x,
                stride_channel_x, stride_channel_dst, channel_ratio_fd, nchannels_dst, stream);
            return;
        }
    }

"""
s = s[:a] + hook + s[b:]
open(p, 'w').write(s)
print('patched v3 hook')
