import re
p = '/home/marvin/llm/experiments/inkling-kernels/llama.cpp/ggml/src/ggml-cuda/mmvq.cu'
s = open(p).read()
if 'mmvq-iq.cuh' not in s:
    s = s.replace('#include "vecdotq.cuh"', '#include "vecdotq.cuh"\n#include "mmvq-iq.cuh"', 1)
    anchor = "    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);\n"
    assert s.count(anchor) == 1
    hook = anchor + """
    if constexpr (type == GGML_TYPE_IQ3_XXS) {
        // Turing-oriented warp-per-row decode kernel (mmvq-iq.cuh); GGML_CUDA_MMVQ_IQ_OLD=1 restores the stock kernel
        const bool has_fusion_iq = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                                   fusion.x_scale != nullptr || fusion.gate_scale != nullptr || fusion.shared_up != nullptr;
        if (ncols_dst == 1 && !has_fusion_iq && ncols_x % 256 == 0 && ggml_cuda_mmvq_iq_new_enabled()) {
            mul_mat_vec_q_iq_warp_launch<type>(vx, vy, ids, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x,
                stride_channel_x, stride_channel_y, stride_channel_dst, channel_ratio_fd, sample_ratio_fd,
                stride_sample_x, stride_sample_y, stride_sample_dst, nchannels_dst, nsamples_dst, stream);
            return;
        }
    }
"""
    s = s.replace(anchor, hook, 1)
    open(p, 'w').write(s)
    print('patched')
else:
    print('already patched')
