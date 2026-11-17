// SenseNova mHC CUDA port of the CPU route and merge operations.
static constexpr int mhc_threads = 256;

struct mhc_view {
    const float * data;
    size_t token_stride;
    size_t stream_stride;

    explicit mhc_view(const ggml_tensor * t)
        : data(static_cast<const float *>(t->data)),
          token_stride(t->nb[1]/sizeof(float)), stream_stride(t->nb[2]/sizeof(float)) {}

    __device__ const float * row(int token, int stream) const {
        return data + token*token_stride + stream*stream_stride;
    }
};

static __device__ void mhc_normalize(float & a, float & b) {
    const float m = fmaxf(a, b);
    const float z = m + logf(expf(a-m) + expf(b-m));
    a -= z;
    b -= z;
}

static __global__ void mhc_route_kernel(mhc_view a, mhc_view b, const float * weights,
                                       const float * bias, float * dst, int width) {
    const int lane = threadIdx.x;
    const float * va = a.row(blockIdx.x, blockIdx.y);
    const float * vb = b.row(blockIdx.x, blockIdx.y);
    float * out = dst + (blockIdx.x + size_t(blockIdx.y)*gridDim.x)*(width+6);
    float sums[9] = {};
    for (int i = lane; i < width; i += mhc_threads) {
        const float x = va[i];
        const float y = vb[i];
        sums[0] += x*x + y*y;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float * w = weights + k*2*width;
            sums[k+1] += w[i]*x + w[width+i]*y;
        }
    }

    __shared__ float partial[9][mhc_threads/32];
    __shared__ float pre[2];
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        for (int offset = 16; offset > 0; offset /= 2) {
            sums[k] += __shfl_down_sync(0xffffffff, sums[k], offset);
        }
        if (lane % 32 == 0) {
            partial[k][lane/32] = sums[k];
        }
    }
    __syncthreads();

    float m00 = 0.0f, m01 = 0.0f, m10 = 0.0f, m11 = 0.0f;
    if (lane == 0) {
        float totals[9] = {};
        for (int k = 0; k < 9; ++k) {
            for (int warp = 0; warp < mhc_threads/32; ++warp) {
                totals[k] += partial[k][warp];
            }
        }
        const float rms = rsqrtf(totals[0]/(2.0f*width) + 1e-6f);
        float logits[8];
        for (int k = 0; k < 8; ++k) {
            logits[k] = totals[k+1]*rms + bias[k];
        }
        pre[0] = 1.0f/(1.0f + expf(-logits[0]));
        pre[1] = 1.0f/(1.0f + expf(-logits[1]));
        out[width] = 2.0f/(1.0f + expf(-logits[2]));
        out[width+1] = 2.0f/(1.0f + expf(-logits[3]));
        m00 = logits[4];
        m01 = logits[5];
        m10 = logits[6];
        m11 = logits[7];
    }
    // Lane 0 and lane 1 normalize independent rows, then independent columns.
    // Preserve the CPU operation order within each FP32 pair.
    if (lane < 2) {
        constexpr unsigned pair_mask = 3;
        m00 = __shfl_sync(pair_mask, m00, 0);
        m01 = __shfl_sync(pair_mask, m01, 0);
        m10 = __shfl_sync(pair_mask, m10, 0);
        m11 = __shfl_sync(pair_mask, m11, 0);
        float left = lane == 0 ? m00 : m10;
        float right = lane == 0 ? m01 : m11;
        for (int i = 0; i < 20; ++i) {
            mhc_normalize(left, right);
            const float other_left = __shfl_xor_sync(pair_mask, left, 1);
            const float other_right = __shfl_xor_sync(pair_mask, right, 1);
            float top = lane == 0 ? left : other_right;
            float bottom = lane == 0 ? other_left : right;
            mhc_normalize(top, bottom);
            const float other_top = __shfl_xor_sync(pair_mask, top, 1);
            const float other_bottom = __shfl_xor_sync(pair_mask, bottom, 1);
            left = lane == 0 ? top : other_bottom;
            right = lane == 0 ? other_top : bottom;
        }
        out[width+2+2*lane] = expf(left);
        out[width+3+2*lane] = expf(right);
    }
    __syncthreads();

    for (int i = lane; i < width; i += mhc_threads) {
        out[i] = pre[0]*va[i] + pre[1]*vb[i];
    }
}

static __global__ void mhc_merge_kernel(mhc_view a, mhc_view b, mhc_view x, mhc_view packed,
                                        float * dst, int width, int tokens) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= width) {
        return;
    }
    const int t = blockIdx.y;
    const int s = blockIdx.z;
    const float va = a.row(t, s)[i];
    const float vb = b.row(t, s)[i];
    const float vx = x.row(t, s)[i];
    const float * gates = packed.row(t, s) + width;
    float * out = dst + (t + size_t(s)*tokens)*2*width;
    out[i] = gates[2]*va + gates[3]*vb + gates[0]*vx;
    out[width+i] = gates[4]*va + gates[5]*vb + gates[1]*vx;
}

template <bool route>
static void ggml_cuda_mhc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const int width = a->ne[0];
    const int tokens = a->ne[1];
    const int streams = a->ne[2];
    const mhc_view left(a);
    const mhc_view right(dst->src[1]);
    if constexpr (route) {
        mhc_route_kernel<<<dim3(tokens, streams), mhc_threads, 0, ctx.stream()>>>(
            left, right, static_cast<const float *>(dst->src[2]->data),
            static_cast<const float *>(dst->src[3]->data), static_cast<float *>(dst->data), width);
    } else {
        mhc_merge_kernel<<<dim3((width + mhc_threads-1)/mhc_threads, tokens, streams),
                           mhc_threads, 0, ctx.stream()>>>(
            left, right, mhc_view(dst->src[2]), mhc_view(dst->src[3]),
            static_cast<float *>(dst->data), width, tokens);
    }
}
