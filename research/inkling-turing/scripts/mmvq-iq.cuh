#pragma once

// Turing-oriented decode (batch 1) mat-vec for the i-quant expert weights of Inkling 2 Mini (RTX 2080 Ti, sm_75).
//
// Why (Nsight Compute on the stock kernel, IQ3_XXS gate/up bank, one launch of 8 experts): 27% of peak DRAM, ~15M warp
// instructions for 42M weights (11 lane-instructions per weight), top stalls lg_throttle (global grid-table loads) and
// barrier (per-row cross-warp reduction). The kernel is bound by instruction issue, not by memory. Two earlier attempts
// (see progress.md: v1 smem grid, v2 smem sign masks) did not help because they kept the same instruction count and added
// a per-block prologue.
//
// v3:
//  * a 16 KB shared table indexed by (grid byte, 4-bit sign nibble) returns 4 already-signed int8 weights in ONE load,
//    so sign application costs nothing in the inner loop (the stock path spends ~4 emulated SIMD-byte ops on it),
//  * persistent blocks: each block builds the table and stages the (shared) q8_1 activation once, then loops over many
//    output rows, so the prologue is amortised,
//  * one warp per row: no cross-warp reduction, no __syncthreads after the prologue.
// The scale/accumulate arithmetic is the stock vec_dot, results differ only in float summation order.

#include "common.cuh"
#include "vecdotq.cuh"

#define MMVQ_IQ_WARPS 8
#define MMVQ_IQ_ROWS  2   // rows in flight per warp

static __device__ __forceinline__ uint32_t mmvq_iq_ld16x2(const void * p, const int byte_off) {
    const uint16_t * q = (const uint16_t *) ((const char *) p + byte_off);
    return (uint32_t) q[0] | ((uint32_t) q[1] << 16);
}

// one 32-weight sub-block of IQ3_XXS against one q8_1 block
static __device__ __forceinline__ float mmvq_iq3_xxs_subblock(
        const block_iq3_xxs * __restrict__ bq3, const int ib32, const int4 ya, const int4 yb, const float y_d,
        const uint32_t * __restrict__ lut_s, const uint8_t * __restrict__ sb_s) {
    const uint32_t q_lo  = mmvq_iq_ld16x2(bq3->qs, 8*ib32 + 0);
    const uint32_t q_hi  = mmvq_iq_ld16x2(bq3->qs, 8*ib32 + 4);
    const uint32_t aux32 = mmvq_iq_ld16x2(bq3->qs, QK_K/4 + 4*ib32);
    const int y[8] = {ya.x, ya.y, ya.z, ya.w, yb.x, yb.y, yb.z, yb.w};

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t qq = l < 2 ? q_lo : q_hi;
        const int sh = ((2*l) & 3) * 8;
        const uint32_t s = sb_s[(aux32 >> (7*l)) & 127];                       // 7 stored sign bits + parity bit
        const uint32_t i0 = (((qq >> sh) & 0xFF) << 4) | (s & 15);
        const uint32_t i1 = (((qq >> (sh + 8)) & 0xFF) << 4) | (s >> 4);
        sumi = ggml_cuda_dp4a((int) lut_s[i0], y[2*l + 0], sumi);
        sumi = ggml_cuda_dp4a((int) lut_s[i1], y[2*l + 1], sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    return __half2float(bq3->d) * y_d * sumi;
}

template <ggml_type type>
static __global__ void __launch_bounds__(MMVQ_IQ_WARPS*32, 2) mul_mat_vec_q_iq_warp(
        const void * __restrict__ vx_ptr, const void * __restrict__ vy_ptr, const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int nchannels_dst, const uint32_t stride_row_x,
        const uint32_t stride_channel_x, const uint32_t stride_channel_dst, const uint3 channel_ratio, const uint3 nrows_fd) {
    static_assert(type == GGML_TYPE_IQ3_XXS, "extend the sub-block helper to add types");
    extern __shared__ __align__(16) int mmvq_iq_smem[];   // [nsub*8 activation ints][nsub scales]
    __shared__ __align__(16) uint32_t lut_s[4096];         // [grid byte][sign nibble] -> 4 signed int8
    __shared__ uint8_t sb_s[128];                          // 7-bit sign field -> 8 sign bits (8th = parity)

    const int tid  = threadIdx.y*32 + threadIdx.x;
    const int lane = threadIdx.x;
    const int nsub = ncols_x / 32;                         // 32-weight sub-blocks per row == q8_1 blocks

    ggml_cuda_pdl_sync();

    // prologue, once per block: activation (shared by all channels/rows) and the signed grid table
    const block_q8_1 * yg = (const block_q8_1 *) vy_ptr;
    int   * ys_q = mmvq_iq_smem;
    float * ys_d = (float *) (mmvq_iq_smem + nsub*8);
    for (int i = tid; i < nsub*8; i += MMVQ_IQ_WARPS*32) {
        ys_q[i] = ((const int *) yg[i >> 3].qs)[i & 7];
    }
    for (int i = tid; i < nsub; i += MMVQ_IQ_WARPS*32) {
        ys_d[i] = __low2float(yg[i].ds);
    }
    for (int i = tid; i < 4096; i += MMVQ_IQ_WARPS*32) {
        const uint32_t g   = iq3xxs_grid[i >> 4];
        const uint32_t nib = i & 15;
        const uint32_t t   = (nib * 0x00204081u) & 0x01010101u;     // bit i of the nibble -> byte i
        const uint32_t m   = t * 0xFFu;                             // 0xFF where the sign is negative
        lut_s[i] = __vsub4(g ^ m, m);                               // (x ^ m) - m: conditional negate per byte
    }
    for (int v = tid; v < 128; v += MMVQ_IQ_WARPS*32) {
        sb_s[v] = (uint8_t) (v | (((uint32_t) __popc(v) & 1) << 7));
    }
    __syncthreads();

    const int total  = nrows_x * nchannels_dst;
    const int stride = gridDim.x*MMVQ_IQ_WARPS;
    for (int r0 = blockIdx.x*MMVQ_IQ_WARPS + threadIdx.y; r0 < total; r0 += MMVQ_IQ_ROWS*stride) {
        const block_iq3_xxs * bx[MMVQ_IQ_ROWS];
        int   dst_off[MMVQ_IQ_ROWS];
        bool  valid[MMVQ_IQ_ROWS];
#pragma unroll
        for (int k = 0; k < MMVQ_IQ_ROWS; ++k) {
            int r = r0 + k*stride;
            valid[k] = r < total;
            r = valid[k] ? r : total - 1;                         // keep the loads convergent, skip the store
            const int ch  = (int) fastdiv((uint32_t) r, nrows_fd);
            const int row = r - ch*nrows_x;
            const uint32_t channel_x = ids ? ids[ch] : fastdiv(ch, channel_ratio);
            bx[k]      = (const block_iq3_xxs *) vx_ptr + (channel_x*stride_channel_x + (uint32_t) row*stride_row_x);
            dst_off[k] = ch*stride_channel_dst + row;
        }

        float acc[MMVQ_IQ_ROWS] = {};
#pragma unroll 3
        for (int s = lane; s < nsub; s += 32) {
            const int4 ya = *(const int4 *) (ys_q + 8*s + 0);
            const int4 yb = *(const int4 *) (ys_q + 8*s + 4);
            const float yd = ys_d[s];
#pragma unroll
            for (int k = 0; k < MMVQ_IQ_ROWS; ++k) {
                acc[k] += mmvq_iq3_xxs_subblock(bx[k] + (s >> 3), s & 7, ya, yb, yd, lut_s, sb_s);
            }
        }
#pragma unroll
        for (int k = 0; k < MMVQ_IQ_ROWS; ++k) {
            const float v = warp_reduce_sum<32>(acc[k]);
            if (lane == 0 && valid[k]) {
                dst[dst_off[k]] = v;
            }
        }
    }
}

static bool ggml_cuda_mmvq_iq_new_enabled() {
    // opt-in: measured neutral on RTX 2080 Ti (see progress.md), so the stock kernel stays the default
    static const bool enabled = getenv("GGML_CUDA_MMVQ_IQ_NEW") != nullptr;
    return enabled;
}

template <ggml_type type>
static void mul_mat_vec_q_iq_warp_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const int ncols_x, const int nrows_x, const uint32_t stride_row_x,
        const uint32_t stride_channel_x, const uint32_t stride_channel_dst, const uint3 channel_ratio,
        const int nchannels_dst, cudaStream_t stream) {
    const int device = ggml_cuda_get_device();
    const int nsm    = ggml_cuda_info().devices[device].nsm;
    const int total  = nrows_x * nchannels_dst;
    const int nblocks = std::min((total + MMVQ_IQ_WARPS - 1)/MMVQ_IQ_WARPS, 2*nsm);
    const uint3 nrows_fd = init_fastdiv_values(nrows_x);
    const dim3 block_nums(nblocks, 1, 1);
    const dim3 block_dims(32, MMVQ_IQ_WARPS, 1);
    const int  nbytes_shared = (ncols_x/32) * (8*4 + 4);
    if (nbytes_shared > 8*1024) {
        GGML_ABORT("mmvq-iq: K too large for the shared activation stage");
    }
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q_iq_warp<type>, lp, vx, vy, ids, dst, ncols_x, nrows_x, nchannels_dst, stride_row_x,
        stride_channel_x, stride_channel_dst, channel_ratio, nrows_fd);
}
