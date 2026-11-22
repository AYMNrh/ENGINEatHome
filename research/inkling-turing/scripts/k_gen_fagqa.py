# Generate fattn-vec-gqa.cu from the stock fattn-vec.cuh kernel with asserted, targeted replacements (turing-iq3-opt).
# The GQA kernel processes the gqa_ratio query heads of one KV head as the kernel's "columns": K/V are loaded (and V
# dequantized) once per block instead of once per query head. Per-head arithmetic and the KV split are those of the
# stock single-column launch, so the results are the same bits.
D = '~/llm/experiments/inkling-kernels/llama.cpp/ggml/src/ggml-cuda/'
src = open(D + 'fattn-vec.cuh').read()
a = src.index('template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size')
b = src.index('#ifdef __clang__\n#pragma clang diagnostic pop')
k = src[a:b]

def rep(old, new, n=1):
    global k
    c = k.count(old)
    assert c == n, (c, n, old[:90])
    k = k.replace(old, new)

rep('static __global__ void flash_attn_ext_vec(', 'static __global__ void flash_attn_ext_vec_gqa(')
rep('''    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half * maskh  = mask ? (const half  *) (mask + nb33*(sequence % ne33) + nb32*(head % ne32) + nb31*ic0) : nullptr;

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);
''', '''    // GQA: one token; column j of this block is query head (head + j) of KV head kv_head
    const int ic0 = 0;
    const int sequence = blockIdx.z / ne12;
    const int kv_head  = blockIdx.z - sequence*ne12;
    const int head     = kv_head*ncols;
    Q += nb03*sequence + nb02*head;
    K += nb13*sequence + nb12*kv_head;
    V += nb23*sequence + nb22*kv_head;

    const half * maskh  = mask ? (const half  *) (mask + nb33*(sequence % ne33)) : nullptr;
    int mask_col[ncols];  // per-head mask rows (Inkling: per-head relative-position bias)
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        mask_col[j] = int((nb32*((head + j) % ne32))/int64_t(sizeof(half)));
    }

    const float slope = 1.0f; // max_bias == 0 (checked by the dispatcher)
    GGML_UNUSED_VARS(max_bias, m0, m1, n_head_log2);
''')
# Q columns are heads, not tokens
k = k.replace('(Q + j*nb01)', '(Q + j*nb02)')
assert k.count('(Q + j*nb02)') == 4, k.count('(Q + j*nb02)')
# every column is a valid query (one token)
for old, new in (('ic0 + j >= int(ne01.z)', 'false'), ('ic0 + j < int(ne01.z)', 'true'),
                 ('ic0 + j_VKQ >= int(ne01.z)', 'false'), ('ic0 + tid < int(ne01.z)', 'true')):
    assert old in k, old
    k = k.replace(old, new)
# per-head mask rows
rep('sum += slope*__half2float(maskh[j*ne11 + i_KQ]);', 'sum += slope*__half2float(maskh[mask_col[j] + i_KQ]);')
# sinks: per head, applied by warp 0 for every column (the stock single-column launch applies it in warp 0)
rep('''    if (sinks && blockIdx.y == 0) {
        const float sink = ((const float *) sinks)[head];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);''', '''    if (sinks && blockIdx.y == 0 && threadIdx.y == 0) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float sink = ((const float *) sinks)[head + j];

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);''')
# outputs: row (token 0, head + j)
rep('dst[(((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;',
    'dst[(((sequence*int(ne01.z) + ic0)*ne02 + head + j_VKQ)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;')
rep('dst_meta[((sequence*int(ne01.z) + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);',
    'dst_meta[((sequence*int(ne01.z) + ic0)*ne02 + head + tid)*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);')

host = r'''
// ---------------------------------------------------------------------------------------------------------------------
// host: replicate launch_fattn's choice of parallel_blocks for the stock single-column vec launch (same KV split, same
// combine order), then launch the GQA kernel with one block per KV head and the stock combine.
// GGML_CUDA_INK_T_FAGQA: 0 = off (default), 1 = exact (stock KV split: same bits, but 3x fewer blocks - measured slower),
// 2 = KV split retuned for one block per KV head (as many blocks as the stock launch; different combine order, not bit-exact)
static int ggml_cuda_fattn_vec_gqa_mode() {
    static const int m = [] {
        const char * env = getenv("GGML_CUDA_INK_T_FAGQA");
        return env ? atoi(env) : 0;
    }();
    return m;
}

bool ggml_cuda_flash_attn_ext_vec_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * KQV) {
    const ggml_tensor * Q     = KQV->src[0];
    const ggml_tensor * K     = KQV->src[1];
    const ggml_tensor * V     = KQV->src[2];
    const ggml_tensor * mask  = KQV->src[3];
    const ggml_tensor * sinks = KQV->src[4];
    const int mode = ggml_cuda_fattn_vec_gqa_mode();
    // short KV ranges cannot be split finely enough to keep the block count: stay on the stock kernel there (measured)
    if (mode == 0 || KQV->src[5] != nullptr || KQV->src[1]->ne[1] < 4096) {
        return false;
    }
    float scale, max_bias, logit_softcap;
    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
    const int gqa_ratio = K->ne[2] > 0 ? (int) (Q->ne[2] / K->ne[2]) : 0;
    if (Q->ne[1] != 1 || Q->ne[0] != 128 || V->ne[0] != 128 || K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q8_0 ||
            gqa_ratio != 3 || Q->ne[2] != gqa_ratio*K->ne[2] || max_bias != 0.0f || logit_softcap != 0.0f || !mask ||
            mask->type != GGML_TYPE_F16 || (sinks && sinks->type != GGML_TYPE_F32) || Q->type != GGML_TYPE_F32 ||
            KQV->type != GGML_TYPE_F32 || Q->nb[0] != sizeof(float) || K->nb[0] != ggml_element_size(K) || V->nb[0] != ggml_element_size(V)) {
        return false;
    }
    constexpr int Dc = 128, NH = 3;
    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;

    // parallel_blocks exactly as launch_fattn computes it for flash_attn_ext_vec<128, 1, q8_0, q8_0, false>
    const dim3 block_dim(WARP_SIZE, ggml_cuda_fattn_vec_get_nthreads_host(ggml_cuda_info().devices[id].cc)/WARP_SIZE, 1);
    int max_blocks_per_sm = 1;
    if (mode == 1) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
            flash_attn_ext_vec<Dc, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false>, block_dim.x*block_dim.y, 0));
    } else {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
            flash_attn_ext_vec_gqa<Dc, NH, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false>, block_dim.x*block_dim.y, 0));
    }
    int parallel_blocks = max_blocks_per_sm;
    const int64_t n_kv     = K->ne[1];
    const int ntiles_dst   = (mode == 1 ? gqa_ratio : 1) * K->ne[2] * Q->ne[3];   // mode 2: one tile per KV head
    const int ntiles_KV    = (n_kv + Dc - 1) / Dc;
    parallel_blocks = std::min(parallel_blocks, ntiles_KV);
    const int blocks_per_wave = nsm * max_blocks_per_sm;
    int nwaves_best = 0, efficiency_percent_best = 0;
    for (int pbt = parallel_blocks; pbt <= ntiles_KV; ++pbt) {
        const int nblocks_total = ntiles_dst * pbt;
        const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
        const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);
        if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
            break;
        }
        if (efficiency_percent > efficiency_percent_best) {
            nwaves_best = nwaves;
            efficiency_percent_best = efficiency_percent;
            parallel_blocks = pbt;
        }
    }

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);
    if (parallel_blocks > 1) {
        dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
        dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
    }
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(Q->ne[2]))));
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);
    const dim3 blocks_num(1, parallel_blocks, K->ne[2]*Q->ne[3]);
    const ggml_cuda_kernel_launch_params lp(blocks_num, block_dim, 0, stream);
    ggml_cuda_kernel_launch(flash_attn_ext_vec_gqa<Dc, NH, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false>, lp,
        (const char *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
        sinks ? (const char *) sinks->data : nullptr, (const int *) nullptr, (const int *) nullptr,
        parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, 0.0f, 0.0f, n_head_log2, logit_softcap,
        (int32_t) Q->ne[0], ne01, (int32_t) Q->ne[2], (int32_t) Q->ne[3], (int32_t) Q->nb[1], (int32_t) Q->nb[2], (int32_t) Q->nb[3],
        (int32_t) K->ne[0], (int32_t) n_kv, (int32_t) K->ne[2], (int32_t) K->ne[3], (int32_t) K->nb[1], (int32_t) K->nb[2], (int64_t) K->nb[3],
        (int32_t) V->nb[1], (int32_t) V->nb[2], (int64_t) V->nb[3],
        (int32_t) mask->ne[1], (int32_t) mask->ne[2], (int32_t) mask->ne[3], (int32_t) mask->nb[1], (int64_t) mask->nb[2], (int64_t) mask->nb[3]);
    CUDA_CHECK(cudaGetLastError());
    if (parallel_blocks > 1) {
        const dim3 block_dim_combine(Dc, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const ggml_cuda_kernel_launch_params lpc(blocks_num_combine, block_dim_combine, parallel_blocks*sizeof(float2), stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<Dc>, lpc, dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}
'''

out = '''// GENERATED by k_gen_fagqa.py from fattn-vec.cuh (turing-iq3-opt). Do not edit by hand.
// Grouped-query decode attention: the 3 query heads of a KV head share one block, so the q8_0 K/V rows are loaded once
// (and V dequantized once) instead of once per query head; each head keeps its own per-head mask row (Inkling's
// relative-position bias) and sink. Same KV split and combine as the stock single-column launch: same bits.
#include "fattn-vec.cuh"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
''' + k + '''#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__
''' + host
open(D + 'fattn-vec-gqa.cu', 'w').write(out)
open(D + 'fattn-vec-gqa.cuh', 'w').write('#pragma once\n#include "common.cuh"\n\n// turing-iq3-opt: grouped-query q8_0 decode attention with per-head masks; returns false if the case is not covered\nbool ggml_cuda_flash_attn_ext_vec_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * KQV);\n')

# hook into the dispatcher: before the stock vec path
f = open(D + 'fattn.cu').read()
if 'fattn-vec-gqa.cuh' not in f:
    f = f.replace('#include "fattn-vec.cuh"', '#include "fattn-vec.cuh"\n#include "fattn-vec-gqa.cuh"', 1)
    assert 'fattn-vec-gqa.cuh' in f
    old = '''        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;'''
    assert f.count(old) == 1
    f = f.replace(old, '''        case BEST_FATTN_KERNEL_VEC:
            if (!ggml_cuda_flash_attn_ext_vec_gqa(ctx, dst)) {
                ggml_cuda_flash_attn_ext_vec(ctx, dst);
            }
            break;''')
    open(D + 'fattn.cu', 'w').write(f)
print('generated fattn-vec-gqa.cu (%d lines)' % out.count('\n'))
