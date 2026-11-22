R = "llama.cpp/"
NL = chr(10)

def edit(path, old, new, count=1):
    s = open(R + path).read()
    assert s.count(old) == count, (path, old[:80], s.count(old))
    open(R + path, "w").write(s.replace(old, new))

def edit_after(path, anchor_start, old, new):
    """replace `old` at its first occurrence after `anchor_start`"""
    s = open(R + path).read()
    a = s.index(anchor_start)
    b = s.index(old, a)
    open(R + path, "w").write(s[:b] + new + s[b + len(old):])

# ====================================================================================================== API
edit("ggml/include/ggml-inkling2.h", "    GGML_API bool ggml_is_inkling2_relbias_pos(const struct ggml_tensor * t);\n", '''    GGML_API bool ggml_is_inkling2_relbias_pos(const struct ggml_tensor * t);

    // DESCRIPTOR node for flash attention that computes the bias itself: the same inputs as ggml_inkling2_relbias_pos, no real output (a
    // one-element F16 dummy, nothing is computed). ggml_flash_attn_ext_set_relbias() hangs it on a flash attention node whose mask is the
    // shared [n_kv, n_tokens] mask: the kernel then adds f16((tab[d,h,t] - sinks[h,t]) + mask[kv,t]) while it loads each mask tile, so the
    // per-head F16 mask is never materialised. The node keeps the op sequence of prefill and decode graphs identical.
    GGML_API struct ggml_tensor * ggml_inkling2_relbias_desc(
            struct ggml_context * ctx,
            struct ggml_tensor  * tab,
            struct ggml_tensor  * sinks,
            struct ggml_tensor  * cell,
            struct ggml_tensor  * tok,
            struct ggml_tensor  * mask);

    GGML_API bool ggml_is_inkling2_relbias_desc(const struct ggml_tensor * t);

    GGML_API void ggml_flash_attn_ext_set_relbias(struct ggml_tensor * fa, struct ggml_tensor * desc);
''')

edit("ggml/src/ggml.c", "struct ggml_tensor * ggml_custom_inplace(\n", r'''// ggml_inkling2_relbias_desc

static const char ggml_inkling2_relbias_desc_tag = 0;

static void ggml_inkling2_relbias_desc_cpu(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(dst); GGML_UNUSED(ith); GGML_UNUSED(nth); GGML_UNUSED(userdata); // a descriptor: flash attention reads the sources
}

struct ggml_tensor * ggml_inkling2_relbias_desc(
        struct ggml_context * ctx,
        struct ggml_tensor  * tab,
        struct ggml_tensor  * sinks,
        struct ggml_tensor  * cell,
        struct ggml_tensor  * tok,
        struct ggml_tensor  * mask) {
    GGML_ASSERT(tab->type == GGML_TYPE_F32 && tab->nb[0] == sizeof(float) && tab->ne[3] == 1);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32 && sinks->ne[0] == 1 && sinks->ne[1] == tab->ne[1] && sinks->ne[2] == tab->ne[2] && sinks->ne[3] == 1);
    GGML_ASSERT(cell->type == GGML_TYPE_I32 && ggml_is_contiguous(cell) && cell->ne[1] == 1 && cell->ne[2] == 1 && cell->ne[3] == 1);
    GGML_ASSERT(tok->type == GGML_TYPE_I32 && ggml_is_contiguous(tok) && tok->ne[0] == tab->ne[2] && tok->ne[1] == 1 && tok->ne[2] == 1 && tok->ne[3] == 1);
    GGML_ASSERT(mask->type == GGML_TYPE_F16 && mask->ne[0] == cell->ne[0] && mask->ne[1] == tab->ne[2] && mask->ne[2] == 1 && mask->ne[3] == 1);

    struct ggml_tensor * args[5] = { tab, sinks, cell, tok, mask };
    return ggml_custom_4d(ctx, GGML_TYPE_F16, 1, 1, 1, 1, args, 5, ggml_inkling2_relbias_desc_cpu, 1, (void *) &ggml_inkling2_relbias_desc_tag);
}

bool ggml_is_inkling2_relbias_desc(const struct ggml_tensor * t) {
    if (t->op != GGML_OP_CUSTOM) {
        return false;
    }
    struct ggml_custom_op_params p;
    memcpy(&p, t->op_params, sizeof(p));
    return p.userdata == (void *) &ggml_inkling2_relbias_desc_tag;
}

void ggml_flash_attn_ext_set_relbias(struct ggml_tensor * fa, struct ggml_tensor * desc) {
    GGML_ASSERT(fa->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(ggml_is_inkling2_relbias_desc(desc));
    GGML_ASSERT(fa->src[3] == desc->src[4]); // the shared mask
    GGML_ASSERT(fa->src[5] == NULL);         // no kv_rows
    GGML_ASSERT(fa->src[0]->ne[2] == desc->src[0]->ne[1] && fa->src[0]->ne[1] == desc->src[0]->ne[2]);
    fa->src[6] = desc;
}

''' + "struct ggml_tensor * ggml_custom_inplace(\n")

# ====================================================================================================== CPU flash attention
edit("ggml/src/ggml-cpu/ops.cpp", "    const bool use_ref = params->use_ref;\n",
     "    const bool use_ref = params->use_ref || dst->src[6]; // the relative-position bias node (src[6]) is applied in the vec-only reference path\n")
edit("ggml/src/ggml-cpu/ops.cpp",
     "            const float mv = mp ? slope*GGML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;\n",
     '''            float mraw = mp ? GGML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;
            if (mp && dst->src[6]) {
                // relative-position bias: the F16 value of (table[d] - sink) + mask, d from the cell and token positions
                const ggml_tensor * rel = dst->src[6];
                const ggml_tensor * tab = rel->src[0];
                const int64_t extent = tab->ne[0] - 1;
                const int32_t cp = ((const int32_t *) rel->src[2]->data)[ic];
                const int64_t dist = (int64_t) ((const int32_t *) rel->src[3]->data)[iq1] - cp;
                const int64_t d = cp >= 0 && dist >= 0 && dist < extent ? dist : extent;
                const float tv = *(const float *) ((const char *) tab->data + d*tab->nb[0] + iq2*tab->nb[1] + iq1*tab->nb[2]);
                const float sv = *(const float *) ((const char *) rel->src[1]->data + iq2*rel->src[1]->nb[1] + iq1*rel->src[1]->nb[2]);
                const float b = tv - sv;
                mraw = GGML_CPU_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16(b + mraw));
            }
            const float mv = mp ? slope*mraw : 0.0f;
''')

# ====================================================================================================== CUDA common
edit("ggml/src/ggml-cuda/fattn-common.cuh", "typedef void (* fattn_kernel_t)(\n", '''// the relative-position bias computed inside the mma kernel (ggml_flash_attn_ext_set_relbias): all null when not used
struct fattn_rel_params {
    const float   * tab;  // [n_rows, n_head, n_tok] F32
    const float   * sink; // [1, n_head, n_tok] F32
    const int32_t * cell; // [n_kv]: the position of each cell, -1 when empty
    const int32_t * tok;  // [n_tok]: the position of each token
    int64_t tab_s1, tab_s2;   // element strides of the head and the token of tab
    int64_t sink_s1, sink_s2; // and of sink
    int n_rows;
};

// the same for one Q head
struct fattn_rel_head {
    const float   * tab;
    const float   * sink;
    const int32_t * cell;
    const int32_t * tok;
    int64_t tab_s2, sink_s2;
    int n_rows;
};

typedef void (* fattn_kernel_t)(
''')
edit("ggml/src/ggml-cuda/fattn-common.cuh",
     "                            const int32_t nb31, const int64_t nb32, const int64_t nb33);\n",
     "                            const int32_t nb31, const int64_t nb32, const int64_t nb33,\n                            const fattn_rel_params rel);\n")
for f in ("fattn-vec.cuh", "fattn-tile.cuh", "fattn-vec-gqa.cu"):
    edit("ggml/src/ggml-cuda/" + f, "                            const int32_t nb31, const int64_t nb32, const int64_t nb33) {\n",
         "                            const int32_t nb31, const int64_t nb32, const int64_t nb33,\n                            const fattn_rel_params rel) {\n    GGML_UNUSED(rel);\n")
edit("ggml/src/ggml-cuda/fattn-common.cuh",
     "    GGML_ASSERT(!kv_rows || !use_sparse);\n",
     '''    GGML_ASSERT(!kv_rows || !use_sparse);

    // the relative-position bias computed in the kernel (descriptor node hung on the flash attention node)
    fattn_rel_params rel = {};
    if (dst->src[6] && ggml_is_inkling2_relbias_desc(dst->src[6])) {
        const ggml_tensor * rd = dst->src[6];
        GGML_ASSERT(!kv_rows && !use_sparse && ncols2 == 1 && mask && mask->ne[2] == 1);
        rel.tab     = (const float   *) rd->src[0]->data;
        rel.sink    = (const float   *) rd->src[1]->data;
        rel.cell    = (const int32_t *) rd->src[2]->data;
        rel.tok     = (const int32_t *) rd->src[3]->data;
        rel.tab_s1  = rd->src[0]->nb[1]/sizeof(float);
        rel.tab_s2  = rd->src[0]->nb[2]/sizeof(float);
        rel.sink_s1 = rd->src[1]->nb[1]/sizeof(float);
        rel.sink_s2 = rd->src[1]->nb[2]/sizeof(float);
        rel.n_rows  = (int) rd->src[0]->ne[0];
    }
''')
edit("ggml/src/ggml-cuda/fattn-common.cuh", "    const bool scan_mask = !use_sparse && mask && mask->ne[2] == 1 &&", "    const bool scan_mask = !use_sparse && !rel.tab && mask && mask->ne[2] == 1 &&")
edit("ggml/src/ggml-cuda/fattn-common.cuh",
     "        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0\n    );",
     "        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,\n        rel\n    );")
edit("ggml/src/ggml-cuda/fattn-common.cuh", "#include \"common.cuh\"\n", "#include \"common.cuh\"\n#include \"ggml-inkling2.h\"\n")

# ====================================================================================================== CUDA mma kernel
P = "ggml/src/ggml-cuda/fattn-mma-f16.cuh"
edit(P, "template<int ncols1, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check, bool use_sparse>\nstatic __device__ __forceinline__ void flash_attn_ext_f16_load_mask(",
     r'''// relative-position bias while the mask tile is loaded: f16((table[d] - sink) + mask), d from the cell and the token position; the same
// arithmetic as the mask kernel (inkling2_relbias_mask), so the tile holds the same F16 values the dense per-head mask held
static __device__ __forceinline__ half2 fattn_rel_val2(const fattn_rel_head & rel, const half2 m2, const int t, const int kv) {
    const float * tb = rel.tab + t*rel.tab_s2;
    const float   sv = rel.sink[t*rel.sink_s2];
    const int     tp = rel.tok[t];
    const float2  mf = __half22float2(m2);
    int d[2];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        const int     cp   = rel.cell[kv + c];
        const int64_t dist = (int64_t) tp - cp;
        d[c] = cp >= 0 && dist >= 0 && dist < rel.n_rows - 1 ? (int) dist : rel.n_rows - 1;
    }
    return make_half2(__float2half(__fadd_rn(__fsub_rn(tb[d[0]], sv), mf.x)), __float2half(__fadd_rn(__fsub_rn(tb[d[1]], sv), mf.y)));
}

template<int ncols1, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check, bool use_sparse>
static __device__ __forceinline__ void flash_attn_ext_f16_load_mask(''')
edit(P, "        const int stride_mask, const int k_VKQ_0, const int i_sup, const int j0, const uint3 ne01,\n        const int32_t * const __restrict__ indices) {\n    constexpr int warp_size = ggml_cuda_get_physical_warp_size();\n    if constexpr (use_cp_async) {",
     "        const int stride_mask, const int k_VKQ_0, const int i_sup, const int j0, const uint3 ne01,\n        const int32_t * const __restrict__ indices, const fattn_rel_head rel) {\n    constexpr int warp_size = ggml_cuda_get_physical_warp_size();\n    if constexpr (use_cp_async) {")
# oob path
edit(P, "                    tile_mask[j_sram*(nbatch_fa + 8) + i] = i < i_sup ? mask_h[int64_t(j_vram)*stride_mask + k_VKQ_0 + i] : half(0.0f);",
     '''                    if (rel.tab && i < i_sup) {
                        const half2 m2 = make_half2(mask_h[int64_t(j_vram)*stride_mask + k_VKQ_0 + i], half(0.0f));
                        tile_mask[j_sram*(nbatch_fa + 8) + i] = __low2half(fattn_rel_val2(rel, m2, j_vram, k_VKQ_0 + i));
                    } else {
                        tile_mask[j_sram*(nbatch_fa + 8) + i] = i < i_sup ? mask_h[int64_t(j_vram)*stride_mask + k_VKQ_0 + i] : half(0.0f);
                    }''')
edit(P, "            ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + 2*i, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + 2*i);",
     '''            if (rel.tab) {
                half2 m2;
                ggml_cuda_memcpy_1<sizeof(half2)>(&m2, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + 2*i);
                const half2 v2 = fattn_rel_val2(rel, m2, j_vram, k_VKQ_0 + 2*i);
                ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + 2*i, &v2);
            } else {
                ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + 2*i, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + 2*i);
            }''')
edit(P, "                ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + i, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + i);",
     '''                if (rel.tab) {
                    half2 m2;
                    ggml_cuda_memcpy_1<sizeof(half2)>(&m2, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + i);
                    const half2 v2 = fattn_rel_val2(rel, m2, j_vram, k_VKQ_0 + i);
                    ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + i, &v2);
                } else {
                    ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + i, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + i);
                }''')
# call sites of load_mask
edit(P, "(mask_h, tile_mask, stride_mask, k_VKQ_0, k_VKQ_sup, jt*ncols1, ne01, indices);", "(mask_h, tile_mask, stride_mask, k_VKQ_0, k_VKQ_sup, jt*ncols1, ne01, indices, rel);")
edit(P, "(mask_h, tile_mask, stride_mask, kb0_next*nbatch_fa, k_VKQ_sup, jt*ncols1, ne01, nullptr);", "(mask_h, tile_mask, stride_mask, kb0_next*nbatch_fa, k_VKQ_sup, jt*ncols1, ne01, nullptr, rel);")
edit(P, "(mask_h, tile_mask, stride_mask, kb0*nbatch_fa, k_VKQ_sup, jt*ncols1, ne01, nullptr);", "(mask_h, tile_mask, stride_mask, kb0*nbatch_fa, k_VKQ_sup, jt*ncols1, ne01, nullptr, rel);")
# iter signature + GGML_UNUSED
edit(P, "        const int kb0,\n        const int kb0_next,\n        const int k_VKQ_sup) {\n", "        const int kb0,\n        const int kb0_next,\n        const int k_VKQ_sup,\n        const fattn_rel_head rel) {\n")
edit(P, "KQ_max, KQ_rowsum, jt, kb0, kb0_next, k_VKQ_sup);", "KQ_max, KQ_rowsum, jt, kb0, kb0_next, k_VKQ_sup, rel);")
s = open(R + P).read()
n1 = s.count("KQ_max, KQ_rowsum, jt, kb0, kb0, k_VKQ_sup);")
n2 = s.count("KQ_max, KQ_rowsum, jt, kb0, kb0 + 1, k_VKQ_sup);")
assert n1 >= 1 and n2 >= 0, (n1, n2)
s = s.replace("KQ_max, KQ_rowsum, jt, kb0, kb0, k_VKQ_sup);", "KQ_max, KQ_rowsum, jt, kb0, kb0, k_VKQ_sup, rel);")
s = s.replace("KQ_max, KQ_rowsum, jt, kb0, kb0 + 1, k_VKQ_sup);", "KQ_max, KQ_rowsum, jt, kb0, kb0 + 1, k_VKQ_sup, rel);")
open(R + P, "w").write(s)
# process_tile signature
s = open(R + P).read()
a = s.index("static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(")
b = s.index(") {", a)
sig = s[a:b]
assert "kb0_max" in sig, sig[-200:]
s = s[:b] + ",\n        const fattn_rel_head rel" + s[b:]
open(R + P, "w").write(s)
# entry: param + per-head descriptor + calls
edit(P, "                            const int32_t nb31, const int64_t nb32, const int64_t nb33) {\n    ggml_cuda_pdl_sync(); // TODO optimize placement",
     "                            const int32_t nb31, const int64_t nb32, const int64_t nb33,\n                            const fattn_rel_params rel_params) {\n    ggml_cuda_pdl_sync(); // TODO optimize placement")
edit(P, "        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;\n",
     '''        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

        // relative-position bias of this Q head (ncols2 == 1: one head per tile)
        fattn_rel_head rel = {};
        if (rel_params.tab && ncols2 == 1) {
            rel.tab    = rel_params.tab  + zt_Q*rel_params.tab_s1;
            rel.sink   = rel_params.sink + zt_Q*rel_params.sink_s1;
            rel.cell   = rel_params.cell;
            rel.tok    = rel_params.tok;
            rel.tab_s2 = rel_params.tab_s2;
            rel.sink_s2 = rel_params.sink_s2;
            rel.n_rows = rel_params.n_rows;
        }
''')
s = open(R + P).read()
old = "ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_rot, kb0_rot + kb0_stop - kb0_start, kb0_max);"
assert s.count(old) >= 2, s.count(old)
s = s.replace(old, old[:-2] + ", rel);")
open(R + P, "w").write(s)
# case function: rel needs ncols2 == 1
edit(P, "    constexpr int ncols = ncols1 * ncols2;\n\n    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);",
     "    constexpr int ncols = ncols1 * ncols2;\n\n    // the relative-position bias is computed per Q head: one head per tile only\n    GGML_ASSERT(!dst->src[6] || ncols2 == 1);\n\n    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);")

# ====================================================================================================== CUDA dispatcher + descriptor node
edit("ggml/src/ggml-cuda/fattn.cu",
     "void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {\n    ggml_cuda_set_device(ctx.device);\n",
     "void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {\n    ggml_cuda_set_device(ctx.device);\n    GGML_ASSERT(!dst->src[6] || ggml_cuda_fattn_rel_ok(ggml_cuda_get_device(), dst));\n")
edit("ggml/src/ggml-cuda/fattn.cu",
     "bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {\n    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;\n}",
     '''// the relative-position bias in the mma kernel: Turing, head size 128, an odd GQA ratio (one Q head per tile), several query rows, no kv_rows,
// GGML_CUDA_INK_REL_FA=0 turns it off (the scheduler then runs such a node on the CPU: the model graph is built without it with LLAMA_INK_RELBIAS_FA=0)
bool ggml_cuda_fattn_rel_ok(int device, const ggml_tensor * dst) {
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_INK_REL_FA"); return !e || atoi(e) != 0; }();
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const int cc = ggml_cuda_info().devices[device].cc;
    return enabled && dst->src[6] && ggml_is_inkling2_relbias_desc(dst->src[6]) && cc == GGML_CUDA_CC_TURING && Q->ne[0] == 128 && dst->src[2]->ne[0] == 128 &&
        K->ne[2] > 0 && (Q->ne[2] / K->ne[2]) % 2 == 1 && Q->ne[1] > 1 && !dst->src[5] && dst->src[3] && dst->src[3]->ne[2] == 1 &&
        ggml_cuda_get_best_fattn_kernel(device, dst) == BEST_FATTN_KERNEL_MMA_F16;
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    if (dst->src[6] && ggml_is_inkling2_relbias_desc(dst->src[6])) {
        return ggml_cuda_fattn_rel_ok(device, dst);
    }
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}''')
edit("ggml/src/ggml-cuda/fattn.cuh", "bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);\n",
     "bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);\nbool ggml_cuda_fattn_rel_ok(int device, const ggml_tensor * dst);\n")
# the descriptor node in the CUDA backend: nothing to compute
edit("ggml/src/ggml-cuda/ggml-cuda.cu",
     "        case GGML_OP_CUSTOM:\n            if (ggml_is_inkling2_relbias_pos(dst) && ggml_cuda_inkling2_relbias_pos_node_supported(dst)) {",
     "        case GGML_OP_CUSTOM:\n            if (ggml_is_inkling2_relbias_desc(dst)) {\n                break; // a descriptor read by flash attention\n            }\n            if (ggml_is_inkling2_relbias_pos(dst) && ggml_cuda_inkling2_relbias_pos_node_supported(dst)) {")
edit("ggml/src/ggml-cuda/ggml-cuda.cu",
     "        case GGML_OP_CUSTOM:\n            return (ggml_is_inkling2_relbias(op) && ggml_cuda_inkling2_relbias_node_supported(op)) ||",
     "        case GGML_OP_CUSTOM:\n            return ggml_is_inkling2_relbias_desc(op) ||\n                   (ggml_is_inkling2_relbias(op) && ggml_cuda_inkling2_relbias_node_supported(op)) ||")

# ====================================================================================================== llama graph + model
edit("src/llama-graph.h", "bool llama_ink_relbias_node_enabled();\n",
     "bool llama_ink_relbias_node_enabled();\n// LLAMA_INK_RELBIAS_FA=0: the node materialises the per-head mask (global layers, several tokens) instead of flash attention computing the bias\nbool llama_ink_relbias_fa_enabled();\n")
edit("src/llama-graph.cpp", "bool llama_ink_relbias_node_enabled() {\n",
     "bool llama_ink_relbias_fa_enabled() {\n    static const bool enabled = [] { const char * e = getenv(\"LLAMA_INK_RELBIAS_FA\"); return !e || atoi(e) != 0; }();\n    return enabled;\n}\n\nbool llama_ink_relbias_node_enabled() {\n")
# hook in build_attn_mha: a pending descriptor is hung on the flash attention node
edit("src/llama-graph.cpp", "        ggml_flash_attn_ext_add_sinks(cur, sinks);\n        GGML_ASSERT(n_kv_max >= 0 && n_kv_max <= INT32_MAX);",
     "        ggml_flash_attn_ext_add_sinks(cur, sinks);\n        if (rel_fa_desc) {\n            ggml_flash_attn_ext_set_relbias(cur, rel_fa_desc);\n        }\n        GGML_ASSERT(n_kv_max >= 0 && n_kv_max <= INT32_MAX);")
edit("src/llama-graph.h", "struct llm_graph_context {" + chr(10) + "    const llama_model & model;" + chr(10),
     "struct llm_graph_context {" + chr(10) + "    // set by build_attn_relbias while it builds a flash attention node whose bias flash attention computes itself" + chr(10) + "    mutable ggml_tensor * rel_fa_desc = nullptr;" + chr(10) + chr(10) + "    const llama_model & model;" + chr(10))
edit("src/llama-graph.cpp",
     "    ggml_tensor * kq_mask = ggml_inkling2_relbias_pos(ctx0, rel_tab, rel_s, rel_cell, rel_tok, attn_mask.raw,\n            cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32);\n    cb(kq_mask, \"kq_mask_rel_pos\", il);\n\n    return build_attn_iswa_impl(*this, inp_attn, wo, wo_b, wo_s, q_cur, k_cur, v_cur, nullptr, kq_mask, nullptr, sinks, nullptr, kq_scale, il);",
     '''    // global layer, several tokens, flash attention: a descriptor node (no mask is built) and flash attention adds the bias itself; the node
    // keeps the op sequence of the decode graph, where the real node is built
    if (cparams.flash_attn && !is_swa && q_cur->ne[2] > 1 && llama_ink_relbias_fa_enabled()) {
        ggml_tensor * desc = ggml_inkling2_relbias_desc(ctx0, rel_tab, rel_s, rel_cell, rel_tok, attn_mask.raw);
        cb(desc, "kq_mask_rel_desc", il);
        rel_fa_desc = desc;
        ggml_tensor * out = build_attn_iswa_impl(*this, inp_attn, wo, wo_b, wo_s, q_cur, k_cur, v_cur, nullptr, attn_mask.raw, nullptr, sinks, nullptr, kq_scale, il);
        rel_fa_desc = nullptr;
        return out;
    }

    ggml_tensor * kq_mask = ggml_inkling2_relbias_pos(ctx0, rel_tab, rel_s, rel_cell, rel_tok, attn_mask.raw,
            cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32);
    cb(kq_mask, "kq_mask_rel_pos", il);

    return build_attn_iswa_impl(*this, inp_attn, wo, wo_b, wo_s, q_cur, k_cur, v_cur, nullptr, kq_mask, nullptr, sinks, nullptr, kq_scale, il);''')

# ====================================================================================================== test: fused flash attention vs the dense mask
T = "tests/test-backend-ops.cpp"
edit(T, "RELBIAS_NODE, RELBIAS_POS };", "RELBIAS_NODE, RELBIAS_POS, FA_REL };")
edit(T, '"INKLING2_RELBIAS_NODE", "INKLING2_RELBIAS_POS" };', '"INKLING2_RELBIAS_NODE", "INKLING2_RELBIAS_POS", "INKLING2_FA_REL" };')
edit(T, "chain == RELBIAS_NODE || chain == RELBIAS_POS ? 5e-4 : 1e-6;", "chain == RELBIAS_NODE || chain == RELBIAS_POS || chain == FA_REL ? 5e-4 : 1e-6;")
edit(T, "            case CONV_STATE:\n            case CONV_DYN: {\n", '''            case FA_REL: {
                // flash attention computing the bias (descriptor node) against flash attention over the real per-head mask of the same
                // positions: out = rel + 1e4*(rel - dense) is the fused result where the two agree bit for bit. n = 1: q8_0 K and V.
                const int64_t n_head = k;
                const int64_t n_kv   = v;
                const int64_t n_ext  = 64;
                const ggml_type kvt  = n == 1 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
                ggml_tensor * w     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, n_ext);
                ggml_tensor * r     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, n_head, n_tok);
                ggml_tensor * sinks = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_head, n_tok);
                ggml_tensor * cell  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);
                ggml_tensor * tokp  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tok);
                ggml_tensor * mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tok);
                ggml_tensor * q     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_head, n_tok);
                ggml_tensor * kk    = ggml_new_tensor_3d(ctx, kvt, 128, n_kv, 8);
                ggml_tensor * vv    = ggml_new_tensor_3d(ctx, kvt, 128, n_kv, 8);
                ggml_set_name(cell, "cell");
                ggml_set_name(tokp, "tokp");
                ggml_tensor * table = ggml_mul_mat(ctx, ggml_pad(ctx, w, 0, 1, 0, 0), r);
                ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
                ggml_tensor * dense = ggml_inkling2_relbias_pos(ctx, table, sinks, cell, tokp, mask, GGML_TYPE_F16);
                ggml_tensor * fa_d = ggml_flash_attn_ext(ctx, qp, kk, vv, dense, 1.0f/sqrtf(128.0f), 0.0f, 0.0f);
                ggml_prec_set_acc(fa_d, GGML_PREC_F32);
                ggml_tensor * desc = ggml_inkling2_relbias_desc(ctx, table, sinks, cell, tokp, mask);
                ggml_tensor * fa_r = ggml_flash_attn_ext(ctx, qp, kk, vv, mask, 1.0f/sqrtf(128.0f), 0.0f, 0.0f);
                ggml_prec_set_acc(fa_r, GGML_PREC_F32);
                ggml_flash_attn_ext_set_relbias(fa_r, desc);
                ggml_tensor * diff = ggml_sub(ctx, fa_r, fa_d);
                out = ggml_add(ctx, fa_r, ggml_scale(ctx, diff, 1e4f));
            } break;
            case CONV_STATE:
            case CONV_DYN: {
''')
edit(T, "            if (chain == RELBIAS_POS && (std::string(t->name) == \"cell\" || std::string(t->name) == \"tokp\" || std::string(t->name) == \"pos\")) {",
     "            if ((chain == RELBIAS_POS || chain == FA_REL) && (std::string(t->name) == \"cell\" || std::string(t->name) == \"tokp\" || std::string(t->name) == \"pos\")) {")
edit(T, "        test_cases.emplace_back(new test_inkling2(test_inkling2::RELBIAS_POS, 16, n_tok, 24, 4098));\n",
     "        test_cases.emplace_back(new test_inkling2(test_inkling2::RELBIAS_POS, 16, n_tok, 24, 4098));\n        if (n_tok > 1) {\n            test_cases.emplace_back(new test_inkling2(test_inkling2::FA_REL, 0, n_tok, 24, 512));\n            test_cases.emplace_back(new test_inkling2(test_inkling2::FA_REL, 0, n_tok, 24, 2048));\n            test_cases.emplace_back(new test_inkling2(test_inkling2::FA_REL, 1, n_tok, 24, 1024));\n        }\n")
print("patched relfa")
