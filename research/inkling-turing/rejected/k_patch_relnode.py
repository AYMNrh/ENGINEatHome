import re, sys
R = "llama.cpp/"

def edit(path, fn):
    s = open(R + path).read()
    s2 = fn(s)
    assert s2 != s, path
    open(R + path, "w").write(s2)

def rep(s, old, new, count=1):
    assert s.count(old) == count, (old[:60], s.count(old))
    return s.replace(old, new)

# ---- ggml-inkling2.h (new header, so that ggml.h does not change and not everything rebuilds) -----------------------------
open(R + "ggml/include/ggml-inkling2.h", "w").write("""#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

    // Inkling 2: the attention mask with the relative-position bias of every head, as ONE node:
    //   dst[kv, t, h] = type((tab[clamp(pos[kv, t], 0, n_rows - 1), h, t] - sinks[0, h, t]) + mask[kv, t])
    // tab F32 [n_rows, n_head, n_tokens] (rows contiguous), sinks F32 [1, n_head, n_tokens], pos I32 [n_kv, n_tokens],
    // mask F16/F32 [n_kv, n_tokens], dst F16/F32 [n_kv, n_tokens, n_head]. Same arithmetic as the chain SUB, GET_ROWS, ADD, CPY
    // that it replaces, but the F32 [n_head, n_kv, n_tokens] intermediates of that chain never exist in the graph, so
    // ggml-alloc does not plan memory for them. Built on GGML_OP_CUSTOM; the CUDA backend runs it with its own kernel.
    GGML_API struct ggml_tensor * ggml_inkling2_relbias(
            struct ggml_context * ctx,
            struct ggml_tensor  * tab,
            struct ggml_tensor  * sinks,
            struct ggml_tensor  * pos,
            struct ggml_tensor  * mask,
            enum ggml_type        type);

    GGML_API bool ggml_is_inkling2_relbias(const struct ggml_tensor * t);

#ifdef __cplusplus
}
#endif
""")

# ---- ggml.c ------------------------------------------------------------------------------------------------------
def f(s):
    old = "struct ggml_tensor * ggml_custom_inplace(\n"
    new = r'''// ggml_inkling2_relbias

static const char ggml_inkling2_relbias_tag = 0; // identifies the custom op to the backends

static void ggml_inkling2_relbias_cpu(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(userdata);

    const struct ggml_tensor * tab  = dst->src[0];
    const struct ggml_tensor * s    = dst->src[1];
    const struct ggml_tensor * pos  = dst->src[2];
    const struct ggml_tensor * mask = dst->src[3];

    const int64_t n_rows = tab->ne[0];
    const int64_t n_head = tab->ne[1];
    const int64_t n_tok  = tab->ne[2];
    const int64_t n_kv   = pos->ne[0];

    for (int64_t t = ith; t < n_tok; t += nth) {
        const int32_t * p = (const int32_t *) ((const char *) pos->data + t*pos->nb[1]);
        for (int64_t h = 0; h < n_head; ++h) {
            const float * tb = (const float *) ((const char *) tab->data + h*tab->nb[1] + t*tab->nb[2]);
            const float   sv = *(const float *) ((const char *) s->data + h*s->nb[1] + t*s->nb[2]);
            for (int64_t kv = 0; kv < n_kv; ++kv) {
                int64_t d = p[kv];
                d = d < 0 ? 0 : d;
                d = d > n_rows - 1 ? n_rows - 1 : d;
                const float m = mask->type == GGML_TYPE_F16 ?
                    GGML_FP16_TO_FP32(((const ggml_fp16_t *) ((const char *) mask->data + t*mask->nb[1]))[kv]) :
                    ((const float *) ((const char *) mask->data + t*mask->nb[1]))[kv];
                const float b = tb[d] - sv;
                const float v = b + m;
                char * out = (char *) dst->data + ((h*n_tok + t)*n_kv + kv)*ggml_type_size(dst->type);
                if (dst->type == GGML_TYPE_F16) {
                    *(ggml_fp16_t *) out = GGML_FP32_TO_FP16(v);
                } else {
                    *(float *) out = v;
                }
            }
        }
    }
}

struct ggml_tensor * ggml_inkling2_relbias(
        struct ggml_context * ctx,
        struct ggml_tensor  * tab,
        struct ggml_tensor  * sinks,
        struct ggml_tensor  * pos,
        struct ggml_tensor  * mask,
        enum ggml_type        type) {
    GGML_ASSERT(tab->type == GGML_TYPE_F32 && tab->nb[0] == sizeof(float) && tab->ne[3] == 1);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32 && sinks->ne[0] == 1 && sinks->ne[1] == tab->ne[1] && sinks->ne[2] == tab->ne[2] && sinks->ne[3] == 1);
    GGML_ASSERT(pos->type == GGML_TYPE_I32 && pos->ne[1] == tab->ne[2] && pos->ne[2] == 1 && pos->ne[3] == 1);
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->ne[0] == pos->ne[0] && mask->ne[1] == tab->ne[2] && mask->ne[2] == 1 && mask->ne[3] == 1);
    GGML_ASSERT(type == GGML_TYPE_F16 || type == GGML_TYPE_F32);

    struct ggml_tensor * args[4] = { tab, sinks, pos, mask };
    return ggml_custom_4d(ctx, type, pos->ne[0], tab->ne[2], tab->ne[1], 1, args, 4, ggml_inkling2_relbias_cpu,
            GGML_N_TASKS_MAX, (void *) &ggml_inkling2_relbias_tag);
}

bool ggml_is_inkling2_relbias(const struct ggml_tensor * t) {
    if (t->op != GGML_OP_CUSTOM) {
        return false;
    }
    struct ggml_custom_op_params p;
    memcpy(&p, t->op_params, sizeof(p));
    return p.userdata == (void *) &ggml_inkling2_relbias_tag;
}

'''
    return rep(s, old, new + old)
edit("ggml/src/ggml.c", f)

# ---- CUDA: inkling2.cuh / inkling2.cu / ggml-cuda.cu ------------------------------------------------------------------
def f(s):
    return s.rstrip("\n") + '''

// the relative-position bias mask as one graph node (ggml_inkling2_relbias, GGML_OP_CUSTOM): the kernel of the RELBIAS fusion
bool ggml_cuda_inkling2_relbias_node_supported(const ggml_tensor * dst);
void ggml_cuda_inkling2_relbias_node(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
'''
edit("ggml/src/ggml-cuda/inkling2.cuh", f)

def f(s):
    return s.rstrip("\n") + '''

// ---- the mask as one graph node (ggml_inkling2_relbias) --------------------------------------------------------------------
bool ggml_cuda_inkling2_relbias_node_supported(const ggml_tensor * dst) {
    const ggml_tensor * tab  = dst->src[0];
    const ggml_tensor * s    = dst->src[1];
    const ggml_tensor * pos  = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (!tab || !s || !pos || !mask) {
        return false;
    }
    const int64_t n_rows = tab->ne[0];
    const int64_t n_head = tab->ne[1];
    const int64_t n_tok  = tab->ne[2];
    const int64_t n_kv   = pos->ne[0];
    return inkling2_is_f32_rows(tab) && tab->ne[3] == 1 &&
        s->type == GGML_TYPE_F32 && ggml_is_contiguous(s) && s->ne[0] == 1 && s->ne[1] == n_head && s->ne[2] == n_tok && s->ne[3] == 1 &&
        pos->type == GGML_TYPE_I32 && pos->nb[0] == sizeof(int32_t) && pos->ne[1] == n_tok && pos->ne[2] == 1 &&
        (mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32) && mask->nb[0] == ggml_type_size(mask->type) &&
        mask->ne[0] == n_kv && mask->ne[1] == n_tok && mask->ne[2] == 1 && mask->ne[3] == 1 &&
        (dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32) && ggml_is_contiguous(dst) &&
        dst->ne[0] == n_kv && dst->ne[1] == n_tok && dst->ne[2] == n_head && dst->ne[3] == 1 &&
        n_kv <= INT_MAX && n_rows <= INT_MAX && n_tok <= 65535 && n_head <= 65535;
}

void ggml_cuda_inkling2_relbias_node(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_inkling2_args args;
    args.type    = GGML_CUDA_INKLING2_RELBIAS;
    args.n_nodes = 1;
    args.dst     = dst;
    args.src[0]  = dst->src[0];
    args.src[1]  = dst->src[1];
    args.src[2]  = dst->src[2];
    args.src[3]  = dst->src[3];
    inkling2_relbias(ctx, args);
}
'''
edit("ggml/src/ggml-cuda/inkling2.cu", f)

def f(s):
    s = rep(s, "static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {\n    switch (dst->op) {\n",
            "static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {\n    switch (dst->op) {\n"
            "        case GGML_OP_CUSTOM:\n            if (!ggml_is_inkling2_relbias(dst) || !ggml_cuda_inkling2_relbias_node_supported(dst)) {\n                return false;\n            }\n"
            "            ggml_cuda_inkling2_relbias_node(ctx, dst);\n            break;\n")
    # supports_op: before the first ARGMAX/COUNT_EQUAL true-case in supports_op
    old = "        case GGML_OP_ARGMAX:\n        case GGML_OP_COUNT_EQUAL:\n            return true;\n"
    new = "        case GGML_OP_CUSTOM:\n            return ggml_is_inkling2_relbias(op) && ggml_cuda_inkling2_relbias_node_supported(op);\n" + old
    return rep(s, old, new)
edit("ggml/src/ggml-cuda/ggml-cuda.cu", f)

# ---- llama-graph.h/.cpp -------------------------------------------------------------------------------------------------
def f(s):
    old = "    //\n    // pooling\n    //\n"
    new = '''    // the same as build_attn above, with the per-head mask built by ONE node (ggml_inkling2_relbias) from the bias table
    // instead of the kq_b chain: no F32 [n_head, n_kv, n_tokens] intermediates for the allocator to plan. n_stream must be 1.
    ggml_tensor * build_attn_relbias(
            llm_graph_input_mem_hybrid_iswa_rel * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur,   // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur,   // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur,   // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * rel_tab, // [n_rows, n_head_q, n_tokens]: r projected on the bias table
            ggml_tensor * rel_s,   // [1, n_head_q, n_tokens]: the sink logit of each query, subtracted from the table
            ggml_tensor * rel_pos, // [n_kv, n_tokens]: the distances
            ggml_tensor * sinks,   // [n_head_q]
                  float   kq_scale,
                    int   il) const;

'''
    return rep(s, old, new + old)
edit("src/llama-graph.h", f)

def f(s):
    old = "void llm_graph_context::build_dense_out(\n"
    new = '''ggml_tensor * llm_graph_context::build_attn_relbias(
        llm_graph_input_mem_hybrid_iswa_rel * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * rel_tab,
        ggml_tensor * rel_s,
        ggml_tensor * rel_pos,
        ggml_tensor * sinks,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * inp_attn = inp->get_attn();

    const auto & attn_mask = is_swa ? inp_attn->self_attn_mask_swa : inp_attn->self_attn_mask;

    GGML_ASSERT(attn_mask.rows == nullptr);
    GGML_ASSERT(attn_mask.raw->ne[3] == 1);

    ggml_tensor * kq_mask = ggml_inkling2_relbias(ctx0, rel_tab, rel_s, rel_pos, attn_mask.raw,
            cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32);
    cb(kq_mask, "kq_mask_rel_pos", il);

    return build_attn_iswa_impl(*this, inp_attn, wo, wo_b, wo_s, q_cur, k_cur, v_cur, nullptr, kq_mask, nullptr, sinks, nullptr, kq_scale, il);
}

'''
    return rep(s, old, new + old)
edit("src/llama-graph.cpp", f)

# ---- inkling2.cpp ---------------------------------------------------------------------------------------------------------
def f(s):
    old = '''        // project r on the bias table first, then gather by distance: only the gathered bias is n_kv x n_tokens x n_head
        ggml_tensor * table = ggml_mul_mat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), Rcur);

        // the sink logit t is per query: exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s))), with a zero sink
        table = ggml_sub(ctx0, table, sinks);
'''
    new = '''        // project r on the bias table first, then gather by distance: only the gathered bias is n_kv x n_tokens x n_head
        ggml_tensor * table = ggml_mul_mat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), Rcur);

        // several tokens: the mask of each head is one node (ggml_inkling2_relbias), so that no F32 [n_head, n_kv, n_tokens]
        // intermediates are planned by the allocator (at 256K they are most of the compute buffer); one token keeps the
        // chain, which the decode fusion takes. LLAMA_INK_RELBIAS_NODE=0 keeps the chain for all batch sizes.
        static const bool relbias_node = [] { const char * e = getenv("LLAMA_INK_RELBIAS_NODE"); return !e || atoi(e) != 0; }();
        if (relbias_node && n_tokens > 1 && n_stream == 1) {
            ggml_tensor * rel_sinks = sinks;

            sinks = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);

            cur = build_attn_relbias(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, table, rel_sinks, rel_pos, sinks, 1.0f/sqrtf(n_embd_head), il);
        } else {
        // the sink logit t is per query: exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s))), with a zero sink
        table = ggml_sub(ctx0, table, sinks);
'''
    s = rep(s, old, new)
    old2 = '''        cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, kq_b, sinks, 1.0f/sqrtf(n_embd_head), il);
'''
    new2 = old2 + "        }\n"
    return rep(s, old2, new2)
edit("src/models/inkling2.cpp", f)

# ---- test-backend-ops ------------------------------------------------------------------------------------------------------
def f(s):
    s = rep(s, "    enum chain_type { CLIP, CONV, SITU, MIX, CTRL, RELBIAS, CONV_STATE, CONV_DYN };",
               "    enum chain_type { CLIP, CONV, SITU, MIX, CTRL, RELBIAS, CONV_STATE, CONV_DYN, RELBIAS_NODE };")
    s = rep(s, '"INKLING2_RELBIAS", "INKLING2_CONV_STATE", "INKLING2_CONV_DYN" };', '"INKLING2_RELBIAS", "INKLING2_CONV_STATE", "INKLING2_CONV_DYN", "INKLING2_RELBIAS_NODE" };')
    s = rep(s, "return chain == MIX || chain == CTRL || chain == RELBIAS ? 5e-4 : 1e-6;", "return chain == MIX || chain == CTRL || chain == RELBIAS || chain == RELBIAS_NODE ? 5e-4 : 1e-6;")
    s = rep(s, "const int range = chain == RELBIAS ? 64 + 1 : n_expert;", "const int range = chain == RELBIAS || chain == RELBIAS_NODE ? 64 + 1 + (n_tok & 1 ? 40 : 0) : n_expert; // some distances past the table")
    old = "            case CONV_STATE:\n            case CONV_DYN: {\n"
    new = '''            case RELBIAS_NODE: {
                // the same mask as one node, against the unfused chain: out = node + 1e4*(node - chain) is the node where the
                // two agree bit for bit, on the CPU and the CUDA backend alike
                const int64_t n_rel  = n;
                const int64_t n_head = k;
                const int64_t n_kv   = v;
                const int64_t n_ext  = 64;
                ggml_tensor * w     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rel, n_ext);
                ggml_tensor * r     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_rel, n_head, n_tok);
                ggml_tensor * sinks = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_head, n_tok);
                ggml_tensor * pos   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, n_tok);
                ggml_tensor * mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tok);
                ggml_set_name(pos, "pos");
                ggml_tensor * table = ggml_mul_mat(ctx, ggml_pad(ctx, w, 0, 1, 0, 0), r);
                ggml_tensor * node  = ggml_inkling2_relbias(ctx, table, sinks, pos, mask, GGML_TYPE_F16);
                ggml_tensor * t2    = ggml_sub(ctx, table, sinks);
                t2 = ggml_cont(ctx, ggml_permute(ctx, t2, 1, 0, 2, 3));
                ggml_tensor * kq_b = ggml_get_rows(ctx, t2, pos);
                kq_b = ggml_reshape_4d(ctx, kq_b, n_head, n_kv, n_tok, 1);
                kq_b = ggml_cont(ctx, ggml_permute(ctx, kq_b, 2, 0, 1, 3));
                ggml_tensor * chain_out = ggml_cast(ctx, ggml_add(ctx, kq_b, ggml_cast(ctx, mask, GGML_TYPE_F32)), GGML_TYPE_F16);
                ggml_tensor * node32 = ggml_cast(ctx, node, GGML_TYPE_F32);
                ggml_tensor * diff   = ggml_sub(ctx, node32, ggml_cast(ctx, chain_out, GGML_TYPE_F32));
                out = ggml_add(ctx, node32, ggml_scale(ctx, diff, 1e4f));
            } break;
'''
    s = rep(s, old, new + old)
    old = "        test_cases.emplace_back(new test_inkling2(test_inkling2::RELBIAS, 16, n_tok, 24, 1024 + n_tok));\n"
    new = old + '''        test_cases.emplace_back(new test_inkling2(test_inkling2::RELBIAS_NODE, 16, n_tok, 24, 256));
        test_cases.emplace_back(new test_inkling2(test_inkling2::RELBIAS_NODE, 16, n_tok, 24, 1024 + n_tok));
        test_cases.emplace_back(new test_inkling2(test_inkling2::RELBIAS_NODE, 16, n_tok, 24, 4096 + 2));
'''
    return rep(s, old, new)
edit("tests/test-backend-ops.cpp", f)

def inc(path, anchor):
    edit(path, lambda s: rep(s, anchor + chr(10), anchor + chr(10) + '#include "ggml-inkling2.h"' + chr(10)))
inc("ggml/src/ggml.c", '#include "ggml.h"')
inc("ggml/src/ggml-cuda/ggml-cuda.cu", '#include "ggml.h"')
inc("src/llama-graph.cpp", '#include "llama-graph.h"')
inc("tests/test-backend-ops.cpp", '#include "ggml.h"')
print("patched")
