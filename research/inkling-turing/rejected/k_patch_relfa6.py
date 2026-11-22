R = "llama.cpp/"
NL = chr(10)
def edit(path, old, new, count=1):
    s = open(R + path).read(); assert s.count(old) == count, (path, old[:70], s.count(old)); open(R + path, "w").write(s.replace(old, new))

# 1. API: flash attention takes the four tensors as REAL sources (src[6..9]) so ggml-alloc and the scheduler see the dependencies
edit("ggml/include/ggml-inkling2.h", "    GGML_API void ggml_flash_attn_ext_set_relbias(struct ggml_tensor * fa, struct ggml_tensor * desc);\n",
     "    // flash attention computes the bias itself: tab, sinks, cell and tok become sources 6..9 of the node (real dependencies for the allocator\n    // and the scheduler); the descriptor node above is then only kept in the graph so that prefill and decode have the same op sequence\n    GGML_API void ggml_flash_attn_ext_set_relbias(struct ggml_tensor * fa, struct ggml_tensor * tab, struct ggml_tensor * sinks,\n            struct ggml_tensor * cell, struct ggml_tensor * tok);\n")
edit("ggml/src/ggml.c", """void ggml_flash_attn_ext_set_relbias(struct ggml_tensor * fa, struct ggml_tensor * desc) {
    GGML_ASSERT(fa->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(ggml_is_inkling2_relbias_desc(desc));
    GGML_ASSERT(fa->src[3] == desc->src[4]); // the shared mask
    GGML_ASSERT(fa->src[5] == NULL);         // no kv_rows
    GGML_ASSERT(fa->src[0]->ne[2] == desc->src[0]->ne[1] && fa->src[0]->ne[1] == desc->src[0]->ne[2]);
    fa->src[6] = desc;
}""", """void ggml_flash_attn_ext_set_relbias(struct ggml_tensor * fa, struct ggml_tensor * tab, struct ggml_tensor * sinks,
        struct ggml_tensor * cell, struct ggml_tensor * tok) {
    GGML_ASSERT(fa->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(fa->src[3] && fa->src[3]->ne[2] == 1 && fa->src[3]->ne[0] == cell->ne[0]); // the shared mask
    GGML_ASSERT(fa->src[5] == NULL);                                                      // no kv_rows
    GGML_ASSERT(fa->src[0]->ne[2] == tab->ne[1] && fa->src[0]->ne[1] == tab->ne[2]);
    GGML_ASSERT(tab->type == GGML_TYPE_F32 && sinks->type == GGML_TYPE_F32 && cell->type == GGML_TYPE_I32 && tok->type == GGML_TYPE_I32);
    fa->src[6] = tab;
    fa->src[7] = sinks;
    fa->src[8] = cell;
    fa->src[9] = tok;
}""")
# 2. CPU flash attention reads src[6..9]
edit("ggml/src/ggml-cpu/ops.cpp", """                const ggml_tensor * rel = dst->src[6];
                const ggml_tensor * tab = rel->src[0];
                const int64_t extent = tab->ne[0] - 1;
                const int32_t cp = ((const int32_t *) rel->src[2]->data)[ic];
                const int64_t dist = (int64_t) ((const int32_t *) rel->src[3]->data)[iq1] - cp;""",
     """                const ggml_tensor * tab = dst->src[6];
                const int64_t extent = tab->ne[0] - 1;
                const int32_t cp = ((const int32_t *) dst->src[8]->data)[ic];
                const int64_t dist = (int64_t) ((const int32_t *) dst->src[9]->data)[iq1] - cp;""")
edit("ggml/src/ggml-cpu/ops.cpp", "const float sv = *(const float *) ((const char *) rel->src[1]->data + iq2*rel->src[1]->nb[1] + iq1*rel->src[1]->nb[2]);",
     "const float sv = *(const float *) ((const char *) dst->src[7]->data + iq2*dst->src[7]->nb[1] + iq1*dst->src[7]->nb[2]);")
# 3. CUDA: rel from src[6..9]
edit("ggml/src/ggml-cuda/fattn-common.cuh", """    if (dst->src[6] && ggml_is_inkling2_relbias_desc(dst->src[6])) {
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
    }""", """    if (dst->src[6]) {
        GGML_ASSERT(!kv_rows && !use_sparse && ncols2 == 1 && mask && mask->ne[2] == 1);
        rel.tab     = (const float   *) dst->src[6]->data;
        rel.sink    = (const float   *) dst->src[7]->data;
        rel.cell    = (const int32_t *) dst->src[8]->data;
        rel.tok     = (const int32_t *) dst->src[9]->data;
        rel.tab_s1  = dst->src[6]->nb[1]/sizeof(float);
        rel.tab_s2  = dst->src[6]->nb[2]/sizeof(float);
        rel.sink_s1 = dst->src[7]->nb[1]/sizeof(float);
        rel.sink_s2 = dst->src[7]->nb[2]/sizeof(float);
        rel.n_rows  = (int) dst->src[6]->ne[0];
    }""")
edit("ggml/src/ggml-cuda/fattn.cu", "return enabled && dst->src[6] && ggml_is_inkling2_relbias_desc(dst->src[6]) && cc == GGML_CUDA_CC_TURING", "return enabled && dst->src[6] && cc == GGML_CUDA_CC_TURING")
edit("ggml/src/ggml-cuda/fattn.cu", "    if (dst->src[6] && ggml_is_inkling2_relbias_desc(dst->src[6])) {\n        return ggml_cuda_fattn_rel_ok(device, dst);", "    if (dst->src[6]) {\n        return ggml_cuda_fattn_rel_ok(device, dst);")
# 4. graph: no output flags, the descriptor stays only for the op sequence and is expanded explicitly
edit("src/llama-graph.h", "    mutable ggml_tensor * rel_fa_desc = nullptr;\n", "    mutable ggml_tensor * rel_fa_srcs[4] = {}; // tab, sinks, cell, tok\n")
edit("src/llama-graph.cpp", "        if (rel_fa_desc) {\n            ggml_flash_attn_ext_set_relbias(cur, rel_fa_desc);\n        }\n",
     "        if (rel_fa_srcs[0]) {\n            ggml_flash_attn_ext_set_relbias(cur, rel_fa_srcs[0], rel_fa_srcs[1], rel_fa_srcs[2], rel_fa_srcs[3]);\n        }\n")
s = open(R + "src/llama-graph.cpp").read()
a = s.index("        // flash attention reads the table and the sink logits through the descriptor")
b = s.index("        rel_fa_desc = nullptr;\n", a) + len("        rel_fa_desc = nullptr;\n")
new = """        // the descriptor has no output and nothing consumes it: expanded so that the op sequence (and the node count) of the prefill graph
        // equals that of the decode graph, where the real mask node is built
        ggml_tensor * desc = ggml_inkling2_relbias_desc(ctx0, rel_tab, rel_s, rel_cell, rel_tok, attn_mask.raw);
        cb(desc, "kq_mask_rel_desc", il);
        ggml_build_forward_expand(gf, desc);
        rel_fa_srcs[0] = rel_tab; rel_fa_srcs[1] = rel_s; rel_fa_srcs[2] = rel_cell; rel_fa_srcs[3] = rel_tok;
        ggml_tensor * out = build_attn_iswa_impl(*this, inp_attn, wo, wo_b, wo_s, q_cur, k_cur, v_cur, nullptr, attn_mask.raw, nullptr, sinks, nullptr, kq_scale, il);
        rel_fa_srcs[0] = rel_fa_srcs[1] = rel_fa_srcs[2] = rel_fa_srcs[3] = nullptr;
"""
s = s[:a] + new + s[b:]
open(R + "src/llama-graph.cpp", "w").write(s)
# 5. the unit test
edit("tests/test-backend-ops.cpp", "                ggml_tensor * desc = ggml_inkling2_relbias_desc(ctx, table, sinks, cell, tokp, mask);\n", "")
edit("tests/test-backend-ops.cpp", "                ggml_flash_attn_ext_set_relbias(fa_r, desc);\n", "                ggml_flash_attn_ext_set_relbias(fa_r, table, sinks, cell, tokp);\n")
print("patched6")
