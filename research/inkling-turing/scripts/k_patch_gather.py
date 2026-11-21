# Fold the per-expert down-scale gather (REPEAT + GET_ROWS) into the MoE weighted reduction (turing-iq3-opt)
p = '/home/marvin/llm/experiments/inkling-kernels/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu'
s = open(p).read()
if 'ggml_cuda_try_fuse_moe_scale_gather' not in s:
    helper = r'''
// turing-iq3-opt: REPEAT(scale table) -> GET_ROWS(., top-k ids) feeding the expert_scale input of the MoE weighted
// reduction: the reduction reads scale[ids[e]] itself, 3 launches -> 1. Same products and order: bit-exact.
// GGML_CUDA_INK_T_DISABLE bit 64.
static __global__ void moe_weighted_reduction_gather_f32(const float * __restrict__ experts, const float * __restrict__ scale_tab,
        const int32_t * __restrict__ ids, const float * __restrict__ weights, float * __restrict__ dst,
        const int64_t n_embd, const int n_expert_used) {
    const int64_t col = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }
    float sum = (experts[col] * scale_tab[ids[0]]) * weights[0];
    for (int e = 1; e < n_expert_used; ++e) {
        sum += (experts[(int64_t) e * n_embd + col] * scale_tab[ids[e]]) * weights[e];
    }
    dst[col] = sum;
}

static int ggml_cuda_try_fuse_moe_scale_gather(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = [] {
        const char * env = getenv("GGML_CUDA_INK_T_DISABLE");
        return env != nullptr && (atoi(env) & 64) != 0;
    }();
    const ggml_tensor * rep = cgraph->nodes[i];
    if (disabled || rep->op != GGML_OP_REPEAT || rep->type != GGML_TYPE_F32 || rep->ne[0] != 1 || rep->ne[2] != 1 ||
            rep->ne[3] != 1 || !ggml_is_contiguous(rep->src[0]) || rep->src[0]->type != GGML_TYPE_F32 ||
            ggml_nelements(rep->src[0]) != rep->ne[1] || ggml_node_get_use_count(cgraph, i) != 1) {
        return 0;
    }
    int j = i + 1;
    while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) ++j;
    if (j >= cgraph->n_nodes) return 0;
    const ggml_tensor * gr  = cgraph->nodes[j];
    const ggml_tensor * ids = gr->src[1];
    if (gr->op != GGML_OP_GET_ROWS || gr->src[0] != rep || ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) ||
            ggml_nrows(ids) != 1 || gr->ne[0] != 1 || gr->ne[1] != ids->ne[0] || gr->ne[2] != 1 ||
            ggml_node_get_use_count(cgraph, j) != 1) {
        return 0;
    }
    int k = j + 1;
    while (k < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[k])) ++k;
    if (k >= cgraph->n_nodes) return 0;
    ggml_cuda_moe_weighted_reduction_match m;
    if (!ggml_cuda_match_moe_weighted_reduction(cgraph, k, m) || m.expert_scale != gr) {
        return 0;
    }
    const int64_t n_embd = m.experts->ne[0];
    const int     n_used = (int) m.experts->ne[1];
    if (m.experts->ne[2]*m.experts->ne[3] != 1 || n_used != ids->ne[0] || !ggml_is_contiguous(m.experts) ||
            !ggml_is_contiguous(m.weights) || !ggml_is_contiguous(m.dst)) {
        return 0;
    }
    constexpr int threads = 256;
    const dim3 blocks(1, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_gather_f32<<<blocks, threads, 0, cuda_ctx->stream()>>>((const float *) m.experts->data,
        (const float *) rep->src[0]->data, (const int32_t *) ids->data, (const float *) m.weights->data, (float *) m.dst->data,
        n_embd, n_used);
    CUDA_CHECK(cudaGetLastError());
    return (k + m.node_count - 1) - i;
}

'''
    anchor = 'static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {'
    assert s.count(anchor) == 1
    s = s.replace(anchor, helper + anchor, 1)
    k = s.index(anchor) + len(anchor)
    s = s[:k] + '''
    if (cuda_ctx->stream_context().concurrent_events.empty()) {
        if (const int n_gather = ggml_cuda_try_fuse_moe_scale_gather(cuda_ctx, cgraph, i)) {
            return n_gather;
        }
    }
''' + s[k:]
    open(p, 'w').write(s)
print('patched gather')
