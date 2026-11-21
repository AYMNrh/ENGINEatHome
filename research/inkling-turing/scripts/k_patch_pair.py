# Horizontal fusion of same-input, same-shape quantized decode mat-vecs (turing-iq3-opt)
D = '/home/marvin/llm/experiments/inkling-kernels/llama.cpp/'

# 1) graph order: Q + gate adjacent; K + V adjacent when they read the same input (SWA without controllers)
p = D + 'src/models/inkling2.cpp'
s = open(p).read()
if 'turing-iq3-opt' not in s:
    old = '''        ggml_tensor * Qcur = build_lora_mm(layer.wq, inputs[0]);
        ggml_tensor * Kcur = convolve(build_lora_mm(layer.wk, inputs[1]), layer.attn_k_conv,
                is_swa ? nullptr : build_lora_mm(layer.attn_k_conv_proj, inputs[1]), state, history, recr, 0);
        ggml_tensor * Vcur = convolve(build_lora_mm(layer.wv, inputs[2]), layer.attn_v_conv,
                is_swa ? nullptr : build_lora_mm(layer.attn_v_conv_proj, inputs[2]), state, history, recr, n_embd_k_gqa);'''
    new = '''        // turing-iq3-opt: projections that read the same input are expanded next to each other, so that the backend can
        // run each pair as one mat-vec launch (Q + gate always, K + V when they share the input); values are unchanged
        ggml_tensor * Qcur      = build_lora_mm(layer.wq, inputs[0]);
        ggml_tensor * gate_proj = build_lora_mm(layer.wqkv_gate, inputs[0]);
        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, gate_proj);
        ggml_tensor * Kproj = build_lora_mm(layer.wk, inputs[1]);
        ggml_tensor * Vproj = build_lora_mm(layer.wv, inputs[2]);
        if (inputs[1] == inputs[2]) {
            ggml_build_forward_expand(gf, Kproj);
            ggml_build_forward_expand(gf, Vproj);
        }
        ggml_tensor * Kcur = convolve(Kproj, layer.attn_k_conv,
                is_swa ? nullptr : build_lora_mm(layer.attn_k_conv_proj, inputs[1]), state, history, recr, 0);
        ggml_tensor * Vcur = convolve(Vproj, layer.attn_v_conv,
                is_swa ? nullptr : build_lora_mm(layer.attn_v_conv_proj, inputs[2]), state, history, recr, n_embd_k_gqa);'''
    assert s.count(old) == 1, 'qkv build'
    s = s.replace(old, new)
    old2 = '        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, inputs[0])));'
    assert s.count(old2) == 1
    s = s.replace(old2, '        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate_proj));')
    open(p, 'w').write(s)

# 2) mmvq: pair mode of the shared_up slice (no ids, no gate)
p = D + 'ggml/src/ggml-cuda/mmvq.cu'
s = open(p).read()
if 'pair mode' not in s:
    old = '''        if (fusion->shared_up) {
            GGML_ASSERT(ids && fusion->gate && fusion->shared_gate && fusion->shared_dst);'''
    new = '''        if (fusion->shared_up && !ids && !fusion->gate && !fusion->shared_gate) {
            // turing-iq3-opt pair mode: a second matrix of the same shape and the same input in the extra grid slice
            GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale && fusion->shared_dst);
            GGML_ASSERT(ne11 == 1 && ne12 == 1 && ne13 == 1 && ne02 == 1 && ne03 == 1);
            GGML_ASSERT(fusion->shared_up->type == src0->type && ggml_are_same_shape(fusion->shared_up, src0));
            GGML_ASSERT(fusion->shared_up->nb[1] == nb01 && ggml_is_contiguous(fusion->shared_up));
            GGML_ASSERT(fusion->shared_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->shared_dst));
            GGML_ASSERT(ggml_are_same_shape(fusion->shared_dst, dst));
            fusion_local.shared_up   = fusion->shared_up->data;
            fusion_local.shared_dst  = (float *) fusion->shared_dst->data;
            fusion_local.shared_stride_col_dst = fusion->shared_dst->nb[1] / ts_dst;
        } else if (fusion->shared_up) {
            GGML_ASSERT(ids && fusion->gate && fusion->shared_gate && fusion->shared_dst);'''
    assert s.count(old) == 1, 'shared_up block'
    s = s.replace(old, new)
    open(p, 'w').write(s)

# 3) matcher in ggml_cuda_try_fuse
p = D + 'ggml/src/ggml-cuda/ggml-cuda.cu'
s = open(p).read()
if 'ggml_cuda_try_fuse_mmvq_pair' not in s:
    helper = r'''
// turing-iq3-opt: two adjacent decode mat-vecs with the same input and same-shape quantized weights run as one mmvq
// launch (the second matrix in the shared_up grid slice: same kernel, same per-row arithmetic). GGML_CUDA_INK_T_DISABLE bit 32.
static int ggml_cuda_try_fuse_mmvq_pair(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = [] {
        const char * env = getenv("GGML_CUDA_INK_T_DISABLE");
        return env != nullptr && (atoi(env) & 32) != 0;
    }();
    ggml_tensor * a = cgraph->nodes[i];
    if (disabled || a->op != GGML_OP_MUL_MAT || !cuda_ctx->stream_context().concurrent_events.empty()) {
        return 0;
    }
    int j = i + 1;
    while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
        ++j;
    }
    if (j >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * b    = cgraph->nodes[j];
    ggml_tensor * src1 = a->src[1];
    const ggml_tensor * wa = a->src[0];
    const ggml_tensor * wb = b->src[0];
    if (b->op != GGML_OP_MUL_MAT || b->src[1] != src1 || !(b->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            !ggml_is_quantized(wa->type) || wb->type != wa->type || !ggml_are_same_shape(wa, wb) || wa->nb[1] != wb->nb[1] ||
            !ggml_is_contiguous(wa) || !ggml_is_contiguous(wb) || wa->ne[2] != 1 || wa->ne[3] != 1 ||
            ggml_backend_buffer_get_usage(wa->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            ggml_backend_buffer_get_usage(wb->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            src1->type != GGML_TYPE_F32 || src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || !ggml_is_contiguous(a) || !ggml_is_contiguous(b) ||
            !ggml_are_same_shape(a, b)) {
        return 0;
    }
    // the path the two nodes would take on their own: plain mmvq
    const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
    if (!ggml_cuda_should_use_mmvq(wa->type, cc, src1->ne[1])) {
        return 0;
    }
    // both outputs are written by one launch while it reads src1 and both weight sets
    auto ovl = [](const ggml_tensor * x, const ggml_tensor * y) {
        const char * x0 = (const char *) x->data; const char * y0 = (const char *) y->data;
        return x0 < y0 + ggml_nbytes(y) && y0 < x0 + ggml_nbytes(x);
    };
    if (ovl(a, src1) || ovl(b, src1) || ovl(a, b) || ovl(a, wb) || ovl(b, wa)) {
        return 0;
    }
    ggml_cuda_mm_fusion_args_host fusion{};
    fusion.shared_up  = wb;
    fusion.shared_dst = b;
    ggml_cuda_mul_mat_vec_q(*cuda_ctx, wa, src1, nullptr, a, &fusion);
    return j - i;
}

'''
    anchor = 'static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {'
    assert s.count(anchor) == 1
    s = s.replace(anchor, helper + anchor, 1)
    # call it first inside try_fuse
    k = s.index(anchor) + len(anchor)
    s = s[:k] + '''
    if (const int n_pair = ggml_cuda_try_fuse_mmvq_pair(cuda_ctx, cgraph, i)) {
        return n_pair;
    }
''' + s[k:]
    open(p, 'w').write(s)
print('patched pair')
