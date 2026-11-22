// ---------------------------------------------------------------------------------------------------------------------
// RELBIAS_T: PAD(W) -> MUL_MAT(tab = pad(W)^T r) -> RELBIAS chain -> FILL(sinks) for one token, ONE launch instead of 4
// (pad, cuBLAS GEMM, relbias mask, fill). The bias table is never materialised: a cell only needs
// tab[d, h] = sum_k W[k, d] r[k, h] (16 terms; the pad row d = n_W is zero), summed in order with fmaf. The unfused
// cuBLAS GEMM sums in its own order, so this fusion may differ from it in the last bits (the only non-exact one here).

template <int vw, typename T_mask, typename T_dst>
static __global__ void inkling2t_relbias_f32(const float * W, const int nW, const float * r, const int64_t r_s1,
        const float * s, const int32_t * pos, const T_mask * mask, T_dst * dst, const int n_kv, const int n_rows,
        float * fill, const int n_fill, const float fill_v) {
    ggml_cuda_pdl_lc();
    constexpr int NK = 16;
    const int kv = (blockIdx.x*blockDim.x + threadIdx.x)*vw;
    const int h  = blockIdx.z;
    ggml_cuda_pdl_sync();

    if (fill && blockIdx.x == 0 && h == 0) {
        for (int j = threadIdx.x; j < n_fill; j += blockDim.x) {
            fill[j] = fill_v;
        }
    }
    if (kv >= n_kv) {
        return;
    }

    float rr[NK];
#pragma unroll
    for (int k = 0; k < NK; ++k) {
        rr[k] = r[h*r_s1 + k];
    }
    const float sh = s[h];

    T_dst v[vw];
#pragma unroll
    for (int c = 0; c < vw; ++c) {
        const int   d = min(max(pos[kv + c], 0), n_rows - 1);
        const float m = ggml_cuda_cast<float>(mask[kv + c]);
        float tab = 0.0f;
        if (d < nW) {
            const float * w = W + (int64_t) d*NK;
#pragma unroll
            for (int k = 0; k < NK; ++k) {
                tab = fmaf(w[k], rr[k], tab);
            }
        }
        v[c] = ggml_cuda_cast<T_dst>(__fadd_rn(__fsub_rn(tab, sh), m));
    }
    T_dst * out = dst + (int64_t) h*n_kv + kv;
    if constexpr (vw == 2 && std::is_same_v<T_dst, half>) {
        *(half2 *) out = make_half2(v[0], v[1]);
    } else {
#pragma unroll
        for (int c = 0; c < vw; ++c) {
            out[c] = v[c];
        }
    }
}

static bool inkling2t_match_relbias_t(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {
    if (!inkling2t_enabled(2)) {
        return false;
    }
    const ggml_tensor * pad = cgraph->nodes[i];
    const ggml_tensor * W   = pad->src[0];
    if (pad->op != GGML_OP_PAD || W->type != GGML_TYPE_F32 || !ggml_is_contiguous(W) || W->ne[0] != 16 || W->ne[2] != 1 ||
            W->ne[3] != 1 || pad->ne[0] != W->ne[0] || pad->ne[1] != W->ne[1] + 1 || pad->ne[2] != 1 || pad->ne[3] != 1 ||
            ggml_get_op_params_i32(pad, 0) != 0 || ggml_get_op_params_i32(pad, 1) != 0 ||
            ggml_get_op_params_i32(pad, 2) != 0 || ggml_get_op_params_i32(pad, 3) != 1) {
        INK_T_FAIL();
    }
    int k = inkling2t_skip_views(cgraph, i + 1);
    if (k >= cgraph->n_nodes) INK_T_FAIL();
    const ggml_tensor * mm = cgraph->nodes[k];
    const ggml_tensor * r  = mm->src[1];
    if (mm->op != GGML_OP_MUL_MAT || mm->src[0] != pad || r->type != GGML_TYPE_F32 || r->nb[0] != sizeof(float) ||
            r->ne[0] != 16 || r->ne[2] != 1 || r->ne[3] != 1 || mm->ne[0] != pad->ne[1] || mm->ne[1] != r->ne[1] || mm->ne[2] != 1) {
        INK_T_FAIL(); // one token
    }
    k = inkling2t_skip_views(cgraph, k + 1);
    ggml_cuda_inkling2_args rb;
    if (k >= cgraph->n_nodes || !inkling2_match_relbias(cgraph, k, rb) || rb.src[0] != mm) INK_T_FAIL();
    int end = k + rb.n_nodes - 1;

    const ggml_tensor * fill = nullptr;
    const int kf = inkling2t_skip_views(cgraph, end + 1);
    if (kf < cgraph->n_nodes && cgraph->nodes[kf]->op == GGML_OP_FILL && cgraph->nodes[kf]->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(cgraph->nodes[kf]) && ggml_nelements(cgraph->nodes[kf]) <= 4096 &&
            (cgraph->nodes[kf]->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        fill = cgraph->nodes[kf];
        end  = kf;
    }

    const ggml_tensor * s    = rb.src[1];
    const ggml_tensor * pos  = rb.src[2];
    const ggml_tensor * mask = rb.src[3];
    const ggml_tensor * dst  = rb.dst;
    const bool types_ok = (mask->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) ||
                          (mask->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    if (!types_ok || mm->ne[1] != s->ne[1] || pos->ne[1] != 1) INK_T_FAIL();

    if (!inkling2t_closed(cgraph, i, end, { dst, fill })) INK_T_FAIL();

    for (const ggml_tensor * in : { W, pos, mask }) {
        if (inkling2_overlap(dst, in) || (fill && inkling2_overlap(fill, in))) INK_T_FAIL();
    }
    if (fill && inkling2_overlap(fill, dst)) INK_T_FAIL();
    // r and s die in the chain, so the mask or the sinks fill may be planned over them: stage them into scratch then
    int stage = 0;
    if (inkling2_overlap(dst, r) || (fill && inkling2_overlap(fill, r))) stage |= 1;
    if (inkling2_overlap(dst, s) || (fill && inkling2_overlap(fill, s))) stage |= 2;
    if ((stage & 1) && r->nb[1] != r->ne[0]*sizeof(float)) INK_T_FAIL();
    if ((stage & 2) && !ggml_is_contiguous(s)) INK_T_FAIL();
    args.n_ext = stage;

    args.type    = GGML_CUDA_INKLING2_RELBIAS_T;
    args.n_nodes = end - i + 1;
    args.dst     = const_cast<ggml_tensor *>(dst);
    args.ext[0]  = W;  args.ext[1] = pad; args.ext[2] = r; args.ext[3] = s; args.ext[4] = pos; args.ext[5] = mask;
    args.ext[6]  = fill;
    return true;
}

static void inkling2t_relbias_t(ggml_backend_cuda_context & ctx, const ggml_cuda_inkling2_args & a) {
    const ggml_tensor * W = a.ext[0], * pad = a.ext[1], * r = a.ext[2], * s = a.ext[3], * pos = a.ext[4], * mask = a.ext[5];
    const ggml_tensor * fill = a.ext[6];
    ggml_tensor * dst = a.dst;

    const int n_kv   = (int) pos->ne[0];
    const int n_head = (int) r->ne[1];
    const int n_rows = (int) pad->ne[1];
    const int nW     = (int) W->ne[1];
    float fill_v = 0.0f;
    if (fill) {
        fill_v = ggml_get_op_params_f32(fill, 0);
    }
    const bool vec2 = n_kv % 2 == 0 && dst->type == GGML_TYPE_F16 && (uintptr_t) dst->data % 4 == 0;
    const int  bs   = 256;
    const int  vw   = vec2 ? 2 : 1;
    const ggml_cuda_kernel_launch_params params(dim3((n_kv/vw + bs - 1)/bs, 1, n_head), dim3(bs, 1, 1), 0, ctx.stream());
    int64_t r_s1 = r->nb[1]/sizeof(float);
    const float * rd = (const float *) r->data;
    const float * sd = (const float *) s->data;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool());
    if (a.n_ext) {
        float * sc = scratch.alloc(ggml_nelements(r) + ggml_nelements(s));
        inkling2t_stage_p sp = {};
        int nb = 0;
        if (a.n_ext & 1) { sp.src[nb] = rd; sp.dst[nb] = sc; sp.n[nb] = (int) ggml_nelements(r); rd = sc; sc += sp.n[nb]; ++nb; }
        if (a.n_ext & 2) { sp.src[nb] = sd; sp.dst[nb] = sc; sp.n[nb] = (int) ggml_nelements(s); sd = sc; ++nb; }
        const ggml_cuda_kernel_launch_params sparams(dim3(1, 1, 1), dim3(512, 1, 1), 0, ctx.stream());
        ggml_cuda_kernel_launch(inkling2t_stage_f32, sparams, sp);
    }
    float * fd = fill ? (float *) fill->data : nullptr;
    const int nf = fill ? (int) ggml_nelements(fill) : 0;
    if (dst->type == GGML_TYPE_F16) {
        if (vec2) {
            ggml_cuda_kernel_launch(inkling2t_relbias_f32<2, half, half>, params, (const float *) W->data, nW, rd, r_s1,
                sd, (const int32_t *) pos->data, (const half *) mask->data, (half *) dst->data, n_kv, n_rows, fd, nf, fill_v);
        } else {
            ggml_cuda_kernel_launch(inkling2t_relbias_f32<1, half, half>, params, (const float *) W->data, nW, rd, r_s1,
                sd, (const int32_t *) pos->data, (const half *) mask->data, (half *) dst->data, n_kv, n_rows, fd, nf, fill_v);
        }
    } else {
        ggml_cuda_kernel_launch(inkling2t_relbias_f32<1, float, float>, params, (const float *) W->data, nW, rd, r_s1,
            sd, (const int32_t *) pos->data, (const float *) mask->data, (float *) dst->data, n_kv, n_rows, fd, nf, fill_v);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// ROUTE_TAIL: the normalisation of the top-k expert weights for one token, ONE launch instead of 4:
//   SUM_ROWS(w) -> SCALE (clamp/eps) -> DIV(w, .) -> SCALE (expert_weights_scale)
// One 512-thread block: the sum is reduce_row_f32 with the 512 threads sum_rows_f32_cuda uses for a single row, the
// elementwise ops are those of scale_f32 / op_div, so the results are the same bits.

static __global__ void __launch_bounds__(512) inkling2t_route_tail_f32(const float * w, float * out, const int n,
        const float s1, const float b1, const float s2, const float b2) {
    ggml_cuda_pdl_lc();
    const float sum = reduce_row_f32(w, n);
    const float den = s1 * sum + b1;
    if ((int) threadIdx.x < n) {
        const float v = w[threadIdx.x] / den;
        out[threadIdx.x] = s2 * v + b2;
    }
}

static bool inkling2t_match_route_tail(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {
    if (!inkling2t_enabled(4)) {
        return false;
    }
    const ggml_tensor * sr = cgraph->nodes[i];
    const ggml_tensor * x  = sr->src[0];
    if (sr->op != GGML_OP_SUM_ROWS || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || ggml_nrows(x) != 1 ||
            x->ne[0] > 512) {
        INK_T_FAIL();
    }
    const int n = (int) x->ne[0];
    int k = inkling2t_skip_views(cgraph, i + 1);
    if (k >= cgraph->n_nodes) INK_T_FAIL();
    const ggml_tensor * sc1 = cgraph->nodes[k];
    if (sc1->op != GGML_OP_SCALE || sc1->src[0] != sr || ggml_nelements(sc1) != 1) INK_T_FAIL();
    k = inkling2t_skip_views(cgraph, k + 1);
    if (k >= cgraph->n_nodes) INK_T_FAIL();
    const ggml_tensor * dv = cgraph->nodes[k];
    if (dv->op != GGML_OP_DIV || dv->src[1] != sc1 || dv->src[0]->data != x->data || !ggml_is_contiguous(dv->src[0]) ||
            ggml_nelements(dv->src[0]) != n || !ggml_is_contiguous(dv) || ggml_nelements(dv) != n) INK_T_FAIL();
    k = inkling2t_skip_views(cgraph, k + 1);
    if (k >= cgraph->n_nodes) INK_T_FAIL();
    const ggml_tensor * sc2 = cgraph->nodes[k];
    if (sc2->op != GGML_OP_SCALE || inkling2t_base(sc2->src[0]) != dv || sc2->src[0]->data != dv->data ||
            !ggml_is_contiguous(sc2) || ggml_nelements(sc2) != n || sc2->type != GGML_TYPE_F32) INK_T_FAIL();

    if (!inkling2t_closed(cgraph, i, k, { sc2 })) INK_T_FAIL();
    // out may only sit exactly on w (each thread reads its own w again before writing the same element)
    if (inkling2_overlap(sc2, x) && sc2->data != x->data) INK_T_FAIL();

    args.type    = GGML_CUDA_INKLING2_ROUTE_TAIL;
    args.n_nodes = k - i + 1;
    args.dst     = const_cast<ggml_tensor *>(sc2);
    args.ext[0]  = x; args.ext[1] = sc1; args.ext[2] = sc2;
    return true;
}

static void inkling2t_route_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_inkling2_args & a) {
    const ggml_tensor * x = a.ext[0], * sc1 = a.ext[1], * sc2 = a.ext[2];
    const ggml_cuda_kernel_launch_params params(dim3(1, 1, 1), dim3(512, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(inkling2t_route_tail_f32, params, (const float *) x->data, (float *) a.dst->data, (int) ggml_nelements(x),
        ggml_get_op_params_f32(sc1, 0), ggml_get_op_params_f32(sc1, 1), ggml_get_op_params_f32(sc2, 0), ggml_get_op_params_f32(sc2, 1));
}

