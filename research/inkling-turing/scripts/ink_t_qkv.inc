// =====================================================================================================================
// Turing kernel work (branch turing-iq3-opt). Fusions added on top of the inkling2-speed glue fusions, for decode.
// GGML_CUDA_INK_T_DISABLE=<bitmask> turns them off individually (1 = QKV_POST, 2 = RELBIAS_T, 4 = ROUTE_TAIL, 8 = q8_1 cache, 16 = CTRL_T).

static bool inkling2t_enabled(int bit) {
    static const int mask = [] {
        const char * env = getenv("GGML_CUDA_INK_T_DISABLE");
        return env ? atoi(env) : 0;
    }();
    return (mask & bit) == 0;
}

// the tensor a chain of view-like nodes reads
static const ggml_tensor * inkling2t_base(const ggml_tensor * t) {
    while (t && (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE)) {
        t = t->src[0];
    }
    return t;
}

static bool inkling2t_is_viewlike(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_NONE;
}

static int inkling2t_skip_views(const ggml_cgraph * cgraph, int k) {
    while (k < cgraph->n_nodes && inkling2t_is_viewlike(cgraph->nodes[k])) {
        ++k;
    }
    return k;
}

// every node of [i0, i1] that the fusion computes (or a view of one) must only be used inside [i0, i1], except outputs
static bool inkling2t_closed(const ggml_cgraph * cgraph, int i0, int i1, std::initializer_list<const ggml_tensor *> outputs) {
    auto in_range = [&](const ggml_tensor * t) {
        for (int k = i0; k <= i1; ++k) {
            if (cgraph->nodes[k] == t) {
                return true;
            }
        }
        return false;
    };
    for (int k = i0; k <= i1; ++k) {
        const ggml_tensor * n = cgraph->nodes[k];
        bool is_out = false;
        const ggml_tensor * nb = inkling2t_is_viewlike(n) ? inkling2t_base(n) : n;
        for (const ggml_tensor * o : outputs) {
            is_out = is_out || o == n || o == nb;   // views of an output are outputs too
        }
        if (is_out) {
            continue;
        }
        const bool internal = inkling2t_is_viewlike(n) ? in_range(inkling2t_base(n)) : true;
        if (!internal) {
            continue;
        }
        if (n->flags & GGML_TENSOR_FLAG_OUTPUT) {
            if (getenv("GGML_CUDA_INK_T_DEBUG")) {
                fprintf(stderr, "ink_t: closed fail output-flag node %d %s op %s\n", k, n->name, ggml_op_name(n->op));
            }
            return false;
        }
        int uses = 0;
        for (int m = k; m <= i1; ++m) {   // from k: a node may list itself as a source (ggml_cast)
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                uses += cgraph->nodes[m]->src[s] == n;
            }
        }
        if (uses != ggml_node_get_use_count(cgraph, k)) {
            if (getenv("GGML_CUDA_INK_T_DEBUG")) {
                fprintf(stderr, "ink_t: closed fail node %d %s op %s uses_in_range %d use_count %d\n", k, n->name, ggml_op_name(n->op), uses, ggml_node_get_use_count(cgraph, k));
            }
            return false;
        }
    }
    return true;
}

// copies up to 4 small contiguous float buffers into private scratch (inputs of a fused run that the allocator planned
// under one of the run's outputs: the fused kernel reads the copy, the unfused graph would have read them earlier)
struct inkling2t_stage_p {
    const float * src[4];
    float *       dst[4];
    int           n[4];
};
static __global__ void inkling2t_stage_f32(const inkling2t_stage_p p) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    for (int b = 0; b < 4; ++b) {
        for (int i = threadIdx.x; i < p.n[b]; i += blockDim.x) {
            p.dst[b][i] = p.src[b][i];
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// QKV_POST: the post-projection run of one attention visit for one token, in ONE launch instead of 8-9:
//   Q: CLIP(w) [* tau] -> FWHT-128        K: CLIP -> FWHT-128 -> SET_ROWS q8_0        V: RMS_NORM -> FWHT-64 -> SET_ROWS q8_0
// One 512-thread block per head row. Each part replays the arithmetic of the kernel it replaces (inkling2_clip_f32 with
// its 512-thread reduction, rms_norm_f32's one-warp float4 path, fwht_cuda's butterfly order, quantize_f32_q8_0_block),
// so the results are the same bits.

struct inkling2t_qkv_p {
    const float * q_in; const float * q_w; int64_t q_w_ne1; int64_t q_w_s1; const float * tau; float * q_rot;
    const float * k_in; float * k_rot; char * k_cache; const int64_t * k_idx; int64_t k_nb1;
    const float * v_in; float * v_rot; char * v_cache; const int64_t * v_idx; int64_t v_nb1;
    float qs0, qs1, qc, qlo, qhi;
    float ks0, ks1, kc, klo, khi;
    float v_eps, scale128, scale64;
    int   n_head, n_head_kv;
};

// fwht_cuda<N> for one row held in shared memory, one warp
template <int N>
static __device__ __forceinline__ void inkling2t_fwht_warp(const float * src, float * reg, const int lane, const float scale) {
    constexpr int warp_size = 32;
    constexpr int el_w = N / warp_size;
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i * warp_size + lane] * scale;
    }
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = ggml_cuda_shfl_xor(val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

// quantize_f32_q8_0_block for 32 values held one per lane (the max is exact in any order)
static __device__ __forceinline__ void inkling2t_q8_0_store(const float v, block_q8_0 * y, const int lane) {
    const float amax = warp_reduce_max(fabsf(v));
    const float d    = amax / ((1 << 7) - 1);
    const float id   = d ? 1.0f/d : 0.0f;
    if (lane == 0) {
        y->d = d;
    }
    const float x0 = v*id;
    y->qs[lane] = roundf(x0);
}

template <bool has_tau>
static __global__ void __launch_bounds__(512) inkling2t_qkv_post_f32(const inkling2t_qkv_p p) {
    ggml_cuda_pdl_lc();
    constexpr int D = 128;
    __shared__ float shared_vals[32];
    __shared__ float row[D];

    const int b    = blockIdx.x;
    const int col  = threadIdx.x;
    const int lane = col % 32;
    const int warp = col / 32;

    ggml_cuda_pdl_sync();

    if (b < p.n_head + p.n_head_kv) {
        // Q or K head: CLIP
        const bool    is_q = b < p.n_head;
        const int     h    = is_q ? b : b - p.n_head;
        const float * x    = (is_q ? p.q_in : p.k_in) + h*D;
        const float * w    = is_q ? p.q_w + (h % p.q_w_ne1)*p.q_w_s1 : nullptr;

        const int num_unroll = 8;
        float temp[num_unroll];
        float sum_temp[num_unroll] = { 0.0f };
        for (int i = col; i < D;) {
            for (int j = 0; j < num_unroll; ++j) {
                if (i < D) {
                    const float a = is_q ? __fmul_rn(x[i], w[i]) : x[i];
                    temp[j] = __fmul_rn(a, a);
                } else {
                    temp[j] = 0.0f;
                }
                i += blockDim.x;
            }
            for (int j = 0; j < num_unroll; ++j) {
                sum_temp[j] += temp[j];
            }
        }
        float sum = 0.0f;
        for (int j = 0; j < num_unroll; ++j) {
            sum += sum_temp[j];
        }
        sum = block_reduce<block_reduce_method::SUM>(sum, shared_vals);

        const float s0 = is_q ? p.qs0 : p.ks0;
        const float s1 = is_q ? p.qs1 : p.ks1;
        const float c  = is_q ? p.qc  : p.kc;
        const float lo = is_q ? p.qlo : p.klo;
        const float hi = is_q ? p.qhi : p.khi;
        const float norm   = s0*sqrtf(sum) + s1;
        const float factor = fminf(fmaxf(c / norm, lo), hi);

        if (col < D) {
            const float a = is_q ? __fmul_rn(x[col], w[col]) : x[col];
            float v = __fmul_rn(a, factor);
            if (has_tau && is_q) {
                v = __fmul_rn(v, *p.tau);
            }
            row[col] = v;
        }
        __syncthreads();

        if (warp == 0) {
            float reg[D/32];
            inkling2t_fwht_warp<D>(row, reg, lane, p.scale128);
            float * out = (is_q ? p.q_rot : p.k_rot) + h*D;
#pragma unroll
            for (int i = 0; i < D/32; ++i) {
                out[i*32 + lane] = reg[i];
            }
            if (!is_q) {
                block_q8_0 * dst = (block_q8_0 *) (p.k_cache + (*p.k_idx)*p.k_nb1) + h*(D/QK8_0);
#pragma unroll
                for (int i = 0; i < D/32; ++i) {
                    inkling2t_q8_0_store(reg[i], dst + i, lane);
                }
            }
        }
    } else {
        // V head: RMS_NORM (rms_norm_f32<32, 1, vec>) -> two FWHT-64 chunks -> q8_0
        const int     h = b - p.n_head - p.n_head_kv;
        const float * x = p.v_in + h*D;
        if (warp == 0) {
            const float4 xs = ((const float4 *) x)[lane];
            float tmp = 0.0f;
            tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
            tmp = block_reduce<block_reduce_method::SUM, 32>(tmp, shared_vals);
            const float scale = rsqrtf(tmp / D + p.v_eps);
            row[4*lane + 0] = scale*xs.x;
            row[4*lane + 1] = scale*xs.y;
            row[4*lane + 2] = scale*xs.z;
            row[4*lane + 3] = scale*xs.w;
        }
        __syncthreads();
        if (warp < 2) {
            float reg[2];
            inkling2t_fwht_warp<64>(row + warp*64, reg, lane, p.scale64);
            float * out = p.v_rot + (h*2 + warp)*64;
            out[lane]      = reg[0];
            out[32 + lane] = reg[1];
            block_q8_0 * dst = (block_q8_0 *) (p.v_cache + (*p.v_idx)*p.v_nb1) + h*(D/QK8_0) + warp*2;
            inkling2t_q8_0_store(reg[0], dst + 0, lane);
            inkling2t_q8_0_store(reg[1], dst + 1, lane);
        }
    }
}

static bool inkling2t_is_fwht(const ggml_tensor * t, int n, const ggml_tensor * src) {
    return t->op == GGML_OP_MUL_MAT && ggml_cuda_op_mul_mat_use_fwht(t) && t->ne[0] == n && t->src[1]->type == GGML_TYPE_F32 &&
           inkling2t_base(t->src[1]) == src && t->src[1]->data == src->data && ggml_is_contiguous(src);
}

static bool inkling2t_is_set_rows_q8(const ggml_tensor * t, const ggml_tensor * src, int64_t ne0) {
    return t->op == GGML_OP_SET_ROWS && t->type == GGML_TYPE_Q8_0 && t->ne[0] == ne0 && ggml_is_contiguous_rows(t) &&
           t->src[1]->type == GGML_TYPE_I64 && ggml_nelements(t->src[1]) == 1 &&
           t->src[0]->type == GGML_TYPE_F32 && t->src[0]->ne[0] == ne0 && ggml_nrows(t->src[0]) == 1 &&
           inkling2t_base(t->src[0]) == src && t->src[0]->data == src->data;
}

static bool inkling2t_debug() {
    static const bool d = getenv("GGML_CUDA_INK_T_DEBUG") != nullptr;
    return d;
}
#define INK_T_FAIL() do { if (inkling2t_debug()) { fprintf(stderr, "ink_t: match fail at line %d (node %d %s)\n", __LINE__, i, cgraph->nodes[i]->name); } return false; } while (0)

static bool inkling2t_match_qkv(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {
    if (!inkling2t_enabled(1)) {
        INK_T_FAIL();
    }
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;

    ggml_cuda_inkling2_args cq, ck;
    if (!inkling2_match_clip(cgraph, i, cq) || cq.src[1] == nullptr) {
        INK_T_FAIL();
    }
    const ggml_tensor * q_clip = cq.dst;
    if (q_clip->ne[0] != 128 || q_clip->ne[2] != 1 || q_clip->ne[3] != 1 || q_clip->ne[1] >= 2*nsm) {
        INK_T_FAIL(); // one token; the 512-thread CLIP configuration
    }
    int k = inkling2t_skip_views(cgraph, i + cq.n_nodes);
    if (k >= cgraph->n_nodes) INK_T_FAIL();

    const ggml_tensor * q_cur = q_clip;
    const ggml_tensor * tau   = nullptr;
    const ggml_tensor * n     = cgraph->nodes[k];
    if (n->op == GGML_OP_MUL && n->src[0] == q_clip && ggml_nelements(n->src[1]) == 1 && n->src[1]->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(n, q_clip) && ggml_is_contiguous(n)) {
        tau   = n->src[1];
        q_cur = n;
        k = inkling2t_skip_views(cgraph, k + 1);
        if (k >= cgraph->n_nodes) INK_T_FAIL();
        n = cgraph->nodes[k];
    }
    if (!inkling2t_is_fwht(n, 128, q_cur)) INK_T_FAIL();
    const ggml_tensor * q_rot = n;

    k = inkling2t_skip_views(cgraph, k + 1);
    if (k >= cgraph->n_nodes || !inkling2_match_clip(cgraph, k, ck) || ck.src[1] != nullptr) INK_T_FAIL();
    const ggml_tensor * k_clip = ck.dst;
    const int64_t n_head_kv = k_clip->ne[1];
    if (k_clip->ne[0] != 128 || k_clip->ne[2] != 1 || k_clip->ne[3] != 1 || n_head_kv >= 2*nsm) INK_T_FAIL();

    k = inkling2t_skip_views(cgraph, k + ck.n_nodes);
    if (k >= cgraph->n_nodes || !inkling2t_is_fwht(cgraph->nodes[k], 128, k_clip)) INK_T_FAIL();
    const ggml_tensor * k_rot = cgraph->nodes[k];

    k = inkling2t_skip_views(cgraph, k + 1);
    if (k >= cgraph->n_nodes) INK_T_FAIL();
    const ggml_tensor * v_norm = cgraph->nodes[k];
    if (v_norm->op != GGML_OP_RMS_NORM || v_norm->ne[0] != 128 || v_norm->ne[1] != n_head_kv || v_norm->ne[2] != 1 ||
            !ggml_is_contiguous(v_norm->src[0]) || v_norm->src[0]->type != GGML_TYPE_F32 ||
            ((uintptr_t) v_norm->src[0]->data % 16) != 0 || ((uintptr_t) v_norm->data % 16) != 0) INK_T_FAIL();

    k = inkling2t_skip_views(cgraph, k + 1);
    if (k >= cgraph->n_nodes || !inkling2t_is_fwht(cgraph->nodes[k], 64, v_norm)) INK_T_FAIL();
    const ggml_tensor * v_rot = cgraph->nodes[k];

    k = inkling2t_skip_views(cgraph, k + 1);
    if (k >= cgraph->n_nodes || !inkling2t_is_set_rows_q8(cgraph->nodes[k], k_rot, 128*n_head_kv)) INK_T_FAIL();
    const ggml_tensor * set_k = cgraph->nodes[k];

    k = inkling2t_skip_views(cgraph, k + 1);
    if (k >= cgraph->n_nodes || !inkling2t_is_set_rows_q8(cgraph->nodes[k], v_rot, 128*n_head_kv)) INK_T_FAIL();
    const ggml_tensor * set_v = cgraph->nodes[k];

    if (!inkling2t_closed(cgraph, i, k, { q_rot, set_k, set_v })) INK_T_FAIL();

    // An input that an output overlaps (other than the safe in-place layouts below) is staged into scratch first; outputs
    // overlapping each other are not handled.
    auto same_slot = [](const ggml_tensor * o, const ggml_tensor * in) {
        return o->data == in->data && ggml_nbytes(o) == ggml_nbytes(in);
    };
    const ggml_tensor * outs[] = { q_rot, k_rot, v_rot, set_k, set_v };
    const ggml_tensor * stage_in[4] = { cq.src[0], ck.src[0], v_norm->src[0], tau };
    const ggml_tensor * own_out[4]  = { q_rot, k_rot, v_rot, nullptr };
    int stage = 0;
    for (int b = 0; b < 4; ++b) {
        const ggml_tensor * in = stage_in[b];
        if (!in) continue;
        for (const ggml_tensor * o : outs) {
            if (inkling2_overlap(o, in) && !(o == own_out[b] && same_slot(o, in))) {
                if (!ggml_is_contiguous(in)) INK_T_FAIL();
                stage |= 1 << b;
            }
        }
    }
    if (inkling2_overlap(cq.src[1], q_rot) || inkling2_overlap(cq.src[1], k_rot) || inkling2_overlap(cq.src[1], v_rot)) INK_T_FAIL();
    if (inkling2_overlap(q_rot, k_rot) || inkling2_overlap(q_rot, v_rot) || inkling2_overlap(k_rot, v_rot)) INK_T_FAIL();
    args.n_ext = stage;

    args.type    = GGML_CUDA_INKLING2_QKV_POST;
    args.n_nodes = k - i + 1;
    args.dst     = const_cast<ggml_tensor *>(set_v);
    args.ext[0]  = cq.src[0];  args.ext[1] = cq.src[1]; args.ext[2] = tau;   args.ext[3] = q_rot;
    args.ext[4]  = ck.src[0];  args.ext[5] = k_rot;     args.ext[6] = set_k;
    args.ext[7]  = v_norm;     args.ext[8] = v_rot;     args.ext[9] = set_v;
    args.ext[10] = cq.node[0]; args.ext[11] = cq.node[1]; args.ext[12] = cq.node[2];
    args.ext[13] = ck.node[0]; args.ext[14] = ck.node[1]; args.ext[15] = ck.node[2];
    return true;
}

static void inkling2t_qkv_post(ggml_backend_cuda_context & ctx, const ggml_cuda_inkling2_args & a) {
    inkling2t_qkv_p p;
    const ggml_tensor * qw = a.ext[1];
    p.q_in    = (const float *) a.ext[0]->data;
    p.q_w     = (const float *) qw->data;
    p.q_w_ne1 = qw->ne[1];
    p.q_w_s1  = qw->nb[1]/sizeof(float);
    p.tau     = a.ext[2] ? (const float *) a.ext[2]->data : nullptr;
    p.q_rot   = (float *) a.ext[3]->data;
    p.k_in    = (const float *) a.ext[4]->data;
    p.k_rot   = (float *) a.ext[5]->data;
    p.k_cache = (char *) a.ext[6]->data;
    p.k_idx   = (const int64_t *) a.ext[6]->src[1]->data;
    p.k_nb1   = a.ext[6]->nb[1];
    p.v_in    = (const float *) a.ext[7]->src[0]->data;
    p.v_rot   = (float *) a.ext[8]->data;
    p.v_cache = (char *) a.ext[9]->data;
    p.v_idx   = (const int64_t *) a.ext[9]->src[1]->data;
    p.v_nb1   = a.ext[9]->nb[1];
    p.qs0 = ggml_get_op_params_f32(a.ext[10], 0); p.qs1 = ggml_get_op_params_f32(a.ext[10], 1);
    p.qc  = ggml_get_op_params_f32(a.ext[11], 0);
    p.qlo = ggml_get_op_params_f32(a.ext[12], 0); p.qhi = ggml_get_op_params_f32(a.ext[12], 1);
    p.ks0 = ggml_get_op_params_f32(a.ext[13], 0); p.ks1 = ggml_get_op_params_f32(a.ext[13], 1);
    p.kc  = ggml_get_op_params_f32(a.ext[14], 0);
    p.klo = ggml_get_op_params_f32(a.ext[15], 0); p.khi = ggml_get_op_params_f32(a.ext[15], 1);
    float eps; memcpy(&eps, a.ext[7]->op_params, sizeof(float));
    p.v_eps    = eps;
    p.scale128 = 1 / sqrtf(128);
    p.scale64  = 1 / sqrtf(64);
    p.n_head    = (int) a.ext[3]->ne[1];
    p.n_head_kv = (int) a.ext[5]->ne[1];

    // stage the inputs an output would overwrite (args.n_ext: bit 0 q_in, 1 k_in, 2 v_in, 3 tau)
    ggml_cuda_pool_alloc<float> scratch(ctx.pool());
    if (a.n_ext) {
        const ggml_tensor * in[4] = { a.ext[0], a.ext[4], a.ext[7]->src[0], a.ext[2] };
        int64_t total = 0;
        for (int b = 0; b < 4; ++b) {
            if (a.n_ext & (1 << b)) total += ggml_nelements(in[b]);
        }
        float * sc = scratch.alloc(total);
        inkling2t_stage_p sp = {};
        const float ** ptrs[4] = { &p.q_in, &p.k_in, &p.v_in, &p.tau };
        int nb = 0;
        for (int b = 0; b < 4; ++b) {
            if (!(a.n_ext & (1 << b))) continue;
            sp.src[nb] = (const float *) in[b]->data;
            sp.dst[nb] = sc;
            sp.n[nb]   = (int) ggml_nelements(in[b]);
            *ptrs[b]   = sc;
            sc += sp.n[nb];
            ++nb;
        }
        const ggml_cuda_kernel_launch_params sparams(dim3(1, 1, 1), dim3(512, 1, 1), 0, ctx.stream());
        ggml_cuda_kernel_launch(inkling2t_stage_f32, sparams, sp);
    }
    const dim3 block_nums(p.n_head + 2*p.n_head_kv, 1, 1);
    const dim3 block_dims(512, 1, 1);
    const ggml_cuda_kernel_launch_params params(block_nums, block_dims, 0, ctx.stream());
    if (p.tau) {
        ggml_cuda_kernel_launch(inkling2t_qkv_post_f32<true>, params, p);
    } else {
        ggml_cuda_kernel_launch(inkling2t_qkv_post_f32<false>, params, p);
    }
}

