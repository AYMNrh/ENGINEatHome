# Same-input q8_1 quantization cache for the mat-vec path (turing-iq3-opt).
D = '~/llm/experiments/inkling-kernels/llama.cpp/ggml/src/ggml-cuda/'

# 1) context members + declaration
p = D + 'common.cuh'
s = open(p).read()
if 'q8_1_cache' not in s:
    anchor = 'struct ggml_backend_cuda_context {'
    assert s.count(anchor) == 1
    s = s.replace(anchor, '''// turing-iq3-opt: q8_1 copies of mat-vec inputs, reused by later mat-vecs of the same graph evaluation that read the
// same tensor (the unfused graph quantizes the same activation once per consumer)
struct ggml_cuda_q8_1_cache_entry {
    const struct ggml_tensor * t    = nullptr;
    const void *               data = nullptr;
    int64_t                    ne[4] = {0, 0, 0, 0};
    size_t                     nb[4] = {0, 0, 0, 0};
    uint32_t                   gen  = 0;
    char *                     buf  = nullptr;
};
#define GGML_CUDA_Q8_1_CACHE_SLOTS 8
#define GGML_CUDA_Q8_1_CACHE_BYTES (64*1024)

''' + anchor, 1)
    # add members right after the opening brace of the struct
    i = s.index(anchor) + len(anchor)
    s = s[:i] + '''
    ggml_cuda_q8_1_cache_entry q8_1_cache[GGML_CUDA_Q8_1_CACHE_SLOTS];
    int      q8_1_cache_next = 0;
    uint32_t q8_1_cache_gen  = 1;
''' + s[i:]
    s += '''
// returns the slot buffer for the q8_1 copy of src1 (hit = already quantized in this graph evaluation),
// or nullptr if the cache cannot be used (size, concurrency, buffers not allocated during a capture)
char * ggml_cuda_q8_1_cache(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, size_t size, bool & hit);
'''
    open(p, 'w').write(s)

# 2) implementation, generation bump per graph evaluation, free in the destructor
p = D + 'ggml-cuda.cu'
s = open(p).read()
if 'ggml_cuda_q8_1_cache(' not in s:
    impl = '''
char * ggml_cuda_q8_1_cache(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, size_t size, bool & hit) {
    static const bool disabled = [] {
        const char * env = getenv("GGML_CUDA_INK_T_DISABLE");
        return env != nullptr && (atoi(env) & 8) != 0;
    }();
    hit = false;
    if (disabled || size > GGML_CUDA_Q8_1_CACHE_BYTES || !ctx.stream_context().concurrent_events.empty()) {
        return nullptr;
    }
    for (auto & e : ctx.q8_1_cache) {
        if (e.buf && e.gen == ctx.q8_1_cache_gen && e.t == src1 && e.data == src1->data &&
                memcmp(e.ne, src1->ne, sizeof(e.ne)) == 0 && memcmp(e.nb, src1->nb, sizeof(e.nb)) == 0) {
            hit = true;
            return e.buf;
        }
    }
    auto & e = ctx.q8_1_cache[ctx.q8_1_cache_next];
    if (!e.buf) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &st));
        if (st != cudaStreamCaptureStatusNone) {
            return nullptr; // no allocation inside a capture: fall back to the pool for this call
        }
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc((void **) &e.buf, GGML_CUDA_Q8_1_CACHE_BYTES));
    }
    ctx.q8_1_cache_next = (ctx.q8_1_cache_next + 1) % GGML_CUDA_Q8_1_CACHE_SLOTS;
    e.t    = src1;
    e.data = src1->data;
    memcpy(e.ne, src1->ne, sizeof(e.ne));
    memcpy(e.nb, src1->nb, sizeof(e.nb));
    e.gen  = ctx.q8_1_cache_gen;
    return e.buf;
}

'''
    anchor = 'static void ggml_cuda_graph_evaluate_and_capture('
    s = s.replace('// debug aid (kernel work): GGML_CUDA_DUMP_GRAPH', impl + '// debug aid (kernel work): GGML_CUDA_DUMP_GRAPH', 1)
    assert 'char * ggml_cuda_q8_1_cache(' in s
    # bump the generation at the start of every graph evaluation (also invalidates on a new graph)
    s = s.replace('\n    FILE * dump_f = ggml_cuda_dump_graph_file(cgraph);\n',
                  '\n    FILE * dump_f = ggml_cuda_dump_graph_file(cgraph);\n    cuda_ctx->q8_1_cache_gen++;\n', 1)
    assert 'q8_1_cache_gen++' in s
    # free the slot buffers with the context
    d = 'ggml_backend_cuda_context::~ggml_backend_cuda_context() {'
    assert s.count(d) == 1, 'destructor'
    s = s.replace(d, d + '\n    for (auto & e : q8_1_cache) {\n        if (e.buf) {\n            CUDA_CHECK(cudaFree(e.buf));\n            e.buf = nullptr;\n        }\n    }', 1)
    open(p, 'w').write(s)

# 3) use it in the mat-vec path
p = D + 'mmvq.cu'
s = open(p).read()
if 'ggml_cuda_q8_1_cache(' not in s:
    old = '''    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, src1_q8_1.get(), ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }
'''
    new = '''    const size_t q8_1_size = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    bool q8_1_hit = false;
    char * q8_1_slot = ggml_cuda_q8_1_cache(ctx, src1, q8_1_size, q8_1_hit);
    ggml_cuda_pool_alloc<char> src1_q8_1_pool(ctx.pool());
    char * src1_q8_1_d = q8_1_slot ? q8_1_slot : src1_q8_1_pool.alloc(q8_1_size);
    if (!q8_1_hit) {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, src1_q8_1_d, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }
'''
    assert s.count(old) == 1, 'mmvq quantize block'
    s = s.replace(old, new)
    old2 = '        src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,'
    assert s.count(old2) == 1
    s = s.replace(old2, '        src0->data, src0->type, src1_q8_1_d, ids_d, fusion_local, dst_d, ne00,')
    open(p, 'w').write(s)
print('patched q8_1 cache')
