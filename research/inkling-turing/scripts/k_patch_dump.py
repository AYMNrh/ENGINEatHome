p = '/home/marvin/llm/experiments/inkling-kernels/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu'
s = open(p).read()
if 'GGML_CUDA_DUMP_GRAPH' in s:
    print('already'); raise SystemExit
# helper before the evaluate function
anchor = 'static void ggml_cuda_graph_evaluate_and_capture('
helper = r'''// debug aid (kernel work): GGML_CUDA_DUMP_GRAPH=<file> writes the nodes of the 3rd evaluated graph with their fusion status
static FILE * ggml_cuda_dump_graph_file(const ggml_cgraph * cgraph) {
    static const char * path = getenv("GGML_CUDA_DUMP_GRAPH");
    static int calls = 0;
    if (!path) return nullptr;
    if (++calls != 3) return nullptr;
    FILE * f = fopen(path, "w");
    if (f) fprintf(f, "n_nodes %d\n", cgraph->n_nodes);
    return f;
}
static void ggml_cuda_dump_node(FILE * f, int i, const ggml_tensor * t, const char * tag) {
    fprintf(f, "%5d %-4s %-14s %-28.28s %-6s [%lld,%lld,%lld,%lld]", i, tag, ggml_op_name(t->op), t->name, ggml_type_name(t->type),
        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        if (t->src[j]) fprintf(f, " | %s:%s:%s", ggml_op_name(t->src[j]->op), t->src[j]->name, ggml_type_name(t->src[j]->type));
    }
    fprintf(f, "\n");
    fflush(f);
}

'''
assert s.count(anchor) == 1
s = s.replace(anchor, helper + anchor, 1)
# open the dump file at the start of the evaluate function body
i0 = s.index(anchor + 'ggml_backend_cuda_context * cuda_ctx')
i1 = s.index('{', i0) + 1
s = s[:i1] + '\n    FILE * dump_f = ggml_cuda_dump_graph_file(cgraph);\n' + s[i1:]
a = '''                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
'''
b = '''                if (ggml_cuda_is_view_or_noop(node)) {
                    if (dump_f) ggml_cuda_dump_node(dump_f, i, node, "view");
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
                if (dump_f) {
                    ggml_cuda_dump_node(dump_f, i, node, nodes_to_skip ? "FUSE" : "K");
                    for (int k = 1; k <= nodes_to_skip; ++k) ggml_cuda_dump_node(dump_f, i + k, cgraph->nodes[i + k], " +");
                }
'''
assert s.count(a) == 1, 'loop anchor'
s = s.replace(a, b, 1)
open(p, 'w').write(s)
print('patched dump')
