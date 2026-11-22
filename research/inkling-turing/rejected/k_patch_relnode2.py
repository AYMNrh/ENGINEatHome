R = "llama.cpp/"
def edit(path, old, new):
    s = open(R + path).read()
    assert s.count(old) == 1, (path, old[:50], s.count(old))
    open(R + path, "w").write(s.replace(old, new))

# decode fusion RELBIAS_T accepts the one-node form of the mask as well as the chain
edit("ggml/src/ggml-cuda/inkling2.cu",
     "static bool inkling2t_match_relbias_t(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {\n",
     '''// the mask as one graph node (ggml_inkling2_relbias): the same inputs and output as the matched chain, one node
static bool inkling2_match_relbias_node(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {
    const ggml_tensor * n = cgraph->nodes[i];
    if (!ggml_is_inkling2_relbias(n) || !ggml_cuda_inkling2_relbias_node_supported(n)) {
        return false;
    }
    args.type    = GGML_CUDA_INKLING2_RELBIAS;
    args.n_nodes = 1;
    args.dst     = const_cast<ggml_tensor *>(n);
    for (int j = 0; j < 4; ++j) {
        args.src[j] = n->src[j];
    }
    return true;
}

static bool inkling2t_match_relbias_t(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {
''')
edit("ggml/src/ggml-cuda/inkling2.cu",
     "if (k >= cgraph->n_nodes || !inkling2_match_relbias(cgraph, k, rb) || rb.src[0] != mm) INK_T_FAIL();",
     "if (k >= cgraph->n_nodes || !(inkling2_match_relbias(cgraph, k, rb) || inkling2_match_relbias_node(cgraph, k, rb)) || rb.src[0] != mm) INK_T_FAIL();")
s = open(R + "ggml/src/ggml-cuda/inkling2.cu").read()
first_inc = s.index("#include")
end_inc = s.index("\n", s.rindex("#include", 0, s.index("static")))
s = s[:end_inc + 1] + '#include "ggml-inkling2.h"\n' + s[end_inc + 1:]
open(R + "ggml/src/ggml-cuda/inkling2.cu", "w").write(s)

# the model: the node for every batch size, so that the prefill and the decode graphs have the same topology
edit("src/models/inkling2.cpp",
     "if (relbias_node && n_tokens > 1 && n_stream == 1) {",
     "if (relbias_node && n_stream == 1) {")
edit("src/models/inkling2.cpp",
     "        // several tokens: the mask of each head is one node (ggml_inkling2_relbias), so that no F32 [n_head, n_kv, n_tokens]\n        // intermediates are planned by the allocator (at 256K they are most of the compute buffer); one token keeps the\n        // chain, which the decode fusion takes. LLAMA_INK_RELBIAS_NODE=0 keeps the chain for all batch sizes.\n",
     "        // the mask of each head is one node (ggml_inkling2_relbias), so that no F32 [n_head, n_kv, n_tokens] intermediates are\n        // planned by the allocator (at 256K they are most of the compute buffer). All batch sizes: the prefill and the decode graph\n        // keep the same topology (a different one makes the scheduler re-plan the buffer at every switch); the decode fusion\n        // RELBIAS_T takes the node as it took the chain. LLAMA_INK_RELBIAS_NODE=0 keeps the chain.\n")
print("patched2")
