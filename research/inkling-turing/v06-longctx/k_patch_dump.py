p = "llama.cpp-diag/src/llama-context.cpp"
s = open(p).read()
NL = chr(10)

helper = r'''
// ---- DIAGNOSTIC (not part of any validated build): LLAMA_DUMP_GRAPH_DIR=<dir> writes every graph the context builds (reserve and real) with
// shapes, sources, flags, views and (after allocation) the buffer offset of every tensor; LLAMA_DUMP_GRAPH_MAX=<n> exits after n real graphs
#include <unordered_map>
#include <cstdlib>
static void ink_dump_graph(const char * tag, ggml_cgraph * gf, bool allocated) {
    const char * dir = getenv("LLAMA_DUMP_GRAPH_DIR");
    if (!dir) {
        return;
    }
    static int n_dump = 0;
    char path[512];
    snprintf(path, sizeof(path), "%s/graph_%03d_%s.tsv", dir, n_dump++, tag);
    FILE * f = fopen(path, "w");
    if (!f) {
        return;
    }
    std::unordered_map<const ggml_tensor *, int> idx;
    const int n = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n; ++i) {
        idx[ggml_graph_node(gf, i)] = i;
    }
    fprintf(f, "# idx\top\tname\ttype\tne0\tne1\tne2\tne3\tflags\tsrcs\tview_src\tview_offs\tbuf\toff\tnbytes\n");
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * t = ggml_graph_node(gf, i);
        std::string srcs;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * s = t->src[j];
            if (!s) {
                srcs += "-|";
                continue;
            }
            auto it = idx.find(s);
            if (it != idx.end()) {
                srcs += std::to_string(it->second) + "|";
            } else {
                srcs += std::string("L:") + s->name + "|";
            }
        }
        long off = -1;
        const char * bn = "-";
        if (allocated && t->buffer && t->data) {
            off = (long) ((const char *) t->data - (const char *) ggml_backend_buffer_get_base(t->buffer));
            bn = ggml_backend_buffer_name(t->buffer);
        }
        fprintf(f, "%d\t%s\t%s\t%s\t%lld\t%lld\t%lld\t%lld\t%d\t%s\t%s\t%zu\t%s\t%ld\t%zu\n", i, ggml_op_name(t->op), t->name, ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], (int) t->flags, srcs.c_str(),
            t->view_src ? t->view_src->name : "-", (size_t) t->view_offs, bn, off, ggml_nbytes(t));
    }
    fclose(f);
    if (allocated) {
        static int n_real = 0;
        const char * mx = getenv("LLAMA_DUMP_GRAPH_MAX");
        if (mx && ++n_real >= atoi(mx)) {
            fprintf(stderr, "graph dump done\n");
            exit(0);
        }
    }
}

'''
marker = "llm_graph_result * llama_context::process_ubatch("
assert s.count(marker) == 1
s = s.replace(marker, helper + marker)

old = "        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {"
assert s.count(old) == 1
i = s.index(old)
j = s.index("        }" + NL, i) + len("        }" + NL)
s = s[:j] + '        { char tag[64]; snprintf(tag, sizeof(tag), "real_%d", (int) ubatch.n_tokens); ink_dump_graph(tag, gf, true); }' + NL + s[j:]

old2 = "    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {"
assert s.count(old2) == 1
open(p, "w").write(s)
print("dump patch applied (real graphs); reserve dumped separately")
s = open(p).read()
old3 = "    this->n_input_tensors = llama_graph_n_input_tensors(gf);\n    this->n_outputs = save_n_outputs;\n"
assert s.count(old3) == 1
s = s.replace(old3, old3 + '    { char tag[64]; snprintf(tag, sizeof(tag), "reserve_%d_%d", (int) n_tokens, (int) n_seqs); ink_dump_graph(tag, gf, false); }\n')
open(p, "w").write(s)
print("reserve dump applied")
