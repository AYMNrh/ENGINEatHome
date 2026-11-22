p = "llama.cpp/src/llama-graph.cpp"
s = open(p).read()
old = "        ggml_set_output(rel_s->view_src ? rel_s->view_src : rel_s);\n"
assert s.count(old) == 1
new = old + "        ggml_set_output(rel_cell); // graph inputs are freed after their last consumer in the graph, which is the descriptor, not flash attention\n        ggml_set_output(rel_tok);\n"
open(p, "w").write(s.replace(old, new))
print("fixed5")
