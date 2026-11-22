p = "llama.cpp/src/llama-graph.cpp"
s = open(p).read()
old = "        ggml_tensor * desc = ggml_inkling2_relbias_desc(ctx0, rel_tab, rel_s, rel_cell, rel_tok, attn_mask.raw);\n"
assert s.count(old) == 1
new = ("        // flash attention reads the table and the sink logits through the descriptor, which the allocator cannot see: keep them allocated\n"
       "        // (graph outputs are never freed or reused); the cell and token positions are graph inputs, kept anyway\n"
       "        ggml_set_output(rel_tab);\n"
       "        ggml_set_output(rel_s->view_src ? rel_s->view_src : rel_s);\n") + old
open(p, "w").write(s.replace(old, new))
print("fixed4")
