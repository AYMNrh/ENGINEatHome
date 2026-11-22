R = "llama.cpp/ggml/src/ggml-cuda/"
NL = chr(10)
def edit(path, old, new):
    s = open(R + path).read(); assert s.count(old) == 1, (path, old[:60], s.count(old)); open(R + path, "w").write(s.replace(old, new))
edit("fattn-mma-f16.cuh", "    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks." + NL,
     "    fattn_rel_head rel = {};" + NL + "    if (rel_params.tab && ncols2 == 1) {" + NL +
     "        rel.tab     = rel_params.tab  + zt_Q*rel_params.tab_s1;" + NL + "        rel.sink    = rel_params.sink + zt_Q*rel_params.sink_s1;" + NL +
     "        rel.cell    = rel_params.cell;" + NL + "        rel.tok     = rel_params.tok;" + NL + "        rel.tab_s2  = rel_params.tab_s2;" + NL +
     "        rel.sink_s2 = rel_params.sink_s2;" + NL + "        rel.n_rows  = rel_params.n_rows;" + NL + "    }" + NL + NL +
     "    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks." + NL)
edit("fattn-vec-gqa.cu", "(int32_t) mask->ne[1], (int32_t) mask->ne[2], (int32_t) mask->ne[3], (int32_t) mask->nb[1], (int64_t) mask->nb[2], (int64_t) mask->nb[3]);",
     "(int32_t) mask->ne[1], (int32_t) mask->ne[2], (int32_t) mask->ne[3], (int32_t) mask->nb[1], (int64_t) mask->nb[2], (int64_t) mask->nb[3],\n        fattn_rel_params{});")
print("fixed")
