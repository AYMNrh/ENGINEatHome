p = "llama.cpp/ggml/src/ggml-cuda/fattn.cu"
s = open(p).read()
old = "    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);\n    const int gqa_ratio = Q->ne[2] / K->ne[2];\n\n    // On Volta the GQA optimizations"
assert s.count(old) == 1
new = "    // the relative-position bias is computed per Q head inside the kernel: one head per tile (no GQA packing)\n    if (dst->src[6]) {\n        use_gqa_opt = false;\n    }\n\n" + old
open(p, "w").write(s.replace(old, new))
print("fixed3")
