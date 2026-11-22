p="llama.cpp/ggml/src/ggml-cuda/mmq.cu"
s=open(p).read()
old="""            ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
"""
new="""            ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
            // inkling-turing: routing is uneven, so size the tile for 1.5x the average expert load; a second
            // mostly-empty column tile re-dequantizes the whole expert slice, which costs more than idle MMA columns.
            // (measured on the 2080 Ti at -ub 512: J=48 vs J=32 +2..3% prefill, bit-exact)
            ncols_opt = (3*ncols_opt + 1) / 2;
"""
assert s.count(old)==1
s=s.replace(old,new); open(p,"w").write(s); print("ok")
