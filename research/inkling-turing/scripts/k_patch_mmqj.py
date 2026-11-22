p="llama.cpp/ggml/src/ggml-cuda/mmq.cu"
s=open(p).read()
old="""        J_best = ggml_cuda_mmq_get_J_best(src0->type, ncols_opt, fallback, cc, smpbo, prec_src1, &nthreads_best);
    }"""
new="""        J_best = ggml_cuda_mmq_get_J_best(src0->type, ncols_opt, fallback, cc, smpbo, prec_src1, &nthreads_best);
        // inkling-turing: experimental override of the MoE tile width (GGML_CUDA_INK_T_MMQJ=<J>), default off
        static const int ink_t_mmqj = getenv("GGML_CUDA_INK_T_MMQJ") ? atoi(getenv("GGML_CUDA_INK_T_MMQJ")) : 0;
        if (ids && ink_t_mmqj > 0) {
            const ggml_cuda_mmq_config cfg = ggml_cuda_mmq_get_config(src0->type, ink_t_mmqj, fallback, cc, prec_src1);
            if (cfg.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(cfg, cc) <= smpbo) {
                J_best = ink_t_mmqj;
                nthreads_best = cfg.nthreads;
            }
        }
    }"""
assert s.count(old)==1
s=s.replace(old,new)
open(p,"w").write(s)
print("ok")
