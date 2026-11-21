D = '/home/marvin/llm/experiments/inkling-kernels/'
H = D + 'llama.cpp/ggml/src/ggml-cuda/inkling2.cuh'
C = D + 'llama.cpp/ggml/src/ggml-cuda/inkling2.cu'
h = open(H).read()
if 'GGML_CUDA_INKLING2_CTRL_T,' not in h:
    h = h.replace('    GGML_CUDA_INKLING2_ROUTE_TAIL,', '    GGML_CUDA_INKLING2_CTRL_T,     // up to 3 loop controllers (CTRL_DOWN + CTRL_OUT each), decode (turing-iq3-opt)\n    GGML_CUDA_INKLING2_ROUTE_TAIL,')
    h = h.replace('    const ggml_tensor * ext[20] = {}; // the tensors of the larger turing-iq3-opt fusions\n',
                  '    const ggml_tensor * ext[20] = {}; // the tensors of the larger turing-iq3-opt fusions\n'
                  '    const ggml_tensor * ext2[16] = {};\n    int n_ext = 0;\n')
    assert 'CTRL_T,' in h and 'ext2' in h
    open(H, 'w').write(h)
c = open(C).read()
inc = open(D + 'ink_t_qkv.inc').read().replace('ggml_is_contiguous_rows(t)', 't->nb[0] == ggml_type_size(GGML_TYPE_Q8_0)')
inc += open(D + 'ink_t_rb.inc').read() + open(D + 'ink_t_ctrl.inc').read()
a = c.index('// =====================================================================================================================\n// Turing kernel work')
b = c.index('bool ggml_cuda_inkling2_match(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {')
c = c[:a] + inc + '\n' + c[b:]
old = '            return inkling2_match_mix(cgraph, i, args) || inkling2_match_ctrl_down(cgraph, i, args) ||'
if 'inkling2t_match_ctrl(cgraph, i, args)' not in c:
    assert c.count(old) == 1
    c = c.replace(old, '            return inkling2t_match_ctrl(cgraph, i, args) || (args = {}, inkling2_match_mix(cgraph, i, args)) ||\n                   inkling2_match_ctrl_down(cgraph, i, args) ||')
    old2 = '        case GGML_CUDA_INKLING2_ROUTE_TAIL: inkling2t_route_tail(ctx, args); break;'
    assert c.count(old2) == 1
    c = c.replace(old2, old2 + '\n        case GGML_CUDA_INKLING2_CTRL_T:     inkling2t_ctrl(ctx, args);       break;')
open(C, 'w').write(c)
print('patched ctrl')
