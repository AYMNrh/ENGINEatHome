import sys
D = '/home/marvin/llm/experiments/inkling-kernels/'
H = D + 'llama.cpp/ggml/src/ggml-cuda/inkling2.cuh'
C = D + 'llama.cpp/ggml/src/ggml-cuda/inkling2.cu'
inc = open(D + 'ink_t_qkv.inc').read().replace('ggml_is_contiguous_rows(t)', 't->nb[0] == ggml_type_size(GGML_TYPE_Q8_0)')

h = open(H).read()
if 'GGML_CUDA_INKLING2_QKV_POST' not in h:
    h = h.replace('    GGML_CUDA_INKLING2_CTRL_TAIL,  // TANH MUL SUB TANH MUL ADD                         : controller output, large batches\n',
                  '    GGML_CUDA_INKLING2_CTRL_TAIL,  // TANH MUL SUB TANH MUL ADD                         : controller output, large batches\n'
                  '    GGML_CUDA_INKLING2_QKV_POST,   // Q CLIP [*tau] FWHT, K CLIP FWHT SET_ROWS, V RMS_NORM FWHT SET_ROWS (decode, turing-iq3-opt)\n'
                  '    GGML_CUDA_INKLING2_RELBIAS_T,  // PAD MUL_MAT + RELBIAS chain + FILL                (decode, turing-iq3-opt)\n'
                  '    GGML_CUDA_INKLING2_ROUTE_TAIL, // SUM_ROWS SCALE DIV SCALE after the top-k           (decode, turing-iq3-opt)\n')
    h = h.replace('    const ggml_tensor * node[6] = {}; // further tensors of the chain: op parameters, views, intermediates\n',
                  '    const ggml_tensor * node[6] = {}; // further tensors of the chain: op parameters, views, intermediates\n'
                  '    const ggml_tensor * ext[20] = {}; // the tensors of the larger turing-iq3-opt fusions\n')
    assert 'QKV_POST' in h and 'ext[20]' in h
    open(H, 'w').write(h)

c = open(C).read()
if 'inkling2t_match_qkv' not in c:
    if '#include "fwht.cuh"' not in c:
        c = c.replace('#include "mmvf.cuh"\n', '#include "mmvf.cuh"\n#include "fwht.cuh"\n', 1)
    anchor = 'bool ggml_cuda_inkling2_match(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {'
    assert c.count(anchor) == 1
    c = c.replace(anchor, inc + '\n' + anchor, 1)
    old = '            return inkling2_match_clip(cgraph, i, args) || inkling2_match_situ(cgraph, i, args);'
    assert c.count(old) == 1
    c = c.replace(old, '            return inkling2t_match_qkv(cgraph, i, args) || (args = {}, inkling2_match_clip(cgraph, i, args)) ||\n'
                       '                   inkling2_match_situ(cgraph, i, args);')
    old = '        case GGML_CUDA_INKLING2_RELBIAS:    inkling2_relbias(ctx, args);    break;'
    c = c.replace(old, old + '\n        case GGML_CUDA_INKLING2_QKV_POST:   inkling2t_qkv_post(ctx, args);  break;')
    open(C, 'w').write(c)
print('patched qkv')
