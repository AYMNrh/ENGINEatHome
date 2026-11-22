p = '~/llm/experiments/inkling-kernels/llama.cpp/ggml/src/ggml-cuda/mmvq.cu'
s = open(p).read()
s = s.replace('GGML_CUDA_INK_T_MOE1: 0 = stock, 1 = down-bank types (default), 2 = all types.',
              'GGML_CUDA_INK_T_MOE1: 0 = stock, 1 = down-bank types, 2 = all types (default; the IQ2_S/IQ3_XXS gate/up\n    // banks gain the most: tg128 61.9 -> 64.2 on an RTX 2080 Ti).')
s = s.replace("        return env ? atoi(env) : 1;\n    }();\n    const bool moe1_type", "        return env ? atoi(env) : 2;\n    }();\n    const bool moe1_type")
assert 'atoi(env) : 2;\n    }();\n    const bool moe1_type' in s
open(p, 'w').write(s)
print('default policy 2')
