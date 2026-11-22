import re
p = "llama.cpp/ggml/src/ggml-cuda/mmq-vec-dot.cuh"
L = open(p).read().split("\n")
assert not any("ink_t_i2f" in x for x in L)
helper = r"""
// inkling-turing: exact int32 -> float for MMA partial sums. Turing issues I2F at 1/4 of the FP32 rate and the
// MMQ epilogue converts every MMA result, which made I2F the top stall of the expert GEMMs. For |x| < 2^22
// (an int8 x int8 MMA over k <= 128 stays far below that), the float 1.5*2^23 + x is exact and holds x in its
// mantissa, so an integer add plus one float subtraction (both full rate) gives the same float as I2F, bit for bit.
static __device__ __forceinline__ float ink_t_i2f(const int x) {
#if defined(TURING_MMA_AVAILABLE) && !defined(GGML_CUDA_INK_T_NO_FAST_I2F)
    return __int_as_float(x + 0x4B400000) - 12582912.0f;
#else
    return (float) x;
#endif
}
"""
i = L.index("using namespace ggml_cuda_mma;")
hl = helper.split("\n")
L[i+1:i+1] = hl
off = len(hl)
for n in [272, 432, 599, 834, 836, 1151]:
    k = n - 1 + off
    s = L[k]
    s2 = re.sub(r"\b(C|Cd|Cm)(\[\d\])?\.x\[l\]", lambda m: "ink_t_i2f(%s)" % m.group(0), s)
    assert s2 != s, (n, s)
    L[k] = s2
    print(n, s2.strip()[:160])
open(p, "w").write("\n".join(L))
