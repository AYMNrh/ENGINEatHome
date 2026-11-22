#!/usr/bin/env python3
"""inkling-turing: bit-exact IQ3_XXS / IQ2_S MMQ loader rewrite.
- sign application: (g ^ m) + t with t = per-byte 0/1 from a 4-bit nibble (no byte carry: IQ grids have no zero bytes)
- optional grid LUT in shared memory (GGML_CUDA_INK_T_IQ_LUT bit 1 = IQ3_XXS 1 KB, bit 2 = IQ2_S 8 KB)
"""
import sys
D = 'llama.cpp/ggml/src/ggml-cuda/'
lut = sys.argv[1] if len(sys.argv) > 1 else '1'

# ---------------------------------------------------------------- mmq-load-tiles.cuh
p = D + 'mmq-load-tiles.cuh'
s = open(p).read()
assert 'ink_t_apply_signs4' not in s

helpers = r'''// ---- inkling-turing: IQ loader helpers -------------------------------------------------------
#ifndef GGML_CUDA_INK_T_IQ_LUT
#define GGML_CUDA_INK_T_IQ_LUT %s // bit 1: IQ3_XXS grid in shared memory, bit 2: IQ2_S grid in shared memory
#endif

// Conditionally negate the 4 int8 lanes of g whose bit is set in nib4 (0..15), bit-exact with
// __vsub4(g ^ __vcmpne4(sel, 0), __vcmpne4(sel, 0)): t = per-byte 0/1, (g ^ 0xFF*t) + t.
// The IQ grids contain no zero bytes, so ~g + 1 never carries into the next byte and one 32-bit add is exact.
static __device__ __forceinline__ int ink_t_apply_signs4(const uint32_t g, const uint32_t nib4) {
    const uint32_t t = (nib4 * 0x00204081u) & 0x01010101u;
    return (int) ((g ^ (t * 0xFFu)) + t);
}

static constexpr __host__ __device__ int ggml_cuda_ink_t_lut_bytes(const ggml_type type) {
    return (type == GGML_TYPE_IQ3_XXS && (GGML_CUDA_INK_T_IQ_LUT & 1)) ? 256*4 :
           (type == GGML_TYPE_IQ2_S   && (GGML_CUDA_INK_T_IQ_LUT & 2)) ? 1024*8 : 0;
}

template <ggml_type type> static __device__ __forceinline__ uint32_t * ink_t_grid_smem() {
    if constexpr (type == GGML_TYPE_IQ3_XXS) {
        __shared__ uint32_t ink_t_tab_iq3xxs[256];
        return ink_t_tab_iq3xxs;
    } else {
        __shared__ uint32_t ink_t_tab_iq2s[2048];
        return ink_t_tab_iq2s;
    }
}

// Called once per tile before the K loop (followed by __syncthreads()).
template <ggml_type type> static __device__ __forceinline__ void ink_t_grid_fill(const int tid, const int nthreads) {
    if constexpr (ggml_cuda_ink_t_lut_bytes(type) > 0) {
        uint32_t * tab = ink_t_grid_smem<type>();
        const uint32_t * src = type == GGML_TYPE_IQ3_XXS ? (const uint32_t *) iq3xxs_grid : (const uint32_t *) iq2s_grid;
        for (int i = tid; i < ggml_cuda_ink_t_lut_bytes(type)/4; i += nthreads) {
            tab[i] = src[i];
        }
    }
}
// -----------------------------------------------------------------------------------------------

'''
anchor = 'template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_s('
assert s.count(anchor) == 1
s = s.replace(anchor, helpers % lut + anchor)

# IQ2_S body (MMA + dp4a share the same sign code; replace the occurrence inside the iq2_s loader only)
i0 = s.index(anchor); i1 = s.index('template <ggml_type type', i0 + 10)
body = s[i0:i1]
old2 = '''            const int * grid_pos = (const int *)(iq2s_grid + (qs[l] | ((qh << (8-2*l)) & 0x300)));

            const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);
'''
new2 = '''            const uint2 * grid_tab = (GGML_CUDA_INK_T_IQ_LUT & 2) ? (const uint2 *) ink_t_grid_smem<GGML_TYPE_IQ2_S>() : (const uint2 *) iq2s_grid;
            const uint2 grid_pos = grid_tab[qs[l] | ((qh << (8-2*l)) & 0x300)];
            const uint32_t sgn = signs_packed_8[l];

            const int grid_l = ink_t_apply_signs4(grid_pos.x, sgn & 0x0F);
            const int grid_h = ink_t_apply_signs4(grid_pos.y, sgn >> 4);
'''
assert body.count(old2) == 1, body.count(old2)
body = body.replace(old2, new2)
s = s[:i0] + body + s[i1:]

# IQ3_XXS body
a3 = 'static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_xxs('
i0 = s.index(a3); i1 = s.index('template <ggml_type type', i0 + 10)
body = s[i0:i1]
old3 = '''            const int2 grid_pos = make_int2(iq3xxs_grid[q3[2*l+0]], iq3xxs_grid[q3[2*l+1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7*l));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
'''
new3 = '''            const uint32_t * grid_tab = (GGML_CUDA_INK_T_IQ_LUT & 1) ? ink_t_grid_smem<GGML_TYPE_IQ3_XXS>() : iq3xxs_grid;
            const uint2 grid_pos = make_uint2(grid_tab[q3[2*l+0]], grid_tab[q3[2*l+1]]);
            // 7 sign bits + parity bit, identical to unpack_ksigns (the xor fixes bit 7 to the parity of bits 0..6)
            const uint32_t v8  = (aux32 >> (7*l)) & 0xFF;
            const uint32_t sgn = v8 ^ ((__popc(v8) & 1) << 7);

            const int grid_l = ink_t_apply_signs4(grid_pos.x, sgn & 0x0F);
            const int grid_h = ink_t_apply_signs4(grid_pos.y, sgn >> 4);
'''
assert body.count(old3) == 1, body.count(old3)
body = body.replace(old3, new3)
s = s[:i0] + body + s[i1:]
open(p, 'w').write(s)

# ---------------------------------------------------------------- mmq.cuh: fill the LUT before the K loop
p = D + 'mmq.cuh'
s = open(p).read()
old = '''    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles(x, x_gate, tile_x, offset_x + kb0, tile_x_i_max, stride_row_x);'''
new = '''    if constexpr (ggml_cuda_ink_t_lut_bytes(type) > 0) {
        ink_t_grid_fill<type>(threadIdx.y*warp_size + threadIdx.x, nwarps*warp_size);
        __syncthreads();
    }

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles(x, x_gate, tile_x, offset_x + kb0, tile_x_i_max, stride_row_x);'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, 'w').write(s)

# ---------------------------------------------------------------- mmq.cu: count the static LUT in the shared-memory checks
p = D + 'mmq.cu'
s = open(p).read()
old = '''        if (mmq_get_nbytes_shared(config, cc) > smpbo) {
            continue;
        }'''
new = '''        if (mmq_get_nbytes_shared(config, cc) + ggml_cuda_ink_t_lut_bytes(type) > smpbo) {
            continue;
        }'''
assert s.count(old) == 1
s = s.replace(old, new)
old = 'if (cfg.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(cfg, cc) <= smpbo) {'
new = 'if (cfg.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(cfg, cc) + ggml_cuda_ink_t_lut_bytes(src0->type) <= smpbo) {'
assert s.count(old) == 1
s = s.replace(old, new)
open(p, 'w').write(s)
print('patched, LUT mask', lut)
