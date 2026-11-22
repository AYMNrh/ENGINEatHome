# v0.4: GPU-side relative distances (commit 7d3e5b1 on branch relpos-gpu, built on e4a7ea6) - 2026-11-22
Fallbacks: **e0be55a** (validated build, unchanged) and **e4a7ea6** (single relbias node with the dense distance matrix). Nothing merged. `LLAMA_INK_RELBIAS_NODE=0` still selects the original unfused chain and the dense matrix. GPU time: 22 of 30 minutes. Raw data: `inkling-kernels/results/` (stress_*, nsys_*, rn4_*, rp2_*), `strata-ab/results/` (depth_*, memprobe*).

## 1. CUDA graph safety (Stage 1, done before any change)
Stress test `stress.py` (24 requests per configuration: prompts of 1 to 1500 tokens, alternating prefill and decode, prompt cache on/off, 5 multi-turn requests that reuse the previous prompt plus answer, prompts longer than the 1024 sliding window, n_predict 1 to 24, `-lv 4`, `GGML_CUDA_GRAPH_DEBUG=1`), on three layouts (ctx 16384 ub 512, ctx 2048 ub 32, ctx 4096 ub 128), for e0be55a, e4a7ea6 and 7d3e5b1:
- 9 runs x 24 requests, no crash, no CUDA error, **0 `sched_graph_realloc` events**, every token and every top-2 logprob identical across the three builds (72 requests compared pairwise).
- Root cause of the earlier crash (recorded in ledger V04-S1): the first node variant (node only for n_tokens > 1, chain for decode) gave the prefill and decode graphs different topologies (11,740 vs 12,172 nodes at ub 64). The scheduler then re-planned the compute buffer at every prefill/decode switch (`sched_graph_realloc`, n_diff growing), moving tensors under decode graphs that were captured as CUDA graphs. The crash needed CUDA graphs on, depended on layout (ub 32/64 yes, ub 128/256 no) and disappeared with `GGML_CUDA_DISABLE_GRAPHS=1`, with either half of the fusion bits disabled, and with the unified topology. The baseline never crashed. One sanitizer run showed a MoE gather fusion kernel (`moe_weighted_reduction_gather_f32`) reading garbage expert ids (a value near 0.46f used as an index), consistent with a stale tensor address; two later sanitizer runs were clean, so this was not reproduced on demand.
- **Unresolved risk:** the exact stale pointer was not identified. The CUDA graph reuse path (`cgraph->uid == graph->uid` skips the property comparison in `ggml_cuda_graph_update_required`) trusts llama.cpp to keep tensor addresses stable; any future change that makes prefill and decode graphs differ in topology can reproduce it. Rule adopted: **prefill and decode graphs must have the same topology**; keep `stress.py` as a gate.
- The position inputs (`[n_kv]`, `[n_tokens]` I32) are ordinary graph inputs, identical topology for every batch size, `can_reuse` checks their sizes. They did not destabilize capture: all 9 stress runs above and the decode-equivalence runs pass.

## 2. Implementation (Stage 2)
- `ggml_inkling2_relbias_pos(tab, sinks, cell, tok, mask, type)` (GGML_OP_CUSTOM like the dense node, CPU reference + CUDA): `d = cell[kv] >= 0 && 0 <= tok[t]-cell[kv] < n_rows-1 ? tok[t]-cell[kv] : n_rows-1`, then the same `(tab[d,h,t] - s[h,t]) + mask[kv,t]` rounded to F16. This is the value the dense int32 matrix held after its host-side clamp, so the arithmetic is unchanged.
- The kernel `inkling2_relbias_mask` got an optional token-position pointer (cell positions replace the matrix); the decode fusion RELBIAS_T takes the same node form. No new CUDA allocations (the staging paths only trigger on overlap, which a non-in-place node cannot have), no new synchronization.
- Host side: `set_input_rel_cell_pos` (one O(n_kv) pass over the cells, -1 for empty) and `set_input_rel_tok_pos` (copies `ubatch->pos`). The dense `[n_kv, n_batch]` input is no longer created in node mode. Sliding-window ring cells, empty cells, future cells and cache layouts are handled by the stored cell positions exactly as the old host loop read them (single stream; with n_stream > 1 the old chain and dense matrix are used, as before).
- Validation: test-backend-ops 100/100 on the Inkling chains (new `INKLING2_RELBIAS_POS`: the position node against the dense node on identical random positions with empty cells, bit for bit on CPU and CUDA) and FLASH_ATTN_EXT 4433/4433; decode equivalence (short and 7K-token prompts) identical tokens, max logprob delta 0.0 vs e0be55a; stress as above; 32K and 64K prefill + 32 greedy tokens with top-2 logprobs identical to e4a7ea6.

## 3. Measurements, new (7d3e5b1) vs e4a7ea6
Host cost of the relative distances (instrumented builds, 32K-token prefill at ctx 262144, ub 256, 320 calls incl. decode):
| | e4a7ea6 | 7d3e5b1 |
|---|---|---|
| CPU fill | 1288 ms | 2.0 ms |
| H2D copy | 170 ms | 0.9 ms |
| bytes uploaded | 2,333 MB | 13.2 MB |
GPU: `inkling2_relbias_mask` (nsys, 512-token prefill at 8K depth, 864 launches) 173.3 ms to 168.1 ms (-3%); flash attention 775 ms to 773 ms; total kernel time 6.905 s to 6.886 s. The mask kernel is still 2.5% of GPU time and FA 11%.
Memory (loaded size, q8_0 KV; the saving is exactly 4 bytes x ub x n_kv):
| context / ub | e4a7ea6 MiB | 7d3e5b1 MiB |
|---|---|---|
| 256K / 256 | 20,080 | 19,824 |
| 256K / 128 | 18,316 | 18,188 |
| 256K / 64 | 17,432 | 17,368 |
| 128K / 512 | 17,488 | 17,230 |
| 128K / 256 | 15,664 | 15,536 |
| 64K+ (73,728) / 512 | 14,800 | 14,654 |
| 64K (65,536) / 512 | 14,416 | 14,286 |
| 64K / 256 | 13,456 | 13,392 |
Peak VRAM during real prefills stayed within 20-26 MiB of the loaded size in every run. 256K at ub 512 still does not fit (the per-head F16 mask remains).
End-to-end prefill (cold, 1 run each, tokens and logprobs identical): 32K ub 512: 1074.6 to 1075.4 tok/s; 32K ub 256: 811.1 to 814.9; 64K ub 512: 939.5 to 950.3 (+1.1%); **bounded 128K, ctx 139264, ub 256: 581.8 to 587.0 tok/s (+0.9%), peak 15,960 to 15,824 MiB, decode 33.8 / 33.6 tok/s (noise)**. For reference e0be55a needed 20.4 GB for the same 128K/256 run. Decode at 0K/8K was not re-benchmarked beyond the stress and equivalence runs (the decode graph and fusion are unchanged); decode numbers in the depth runs differ by noise.
Why the end-to-end gain is only about 1%: the host fill (1.3 s) overlaps GPU work of the previous micro-batch, so only the part that waits at the copy shows up. At 256K depth the matrix is 67M entries per micro-batch; my earlier estimate of roughly 25% host overhead there is **not validated** (the 256K cold prefill was not run). What is measured: the host work is now independent of the number of tokens in the batch and about 1 MB per call at 256K.

## 4. inkling-fast2 (nahcmon/inkling-fast-llama.cpp, branch inkling-fast2, `ggml/src/ggml-cuda/fattn-banded.cu`, 1160 lines)
What it does: a separate flash-attention implementation for Inkling. (a) Scalar F32 kernels (one query row per warp, optional split-K + merge) and (b) a Turing/Ampere tensor-core kernel (`flash_attn_ext_banded_mma`, FP16 operands, FP32 accumulate, BK 64, `ldmatrix`, `cp.async` only where available, on Turing two `mma.sync.m16n8k8` per k16 step). The relative bias is never expanded per head: the kernel gets the small bias table `rel` [extent, n_head, n_q] and a single shared mask, and adds `rel[d, h, q]` to each score inside the tile, with `d = iq + (n_kv - n_q) - ik` derived from indices; the per-head F16 mask and the distance matrix do not exist. A pre-pass (`..._tile_flags`) scans the shared mask once per (q-tile, k-tile) and records 0 (fully masked: skip), 1 (visible, all-zero mask: no mask add) or 2 (needs the mask), so the sliding-window "band" and the causal triangle are skipped tile by tile. Sinks are folded into the online softmax.
What does not transfer as is:
- The distance `iq + n_kv - n_q - ik` assumes position == cell index with the new tokens last. Our sliding-window caches are rings (cell != position, wrap-around) and a prompt cache can leave arbitrary cell layouts, so we need cell positions (what the position node uses).
- It supports F16/BF16/F32 K and V only (no q8_0/q4_0 KV), accumulates F32 output without the fork's bit-exact guarantees and has no GQA-3 or vec decode path. Our Q8 KV and the bit-exact kernels must stay.
- It targets full-attention scoring in its own kernel family; replacing our FA needs separate validation (earlier FA experiments: tile width, GQA packing all rejected; prefill FA is latency bound, 244 registers, 2 blocks per SM).
What transferred: deriving the bias from positions instead of building an n_kv x n_tokens matrix (done, in the node kernel, with cell positions instead of index arithmetic); keeping the per-token bias table small; tile-level skipping is the one idea we have not used.

## 5. Go / no-go
**GO for accepting 7d3e5b1 as the node implementation (still unmerged, e4a7ea6 and e0be55a stay as fallbacks).** Evidence: bit-identical outputs everywhere tested, stable under the graph stress test, 4 bytes x ub x n_kv less memory (256 MiB at 256K/ub 256), host work down by three orders of magnitude, small positive prefill effect (+0.1 to +1.1% measured). Conditions: run `stress.py` and the backend-ops suite before any merge; keep the prefill/decode topology rule; the root cause of the stale pointer is still open.
Honest scale: this change is about memory hygiene and removing needless host work; it does not change the long-context economics. The big remaining item is the per-head F16 mask (55 bytes per element, 256K ub 512 does not fit).

## 6. Proposal: relbias inside flash attention (not implemented, needs approval)
Fuse the bias into the mask-load step of our existing `fattn-mma-f16` (the kernel that is latency bound in prefill) instead of writing a per-head F16 mask to global memory: the tile loader would compute `(tab[d,h,t] - s[h,t]) + mask[kv,t]`, `d` from the cell/token positions, rounded to F16 exactly as the node kernel does, so the values (and the output) stay bit-identical, Q8 KV and our MMA kernel stay. Expected benefit (estimates, not measured): compute buffer drops from about 55 to about 7 bytes per (ub x n_kv) element (at 256K and ub 256 from about 3.7 GiB to about 0.5 GiB; ub 512 would fit at about 17 GiB total); removes the mask kernel (2.5% of GPU time at 8K) and the write+read of the per-head mask (part of the FA stall time); tile skipping of fully masked sliding-window tiles could be added later. Risks: modifying the FA tile loader (register pressure is already 244, 128 threads, 2 blocks/SM), per-head table gathers in the tile loader, a new FA variant per head size, test coverage (4433 FA cases plus new ones). Order of work: after accepting 7d3e5b1, prototype behind an env switch, validate bit-exactness and register use, then measure prefill at 8K/32K/128K.
