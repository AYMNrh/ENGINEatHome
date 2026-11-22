# v0.5: relative bias inside our flash attention - feasibility and prototype (2026-11-22) - NO-GO as built
Baseline: **7d3e5b1** (branch relpos-gpu), accepted earlier as the validated research baseline; fallbacks e4a7ea6 and e0be55a. Source tree restored to 7d3e5b1, clean. Nothing merged, no production change. GPU time about 17 of 30 minutes. Raw data: `inkling-kernels/results/` (stress_rf*, stress_rfg*, regs), rejected patches `results/rejected/relfa_inkernel_bias_rejected.patch`, `relfa_structural_rejected.patch` (+ the `k_patch_relfa*.py` scripts that built them).

## Stage 1 - feasibility
- **Where the remaining compute-buffer bytes are.** From the 7d3e5b1 load probes (256K, ub 128 to ub 256, q8_0): 1,636 MiB for 128 x 262,144 elements = **51.1 bytes per (ub x n_kv) element** (the earlier 55 included the distance matrix that 7d3e5b1 removed). 48 bytes are the per-head F16 mask `[n_kv, n_tok, 24]` that flash attention reads, about 2 bytes the shared dense mask, about 1 byte other. Removing the per-head mask would leave roughly **3 bytes** per element, not 7 (my earlier 55 to 7 projection is superseded: the real target is 51 to about 3). Estimated, not measured: 256K at ub 256 from 19.8 GB to about 16.9 GB, ub 512 would fit at about 17.4 GB (static 15.5 GB + 3 B x 512 x 262,144 = 0.4 GB + the bias tables of the 8 global visits, 98 KB x ub each, about 0.4 GB).
- **Smallest fusion point.** The mask enters the mma kernel in `flash_attn_ext_f16_load_mask` (fattn-mma-f16.cuh), which copies a `ncols1 x nbatch_fa` F16 tile into shared memory once per K tile; the Turing (no cp.async) branches are plain half2 copies. Computing `f16((tab[d,h,t] - sink[h,t]) + mask[t,kv])` there, with `d` from `cell[kv]` and `tok[t]`, is a few lines. Prefill flash attention is the D=128, ncols1 32/16/8, ncols2 = 1 instance.
- **Registers / occupancy** (cuobjdump, measured): prefill kernel 204 to 212 registers (ncols1 32), 208 to 208 (ncols1 8/16), no spills, LOCAL 0, STACK unchanged 32 B, still 2 blocks/SM. The loader has room; the kernel is the one that is latency bound (long_scoreboard 56%), so +8 registers is not what limits it.
- **Exact arithmetic.** The tile value is produced with the same expression as the mask kernel: `__fadd_rn(__fsub_rn(tab, s), mask)` then round-to-nearest-even F16, so the F16 values and the softmax input are identical to the dense per-head mask (unit tests below confirm bit for bit, including q8_0 K/V).
- **Can the operator take position metadata without new large intermediates?** Yes: four small tensors (the bias table `[1025,24,n_tok]`, the sink logits `[1,24,n_tok]`, cell positions `[n_kv]`, token positions `[n_tok]`) as extra sources of the flash attention node; the kernel argument list gained one by-value struct (all four flash attention kernels share the typedef). Required side changes: the shared mask has ne[2] = 1, which makes the dispatcher pick GQA packing (ncols2 = 4 for ratio 3, the shape of my earlier rejected FA experiment), so the rel path forces ncols2 = 1 (one Q head per tile, the same kernel configuration as today); the `KV_max` mask scan must stay off; CPU flash attention needs the same bias for test-backend-ops; the cp.async (Ampere) loader branch is not covered, so the fusion is Turing only.
- **Graph topology.** The decode graph (one token) keeps the real node and the RELBIAS_T fusion; to keep the op sequence of prefill and decode graphs identical (the rule from the earlier crash) the prefill graph needs a placeholder node per global layer.
- **CUDA graph stale-pointer risk (still unresolved).** Same class as before: tensors that the allocator plans but a kernel reads through a pointer the graph does not show. It is the reason I did not go further.

## Stage 2 - prototype and result
Two variants were built on top of 7d3e5b1 and kept as rejected patches:
1. `relfa_inkernel_bias_rejected.patch`: a descriptor custom node (no output) hung on the flash attention node as source 6, loader fusion as above, tables/sinks/positions pinned with `ggml_set_output`.
2. `relfa_structural_rejected.patch`: the four tensors as real sources 6..9 of the flash attention node (no hidden dependency), the descriptor only as an expanded placeholder.
- **What works:** compile, 106/106 test-backend-ops (new `INKLING2_FA_REL`: fused flash attention vs flash attention over the real per-head mask of the same positions, bit for bit on CPU and CUDA, K/V f16 and q8_0, n_tok 7 and 512, n_kv 512 to 2048) and FLASH_ATTN_EXT 4433/4433.
- **What fails (stress.py, 24 requests x 3 layouts, same harness that passed 9 runs for e0be55a/e4a7ea6/7d3e5b1):**

| variant | ctx/ub 16384/512 | 2048/32 | 4096/128 |
|---|---|---|---|
| 1, final | 24/24 identical, 19 scheduler re-plans | crash (illegal address) after 3 requests | 24 requests, 11 differ, 27 re-plans |
| 2 | crash after 8 requests, 8 re-plans | crash after 3 | crash after 5, 9 re-plans |
| 1 with CUDA graphs disabled | 1 differs | crash after 14 | 10 differ |
- Decode-equivalence (short and 7K prompts) differed or the server died for both variants. Reference builds: 0 differences, 0 re-plans.
- The reserve graph has 11,740 nodes (same as 7d3e5b1) but the real prefill graphs of the fused variants have 11,862, and the scheduler re-planned the compute buffer repeatedly, the very hazard documented for the earlier topology crash. CUDA graphs disabled still fails, so this is not only the capture class; the exact corrupting access was not located (the unit tests rule out the kernel arithmetic and the tile loader on contiguous tensors; model-only differences left: views of the KV cache, scheduler split copies, allocator lifetimes of the extra sources).
- Stopped after two integration variants, as asked: no uncontrolled attention rewrite.

## Stage 3 - not run
Acceptance criteria 1 (correct results) and 2 (no graph crashes / re-planning) fail, so no 32K/64K/128K or 256K/ub 512 benchmark was run (it would not be meaningful). No throughput or peak-VRAM numbers for the fused kernel exist.

## Remaining safety risks
- The stale-pointer / re-planning class is still not explained; fused-FA work must not resume before it is: reproduce on a CPU-only ggml-alloc/scheduler test (build the pp and tg graphs, assert equal node count and no realloc) and find where the 122 extra nodes of the real prefill graph come from.
- Hidden dependencies (descriptor nodes, kernels reading tensors through other tensors' pointers) are unsafe with ggml-alloc and the scheduler; use real sources only.
- Turing-only (cp.async branch) and head size 128 only.

## Go / no-go
**NO-GO** for relative bias inside flash attention in this form. 7d3e5b1 remains the baseline. The memory prize is real (about 3 GB at 256K/ub 256 and a fitting ub 512) but the integration risk exceeds what an isolated change can carry.

## Alternatives (not implemented)
1. **Policy, available now:** ub 256 at 256K (19.8 GB, 813 tok/s measured at 32K depth) or ub 128 (18.2 GB) for longer contexts; no code.
2. **Mask node from positions (recommended next, low risk, same pattern as the accepted 7d3e5b1):** the dense shared mask (2 bytes per element) is also built on the CPU every micro-batch (`set_input_attn_mask`) and uploaded (134 MB per ub-256 batch at 256K). A custom node that produces the same F16 mask on the GPU from the cell and token positions (causality, sliding window, sequence) would remove that host work and upload with the exact mechanism proven here; it needs the existing mask semantics reproduced bit for bit (0 / -inf), which is simpler than the bias.
3. **Banded bias (needs separate approval):** bias saturates at distance 1024, so a per-head far-value plus a 1024-wide band would shrink the dense mask from `n_kv` to 1024 columns, but still needs a flash attention loader change and inherits the risks above.
4. **inkling-fast2's own kernel family** would replace our flash attention, drop Q8 KV and lose bit-exactness: not recommended.
