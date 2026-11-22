# Inkling 2 Mini on RTX 2080 Ti (sm_75): custom CUDA kernel work - live progress

- **Status:** STOPPED (2026-11-22 00:10, usage limit). Branch HEAD 6e225ed, build in `bin-best/`. Default bit-exact. **Read HANDOVER.md to resume.** GPU idle, Swift restored.
- **Hardware / model:** RTX 2080 Ti 22 GB (sm_75, measured 547 GB/s read, 90 TFLOPS FP16 / 152 TOPS INT8 at 175 W), `Inkling-2-Mini-UD-IQ3_XXS` mainline GGUF, Q8_0 KV, FA on, -ub 512
- **Code:** `~/llm/experiments/inkling-kernels/llama.cpp`, branch `turing-iq3-opt` (base `realugbun/llama.cpp@inkling2-speed` 7669d89), pure fork in `bin-base/`
- **Method:** inference-optimisation skill (`refs/skill`): ceilings first, one change per experiment, interleaved control/candidate/control in the same binary (env switches), greedy-equivalence check for every lossless claim, ledger of all runs incl. rejected ones.
- **Isolation:** Swift production untouched; every GPU session via `kgpu.sh` -> `gpuexec.py` (waits for Swift idle, never forces, always restores).

## Round 4 (2026-11-21/22): GQA attention and prefill expert tiles. Research stopped; see HANDOVER.md
Every item below is bit-exact unless marked otherwise. Validation for each:
- test-backend-ops: MUL_MAT_ID 1079/1079, MUL_MAT 2726/2726, FLASH_ATTN_EXT 4430/4430.
- Greedy decode on ~7K-token prompts identical to the control (logprob delta 0.0).

| commit | change | result (same-session interleaved) |
|---|---|---|
| 8e7e770 | GQA-batched q8_0 decode attention: KV read and dequantized once per KV head for its 3 query heads; per-head relbias mask rows kept | mode 2 (opt-in, not bit-exact): tg 32K 49.4 -> 52.1 |
| 6e225ed | GQA mode 1 (stock split, bit-exact) **on by default**, n_kv >= 4096 | tg 32K 49.06 -> **50.79 (+3.5%)**, 8K 56.18 -> 56.53 |
| 7b00f80 | IQ3_XXS/IQ2_S MMQ loaders: arithmetic sign application `(g ^ 0xFF*t) + t` replaces emulated `__vcmpne4`/`__vsub4`; IQ3_XXS (1 KB) and IQ2_S (8 KB) grids in smem; MoE tile J sized for 1.5x the average expert load (J=48) | pp512 0K 1210 -> 1299 (+7.4%), 8K 1054 -> 1117 |
| 769efa6 | exact full-rate int32->float in the Turing MMQ epilogue (`IADD 0x4B400000` + `FADD`); I2F is quarter-rate on sm_75 and was the top stall | pp512 0K 1296 -> **1361 (+5.0%)**, 8K 1118 -> **1159** |

Rejected this round:
- **MMQ 2 blocks/SM (I=64, 128 threads), all MoE types or NVFP4 only:** neutral/−0.7%. 255 registers per thread hold the SM at 8 warps either way.
- **GQA mode 1 without the depth gate:** slower at 0K.
- **Tile widths J=24/40/64/128:** all worse than J=48.

Cumulative, bit-exact default vs pure fork:
- Decode: 0K 57.0 -> 62.4 (+9.5%), 8K 53.1 -> 56.5 (+6.3%), 32K 46.7 -> 50.8 (+8.8%).
- Prefill pp512: 0K 1222 -> 1361 (+11.4%), 8K 1073 -> 1159 (+8.0%).

The opt-in fast switches (`GGML_CUDA_INK_T_MOE1=3`, `GGML_CUDA_INK_T_FAGQA=2`) give 64.4 at 0K and 51.7 at 32K.

## Checkpoint 3 result (acceptance, 3 reps, interleaved in one session)
| tg128 | pure fork | **turing-iq3-opt default (bit-exact)** | same binary, new fusions off | IQ opt-in (`GGML_CUDA_INK_T_MOE1=3`, 1 rep) | ceiling (THEORY.md) |
|---|---|---|---|---|---|
| 0K | 56.99 | **62.4 (+9.5%)** | 56.2 | 64.4 (+12.9%) | 170 |
| 8K | 53.11 | **56.45 (+6.3%)** | 52.5 | 57.8 (+8.9%) | 158 |
| 32K | 46.73 | **49.4 (+5.7%, 1 rep)** | - | - | 140 |
| pp512 0K / 8K | 1222 / 1073 | 1204 / 1061 | 1205 / 1059 | | ~7,600 compute ceiling |
Prefill is ~1% lower in all runs of the new binary incl. its all-off control: session drift (the fork ran first) or a small cost of the graph reorder - not yet verified. Decode: 37% of the bandwidth ceiling (was 34%).

## Commits this round (each validated; decode identical to the pure fork unless noted)
| commit | change | evidence |
|---|---|---|
| afeadd0 | MoE-kernel routing for i-quants made opt-in; default bit-exact | decode delta 0.0 again |
| 653b3a3 | **CTRL_H** loop controllers: Q/K/V (and FFN) controllers batched horizontally, 8 -> 2 launches per loop visit; norms emulate the fork's 1024-thread reduction exactly | 0K 61.54 / **62.42** / 61.18, 8K 55.32 / **56.05** / 55.07 |
| fff878e | **Horizontal Q8_0 projections**: graph reorder (Q+gate, K+V adjacent; allocator plans the new order, verified exact with the fusion off) + mmvq pair mode (2nd matrix in the shared_up grid slice) + staging of inputs the allocator planned under a fused output | 0K 62.44 / **62.74** / 62.08, 8K 56.06 / **56.40** / 55.83. Found and fixed an upstream gap on the way: `has_fusion` ignored `shared_up` (the extra slice recomputed matrix A out of bounds; caught by the decode check: divergence at token 2) |
| 2aba094 | down-scale **gather folded** into the MoE weighted reduction | exact; within noise (63.07 / 63.00 / 62.65): kept, -44 launches |

### Rejected this round
- **CTRL_T** (all controllers in one kernel with an in-kernel grid barrier): bit-exact but 27-31 us per launch vs 12.5 us for the fork's pair (barrier + serialized latency, 14% DRAM) -> tg 60.44 vs controls 61.61/61.18. Lesson: on Turing at this size, horizontal batching beats vertical fusion through a grid barrier.

## Item 4/5 investigation (no code yet)
- **Long-context attention:** FA vec decode costs 1.59 ms/token at 8K and 3.52 ms at 32K (160-194 GB/s on the KV bytes). Nsight Compute on the 8K full-attention launch: 99.7 us, DRAM 20.7 MB (= the KV once) but **66 MB of L2 reads (3.7x)**, 244 regs/thread, 24% occupancy. Cause: llama.cpp only packs GQA heads when all heads share one mask (`mask->ne[2] == 1`); Inkling's relative-position bias is per head (ne[2] = 24), so each of the 3 query heads per KV head re-loads and re-dequantizes the q8_0 cache. **Next:** a GQA-batched q8_0 decode kernel that reads each KV head once and applies per-head mask rows (est. ~2x on full-attention visits: ~0.3 ms/token at 8K, ~1.4 ms at 32K).
- **Prefill MMQ** (65% of prefill, 16% of the compute ceiling): not started this round; evidence in THEORY.md (1 block/SM from the 48 KB tile, INT8 pipe 8-11%, ~32 tokens/expert).

## Switches
`GGML_CUDA_INK_T_DISABLE=<bits>`: 1 QKV_POST, 2 RELBIAS_T, 4 ROUTE_TAIL, 8 q8_1 cache, 16 CTRL_H, 32 Q8_0 pairs, 64 scale gather. `GGML_CUDA_INK_T_MOE1=3` opt-in IQ routing (not bit-exact). `GGML_CUDA_INK_T_DEBUG=1` match traces, `GGML_CUDA_DUMP_GRAPH=<file>` node dump. The graph reorder (model level) has no switch.

## Earlier checkpoints (summary)
- CP2: F1 QKV_POST, F2 RELBIAS_T, F3 ROUTE_TAIL, F4 q8_1 cache (+8.7% bit-exact), D1 IQ routing.
- CP1: profile (decode kernel-bound, 2,812 launches/token), four IQ3_XXS mat-vec variants neutral (instruction/MIO bound) - opt-in only, not to be retried without a new hypothesis; graph probes were noise.

## Next steps (ranked by gap x likelihood, THEORY.md)
1. GQA-batched q8_0 decode attention with per-head masks (long context).
2. Prefill MMQ tiles for 16-64 tokens per expert (largest relative gap).
3. Remaining zero-byte glue (~4 ms/token): conv/mix chains, top-k + ROUTE_TAIL merge.
4. Verify the ~1% prefill difference (reorder vs drift) with an interleaved prefill-only run.

## Files
`THEORY.md` ceilings, `ledger.jsonl` experiments, `results/` raw bench/decode JSON + hwbench, `prof/` nsys/ncu/graph dumps, `sessions/` logs, fusion sources `ink_t_qkv.inc`, `ink_t_rb.inc`, `ink_t_ctrl.inc`, patch scripts `k_patch_*.py`.

## v0.2 round 1 (2026-11-22): DSpark parked
- Rejected: NVFP4 register prefetch (+1%, V02-1), grouped-expert verify kernel (-28% at N=4, V02-4). The Mini drafter does not load in the fork (needs the rnn_hidden head port). Economics in THEORY.md. Default build unchanged (HEAD 6e225ed, tree clean).

- v0.2 accepted: NVFP4 half-K tile (e0be55a) pp512 0K 1366->1401 (+2.5%), 8K +2.1%, bit-exact. IQ3_XXS half-K neutral (rejected).

## v0.2 round 2 (2026-11-22): prefill attention and IQ lookup experiments, all rejected
Validated build is unchanged: HEAD e0be55a (NVFP4 half-K tile). Ledger entries V02-7 .. V02-STOP; patches in results/.
- V02-7 FA query tile width (sm_75, DKQ=128, ncols2=1): 16 columns pp512 8K 1193 -> 1113 (-7%), 32K 891 -> 780 (-12%); 8 columns -30% / -47%; 64 columns aborts at launch. The 32-column tile is the optimum.
- V02-8 FA GQA packing for per-head masks (ncols2=4 for ratio 3, 3 real + 1 padded head, per-head mask tile): FLASH_ATTN_EXT 4433/4433 but pp512 8K +0.3%, 32K -2.7%, decode logprob delta 6e-2 (not bit-exact). Patch fagqa3_v02_rejected.patch.
- V02-9 IQ3_XXS lookup split smem/L1-global: bit-exact, pp512 0K -1%. Random 8-bit grid indices collide ~3.5-way on 32 banks regardless of replication or padding; only fewer lookups could help. Patch iqmix_v02_rejected.patch.
- Nsight Compute of the stock prefill FA at 8K depth (ncols 32, 128 threads, 244 regs, 2 blocks/SM): long_scoreboard 56%, issue active 14%, SM 14%, DRAM 15% (only ~21 MB read per call), L1 hit 22-28%. It is latency-bound, not bandwidth-bound; Turing has no cp.async and the kernel has no register headroom for a prefetch. Prefill FA is ~10% of a ubatch at ~4K average depth and grows with depth.
- inkling2_relbias_mask materializes n_kv x n_tok x 24 f16 per layer (~214 MB for a full-attention layer at 8K); fusing the gather into FA could save ~2.5-3% at 8K but cannot lift the FA latency ceiling.
- Not done: attention CUDA source study of Maya/NInfer-all/Basalt/SGLang, fp32 sgemm tensor-core opt-in. Attention tuning is stopped after three failed hypotheses.
