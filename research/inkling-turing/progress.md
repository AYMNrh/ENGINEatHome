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

## v0.2 round 3 (2026-11-22): source reading and fragment-direct IQ3_XXS, rejected
Validated runtime unchanged: HEAD e0be55a. Ledger V02-10, V02-RES. Optimization round declared complete.
- Source reading (Maya, NInfer-all, Basalt, SGLang). Verified from source: Maya's NVFP4 expert kernel keeps weights packed in global memory, 32-bit loads, register nibble decode, one-block prefetch; its MMA is m16n8k16 fp16 (sm_80+). NInfer-all GGUF prefill is ggml-style MMQ (same as ours); its Q4-Q8 prefill needs cp.async + BF16 MMA, its NVFP4 kernel is sm_120a only. SGLang's native NVFP4 path calls FlashInfer Blackwell kernels, tile width depends on token count, rows padded to 128 per expert for scale blocks. Nothing is portable to sm_75 beyond our MMQ.
- Unverified / not read: Maya prefill files; any Basalt prefill expert GEMM (none found in the files read, not proven absent); SGLang Triton/CUTLASS fallbacks for pre-Blackwell; findings partly come from subagent summaries and were spot-checked only.
- Three designs compared: (1) fragment-direct dequant (no smem weight tile: ~8 KB smem, 128 regs; risk: redundant global gathers; exact by construction), (2) token-count-adaptive tile width (J=24/40/64/128 already measured worse than J=48; not implemented), (3) persistent tile scheduler (tail waves small, est. <1%; not implemented).
- V02-10 result (design 1, IQ3_XXS): MUL_MAT_ID 1074/1077, decode logprob delta 0.9 (mapping bug not chased), pp512 0K 1404 -> 1222 (-13%), 8K 1194 -> 1074 (-10%). Cause: six redundant 32-bit global loads per lane per 32-value fragment plus sign/popc work exceed the saved shared-memory round trip. Patch results/iq3_direct_v02_rejected.patch.
- Four consecutive failed hypotheses (V02-7..V02-10) after the last accept (V02-5). Remaining unstarted options: opt-in fp32 sgemm to tensor cores (<=4%), fused relbias gather in FA (~2.5-3% at 8K), NVFP4 128-bit loads, decode glue items in HANDOVER section 5. DSpark stays parked.

## v0.4 (2026-11-22): single relbias node, GPU-side distances (details in RELPOS_GPU.md)
- e4a7ea6 (branch relbias-node): the per-head mask is one GGML_OP_CUSTOM node. ggml-alloc used to plan the F32 intermediates of the fused chain (199 bytes per ub x n_kv); now 55. 256K/ub 256 loads (20,080 MiB), 128K/ub 512 loads (17,488), outputs identical to e0be55a. Patch 0019.
- Lesson: prefill and decode graphs must have the same topology. A node used only for n_tokens > 1 made the scheduler re-plan the buffer at every switch and a captured CUDA graph crashed (illegal access, layout dependent, stale tensor address not identified). `stress.py` (alternating prefill/decode, ub 32..512, prefix reuse, multi-turn, prompt > SWA window) is the gate; 9 runs clean with 0 reallocations.
- 7d3e5b1 (branch relpos-gpu): the node takes cell positions and token positions; the CPU-built int32 [n_kv, n_tokens] matrix and its upload are gone (32K prefill at ub 256: 1288 ms fill + 170 ms copy + 2.3 GB -> 2 ms + 0.9 ms + 13 MB; memory -4 bytes x ub x n_kv; prefill +0.1..+1.1%; bounded 128K 581.8 -> 587.0 tok/s, 15.8 GB peak). Bit-exact vs e4a7ea6 and e0be55a. Not merged; fallbacks e0be55a and e4a7ea6, LLAMA_INK_RELBIAS_NODE=0 keeps the chain. Patch 0020.
- Still open: the per-head F16 mask (55 bytes/element; 256K at ub 512 does not fit; mask kernel 2.5% of GPU time at 8K). Proposal: compute the bias in the tile loader of our fattn-mma-f16 (inkling-fast2 idea, adapted to rings and Q8 KV); needs approval. The 256K host-overhead estimate stays unvalidated.

## v0.5 (2026-11-22): relative bias inside flash attention: NO-GO as built (details in RELFA_FEASIBILITY.md)
- Baseline stays 7d3e5b1 (relpos-gpu); fallbacks e4a7ea6, e0be55a. Remaining compute buffer: 51.1 bytes per ub x n_kv element (48 = per-head F16 mask), target about 3 (not 7).
- Fusion point is the mask-tile load of fattn-mma-f16 (D=128, ncols1 32/16/8, ncols2=1): registers 204 -> 212, no spills, exact arithmetic reproduced; unit tests (new INKLING2_FA_REL, f16 and q8_0) bit-exact, FA 4433/4433.
- Two model-integration variants (descriptor node + pinned outputs; real sources 6..9) failed the stress harness: crashes, differing logprobs, scheduler re-plans (reserve graph 11,740 nodes vs real prefill 11,862); also with CUDA graphs disabled. Rejected patches kept. Rule reinforced: real sources only, same node count for prefill and decode graphs, no hidden tensor dependencies.
- Next: policy ub 256 at 256K; GPU mask node from positions (same pattern as 7d3e5b1); understand the node-count difference with a CPU-only allocator test before touching FA again.


## v0.6 (2026-11-22): graph audit of the rejected relbias-FA fusion + long-context qualification at 128K / 185K
Baseline unchanged: **7d3e5b1** (branch relpos-gpu); fallbacks e4a7ea6 (relbias-node) and e0be55a unchanged. FA relbias fusion stays rejected. No kernel or production change.

**Phase 1 (offline, `graph_audit.py` + `k_patch_dump.py`, CPU dumps and one 7 s CUDA dump of the rejected fused variant).**
- The llama-level graph has the same node count in reserve (pp), real prefill and decode graphs, fused or not (CPU 11,164, CUDA 11,740). The 11,862 seen in `sched_graph_realloc` is the scheduler graph after split/copy insertion (122 nodes added by ggml_backend_sched), not a graph-builder difference.
- The fused variant breaks op-sequence equality between the prefill and the decode graph: identical node count and op histogram but the op SEQUENCE diverges at node 1131 (PAD + bias MUL_MAT + descriptor CUSTOM sit before the Q-clip chain in prefill, after it in decode) and 280 nodes have different source lists (unfused: 0). The allocation planned for one graph does not fit the other: scheduler re-plans with unchanged backend ids and graph size. Independent of CUDA graphs, which is why disabling them did not help.
- Live-range overlap check on the real graphs: 0 overlaps in about 7,200 tensors (fused and unfused); the exact corrupting access (hidden read through another tensor pointer / split copy) is still not located.
- Deterministic regression test (no GPU): pp, tg and real graphs must have equal node count, equal op sequence, equal source index lists, zero live-range overlaps, zero re-plans in a CPU stress. 7d3e5b1 passes, the fused variant fails.
- Safe future integration: identical nodes in identical positions for prefill and decode (change what they compute, not the topology), real sources only, run the audit on CPU before any GPU test.

**Phase 2 (7d3e5b1, q8_0 KV, ctx 204800, ub 512, np 1, real docker sandbox, 4 turns, prefix cache).**
| size (tokenizer) | cold prefill | prefill tps | hidden tests | rules | retrieval/recall/rules | warm TTFT | cached follow-up | decode tps | VRAM loaded/peak |
|---|---|---|---|---|---|---|---|---|---|
| 127,671 | 167.9 s | 767 | 7/7 | ok | 3/3 | 1.44 s | 0.85-0.91 s | 35.8 | 20,542 / 20,570 MiB |
| 185,099 | 285.7 s | 652 | pass | ok | 3/3 | 1.85 s | 1.04-1.15 s | 30.5 | 20,542 / 20,572 MiB |
No model error, no harness failure, 0 scheduler re-plans, no CUDA errors in the server log. GPU time about 11 of 30 minutes.
