# HANDOVER: Inkling 2 Mini custom CUDA kernels on the RTX 2080 Ti 22 GB (sm_75)

Status 2026-11-22 00:10: research **stopped** at the user's request (usage limit). All validated work is committed. The GPU is idle and Swift production is restored.

## 1. Current best configuration
- **Host / GPU:** homelab `marvin@<ip>` (Tailscale). RTX 2080 Ti 22 GB, sm_75, 175 W limit.
- **Code:** `~/llm/experiments/inkling-kernels/llama.cpp`, branch `turing-iq3-opt`, HEAD `6e225ed`. Base is `realugbun/llama.cpp@inkling2-speed` 7669d89, the local clone `refs/speed-fork`.
- **Binaries:** `~/llm/experiments/inkling-kernels/bin-best/` holds the HEAD build. `bin-base/` is the pure fork and `bin-cp3/` is checkpoint 3 plus GQA opt-in.
- **Model:** `model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf`. This is the mainline-format GGUF; the old Unsloth `inkling2` GGUF does not load in this fork.
- **Run flags:** `-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 512`.
- **Default build is bit-exact.** Its greedy output is identical to the pure fork, verified on short prompts and on ~7K-token prompts.
- **Opt-in "fast" switches (not bit-exact):**
  - `GGML_CUDA_INK_T_MOE1=3`: single-token i-quant MoE routing, +3% decode at 0K.
  - `GGML_CUDA_INK_T_FAGQA=2`: retuned GQA attention split, +1.8% over mode 1 at 32K.
  - These two were never benchmarked together in one run.

### Switches
| env | default | meaning |
|---|---|---|
| `GGML_CUDA_INK_T_DISABLE=<bits>` | 0 | disable fusions: 1 QKV_POST, 2 RELBIAS_T, 4 ROUTE_TAIL, 8 q8_1 cache, 16 CTRL_H, 32 Q8_0 pairs, 64 scale gather |
| `GGML_CUDA_INK_T_FAGQA` | **1** | GQA-batched q8_0 decode attention, active for n_kv >= 4096. 0 = off, 1 = bit-exact, 2 = faster split (not bit-exact) |
| `GGML_CUDA_INK_T_MOE1` | 0 | 3 = route single-token i-quant MUL_MAT_ID through the MoE kernel (not bit-exact) |
| `GGML_CUDA_INK_T_MMQJ=<J>` | (policy) | override the MoE MMQ tile width. The default policy sizes J for 1.5x the average expert load, which is J=48 at -ub 512 |
| `GGML_CUDA_MMVQ_IQ_NEW=1` | 0 | old IQ3_XXS mat-vec variants (neutral, kept for reference) |
| build: `GGML_CUDA_INK_T_IQ_LUT` | 3 | IQ grids in shared memory (bit 1 IQ3_XXS, bit 2 IQ2_S) |
| build: `GGML_CUDA_INK_T_NO_FAST_I2F` | undef | restore the plain int->float cast in MMQ epilogues |
| debug | | `GGML_CUDA_INK_T_DEBUG=1` match traces, `GGML_CUDA_DUMP_GRAPH=<file>` node dump |

## 2. Cumulative results
All figures are llama-bench, same-session interleaved controls, 2–3 reps.

| metric | b12568 runtime | pure fork | CP3 (bit-exact) | **now (bit-exact default)** | opt-in fast | ceiling |
|---|---|---|---|---|---|---|
| tg128 0K | 39.4 | 57.0 | 62.4 | **62.4** | 64.4 (MOE1=3) | 170 |
| tg128 8K | 37.3 | 53.1 | 56.5 | **56.5** | 57.8 (MOE1=3) | 158 |
| tg128 32K | – | 46.7 | 49.4 | **50.8** | 51.7 (FAGQA=2) | 140 |
| pp512 0K | ~600 | 1222 | 1210 | **1361 (+11% vs fork)** | | ~7,600 compute |
| pp512 8K | ~500 | 1073 | 1056 | **1159 (+8% vs fork)** | | |

### Prefill steps this round (pp512 0K, bit-exact, each validated)
1. J=48 MoE tile policy: +2–3%.
2. Arithmetic IQ sign application plus IQ3_XXS grid in shared memory: +3%.
3. IQ2_S grid in shared memory: +2.5%.
4. Exact full-rate I2F in the MMQ epilogue: +5%.

Together, 1210 -> 1361.

## 3. Build and validation commands
```
cd ~/llm/experiments/inkling-kernels/llama.cpp
cmake --build build -j 12 --target llama-bench llama-server test-backend-ops   # ~3 min after an mmq header change
# GPU sessions MUST go through the Swift-safe wrapper (waits for Swift idle, stops it, always restores):
bash ../kgpu.sh <script.sh>      # log in ../sessions/<time>-<script>.log ; script examples: ../k_s15.sh .. k_s21.sh
# inside a session:
LD_LIBRARY_PATH=$B $B/test-backend-ops -o MUL_MAT_ID -b CUDA0          # 1079 cases; also -o MUL_MAT (2726), -o FLASH_ATTN_EXT
LONG_CHECK=1 python3 k_decode_check.py run $B out.json [ENV=val]       # greedy + logprobs, ~7K-token prompts
python3 k_decode_check.py cmp ref.json out.json                         # "IDENTICAL TOKENS (max logprob delta 0.00e+00)" = bit-exact
$B/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 512 -n 0 -d 0 -r 3   # prefill screen
```
Profiling tools:
- `k_ncu.sh` and `k_ncusum.py`: metric sets.
- `k_s18.sh` plus `k_sass.py`: SASS source counters with per-barrier-segment stall breakdown.
- `k_prof.sh` and `k_buckets.py`: nsys kernel buckets.

## 4. Source map (branch `turing-iq3-opt`, all under `ggml/src/ggml-cuda/` unless noted)
- **`inkling2.cu` / `.cuh`:** decode graph fusions. They are inserted from `ink_t_qkv.inc` (QKV_POST, staging, closure helpers), `ink_t_rb.inc` (RELBIAS_T, ROUTE_TAIL) and `ink_t_ctrl.inc` (CTRL_H), using `k_repatch_qkv.py` in the work dir.
- **`ggml-cuda.cu`:** q8_1 cache, graph dump, Q8_0 pair fusion, MoE scale-gather fold. `common.cuh` holds the cache structs.
- **`mmvq.cu`:** pair mode, MOE1 routing, and the `has_fusion` fix. The fix makes `has_fusion` include `shared_up`; without it a pair launch corrupts output.
- **`fattn-vec-gqa.cu` / `.cuh`:** GQA decode attention, generated by `k_gen_fagqa.py` from the stock vec kernel. It is hooked into `fattn.cu` under `BEST_FATTN_KERNEL_VEC`.
- **`mmq-load-tiles.cuh`:** `ink_t_apply_signs4`, `ink_t_grid_smem`/`ink_t_grid_fill`, and the rewritten IQ3_XXS/IQ2_S loaders.
- **`mmq.cuh`:** fills the shared-memory LUT before the K loop.
- **`mmq.cu`:** J policy and `MMQJ` override; the LUT bytes are counted in the shared-memory checks.
- **`mmq-vec-dot.cuh`:** `ink_t_i2f` in the Turing epilogues.
- **`src/models/inkling2.cpp`:** graph reorder (Q+gate and K+V adjacent).
- **`tests/test-backend-ops.cpp`:** Inkling-shape FA cases (`ink=1`, `INK_FA_NMSE`).
- **Work dir:** `THEORY.md` (ceilings and gap analysis), `ledger.jsonl` (every experiment, including rejected ones), `progress.md`, `results/`, `prof/`, `sessions/`.

## 5. Remaining bottlenecks (measured)
**Decode, 0K: 15.97 ms GPU per token, 37% of the bandwidth ceiling.**
- Expert mat-vecs: 5.7 ms at 42% of bandwidth (gap ~3 ms).
- Zero-byte glue kernels (~2,400 launches per token): ~4.4 ms.
- Loop controllers: 1.2 ms.
- Inter-kernel idle: ~1.6 ms.

**Decode, long context.** Full-attention FA is 3.5 ms per token at 32K before GQA; GQA mode 1 recovers ~0.6 ms.

**Prefill (pp512 1361 = 18% of the INT8 compute ceiling).** The MoE GEMMs dominate: NVFP4 down ~22%, IQ3_XXS gate_up ~20%, IQ2_S ~15% of prefill.
- All of them run at 255 or 222–246 registers per thread, which means 8 warps per SM and 25% occupancy.
- After this round's fixes:
  - **NVFP4 loader:** latency-bound; 69% of loader stall samples are `long_scoreboard`.
  - **IQ loaders:** `mio_throttle` and shared-memory bank conflicts on the LUT (22M conflicts for IQ3_XXS).
  - **MMA epilogue:** FFMA/FMUL scale math, now larger than IMMA itself. IMMA is only 9–13% of instructions.

## 6. Rejected approaches (do not repeat without a new hypothesis)
- **IQ3_XXS decode mat-vec variants:** four warp-per-row/LUT kernels were all neutral (instruction/MIO bound).
- **Shared-memory LUT for decode expert dequant:** rejected. It does help prefill, which is throughput-bound.
- **CUDA-graph configuration probes:** noise.
- **CTRL_T:** one kernel with an in-kernel grid barrier for all controllers. It was 27–31 µs vs 12.5 µs and slower end to end; horizontal batching (CTRL_H) won instead.
- **GQA attention mode 1 without the n_kv >= 4096 gate:** slower at 0K (61.4 vs 63.0).
- **MMQ 2 blocks/SM (I=64, 128 threads), all types or NVFP4 only:** neutral (−0.7%). At 255 registers the SM still holds only 8 warps; it only splits the barrier domain.
- **MoE tile J=64/40/128/24:** worse than J=48.

## 7. Prioritized next steps (estimated gain x likelihood)
1. **Prefill register diet plus software pipelining for the MoE MMQ** (largest gap).
   - Get the NVFP4/IQ kernels to ≤128 registers, giving 16 warps per SM; or double-buffer the next K-tile's raw weights in registers during `vec_dot`.
   - Turing has no cp.async.
   - Evidence: NVFP4 loader 69% `long_scoreboard`, 8 warps per SM. Expected +10–20% prefill.
2. **Cheaper MMA epilogue.**
   - Each 32-K (16-K for IQ2_S/NVFP4) MMA result is scaled with 2 FMUL + FFMA per element.
   - Accumulating int32 over sub-blocks with equal scale, or pre-multiplying dA*dB once per (i,j) pair, would cut FP work. Bit-exactness must be re-checked: rounding order changes, so it would be opt-in.
3. **Bank-conflict-free IQ LUT.** Replicate the 1 KB IQ3_XXS grid per bank group, or mix global and shared-memory lookups to balance the LSU and MIO pipes.
4. **NVFP4 loader with fewer, wider loads.**
   - 5x LDG.64 per 36-byte block instead of ~12 narrow loads.
   - Constant-folded `kvalues_fp4` in PRMT immediates.
   - A bit-exact integer UE4M3->fp32 path; sm_75 emulates the FP8 conversion.
5. **Decode glue.**
   - ~4.4 ms per token remains in zero-byte kernels: conv/mix chains, top-k + ROUTE_TAIL merge.
   - Persistent per-layer kernels are the long-term lever; under graphs, horizontal fusion has worked and grid barriers have not.
6. **Long context.**
   - Make GQA mode 2's split bit-exact: same combine order with fewer, fatter blocks.
   - Prefill FA at depth: pp drops 1361 -> 1159 from 0K to 8K.
   - Do not run 128K/256K until asked.

## 8. Constraints (from the user, still in force)
- RTX 2080 Ti only; preserve Swift production. Every GPU run goes through `kgpu.sh`/`gpuexec.py`, which never forces and always restores.
- The bit-exact build stays the default; non-bit-exact optimizations stay opt-in.
- Benchmarks:
  - IQ3_XXS only.
  - Short pp512/tg128 screens at 0K/8K; 32K only for significant changes.
  - Same-session interleaved controls.
  - No 128K/256K runs, no quantization sweeps, no full coding suites, no model downloads.
- Stop after three consecutive hypotheses below +2% end to end. The current streak is 0: NVFP4 2 blocks/SM failed, then GQA mode 1 passed with +3.5% at 32K.

## 9. v0.2 round 1 addendum (2026-11-22)
- Accepted (bit-exact, commit e0be55a): NVFP4 MMQ half-K tile (layout NVFP4H, K_vram 128, x tile 84 -> 44 ints/row, 2 blocks/SM for J<=48, regs 138 -> 108-114, no spills). pp512 0K 1366 -> 1401 (+2.5%), 8K 1171 -> 1196 (+2.1%). Correction to section 5: the J=48 MoE kernels use 120-138 registers (255 applied to J=128 tiles); the limiter for a second resident block was shared memory (~50 KB), not registers.
- Rejected: NVFP4 register prefetch (+1%, results/nvpf_v02_rejected.patch), grouped-expert multi-token verify kernel (-28% at N=4, results/grp_verify_v02_rejected.patch), IQ3_XXS half-K tile (0%, results/iq3_halfk_v02_rejected.patch; IQ2_S cannot fit 2 blocks because of its 8 KB LUT).
- DSpark parked: economics in THEORY.md (verify cost, break-even L), ledger V02-3/V02-REF. The Mini drafter (rnn_hidden Markov head) needs a graph port into dspark.cpp/dflash.cpp; Basalt v0.5.5 dspark.cu has the math.
- Prefill profile after this round (pp512 0K, nsys): NVFP4 down 18%, IQ3_XXS 17.5%, IQ2_S 12%, Q8_0 MMQ 9.8%, FA 3.9%, fp32 cuBLAS sgemm (relbias projection etc.) ~6.5%, conv 3.4%, quantize 2.4%. At 8K depth FA is 11.4% and inkling2_relbias_mask 2.5%.
- Next candidates: (1) fuse relbias bias gather into the FA mma kernel (per-token table [1024 x 24] instead of the n_kv x 24 x n_tok mask, est. 3% at 8K, more at 32K); (2) fp32 sgemm -> fp16/INT8 tensor-core path (opt-in, not bit-exact, est. up to 4%); (3) IQ3_XXS LUT bank conflicts (replicate grid per bank group); (4) NVFP4 wide 128-bit global loads + exact integer UE4M3 scale path.

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
