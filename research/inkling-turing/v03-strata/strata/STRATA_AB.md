# Strata v0.2.19 / strata-inkling 0.1.65 vs Inkling-Turing e0be55a on the RTX 2080 Ti 22 GB (2026-11-22)
Isolated under `~/llm/experiments/strata-ab/` (binary `bin13/`, source clone `src/` at tag v0.2.19, raw results `results/`, drivers `ab.py`, `appendtest.py`, GPU session logs in `../inkling-kernels/sessions/142204, 142954, 143941, 144229, 144855`). Our CUDA fork, the gateway and Swift were not modified; Swift was stopped/restored by `kgpu.sh` for every session (GPU time about 27 of 30 minutes).

## What Strata ships (source + binary inspection)
- Release v0.2.19 (tag c6f6427d9de9, MIT). `strata-inkling` 0.1.65 CUDA13 Linux package (sha256 verified): `BUILD.json` archs `75;80;86;89;120` and `cuobjdump` shows sm_75 cubins (sm_80/86/89/120 too). Driver minimum 580.65; this host has 580.178. `docs/OLDER_GPUS.md` lists RTX 20 (Turing, cc 7.5) as supported and says `STRATA_BF16_TC=1` is an opt-in (off by default on sm_75). The release notes do not mention Turing, and every benchmark in the notes is 5090/4090/3090/Arc/Strix/Spark; the model card contains one 2080 Ti 11 GB row (SQ-10 hybrid: 1,304 prompt tok/s, 42.8 decode tok/s, vendor-reported; not comparable to this 22 GB card).
- The sm_120a FP4 kernels (`ink_moe_fp4_sm120.cu`) are Blackwell-only; on sm_75 the INT8/FP16 paths run (`sm120_tc=0` in the load log).
- Packs: SQ-10 10.5 GB, SQ-15 14.35 GB (vendor: 96.68% top-1 agreement with original weights, KL 0.00889), SQ-20 19.35 GB (lossless SF4+MX8). We used SQ-15 (the largest that leaves KV room on 22 GB), sha256 verified.

## Does our GGUF load? No.
`strata-inkling --info` on `Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf` (+ our dspark Q8_0 GGUF) fails at parse: `GGUF: missing tensor loop_res_query.weight`. Strata's GGUF loader targets the llama.cpp-master/Unsloth shard-rewrite and PR #32187 layouts (global `loop_res_query`, `loop_kv_res_query`, ...); our mainline-support GGUF stores per-block `blk.N.loop_*` tensors and `output_res_query.weight`. Even past the layout, the expert path would not work: our gate/up experts are IQ3_XXS (13 layers), IQ2_S (9), IQ2_XXS/IQ2_XS (1 each) and the down experts NVFP4 (20) / IQ4_NL (4); Strata's GGUF_RAW CUDA expert kernels in the Inkling engine cover Q8_0/Q4_K/Q6_K/IQ4_NL/IQ4_XS/MXFP4/BF16/F16 (IQ3_S in the header comment), not IQ3_XXS/IQ2_*, and NVFP4 is only bit-exact-repacked to SF4 when gate/up are NVFP4 too. So all comparisons below use **SQ-15** (14.35 GB, ~23% larger than our 11.66 GB GGUF and a different quantization): a speed comparison, not a same-weights comparison.

## A/B results (same token-id prompts, greedy, 128 new tokens, one run each, raw code-listing prompts)
Ours: llama-server e0be55a, q8_0 KV, -ub 512 (8K/32K) / -ub 256 (128K), no speculation. Strata: `--serve`, `--parallel 1`, `--no-anti-loop`, default KV (fp16) unless noted. "a" = cold prompt; at 32K the first 8165 tokens were already cached in both engines' slot only for ours (Strata re-prefilled everything, see reuse below).

| Metric | Ours (UD-IQ3_XXS) | Strata SQ-15, DSpark off | Strata SQ-15, DSpark W=4 |
|---|---|---|---|
| 8K prefill (cold) | 1124 tok/s, TTFT 7.30 s | 921 tok/s, TTFT 8.9 s | 921 tok/s, TTFT 8.9 s |
| 8K decode | 55.1 tok/s | 30.7 tok/s | 76.9 tok/s (run a), 49.4 (run b) |
| 8K peak VRAM | 16.1 GB | 16.7 GB | 18.6 GB |
| 32K prefill | 1027 tok/s on the 25.1K new tokens (7.7K cached), TTFT 24.4 s | 812 tok/s on 32.8K tokens, TTFT 40.4 s | 799 tok/s, TTFT 41.0 s |
| 32K decode | 50.1 tok/s | 27.0 tok/s | 60.7 (a), 49.9 (b) |
| 32K peak VRAM | 16.1 GB | 18.7 GB | 20.6 GB |
| DSpark acceptance (drafted tree nodes : accepted tokens) | - | - | 8K 152:98 (64%) and 214:74 (35%); 32K 158:96 (61%) and 197:89 (45%) |
| 128K (131,072-token prompt) | cold prefill 227 s = 577 tok/s, decode 35.9, peak 20.4 GB; edited-tail warm turn TTFT 1.0 s (130,812 of 131,072 tokens cached) | fp16 KV, 135K context: CUDA out-of-memory | fp8 KV (`--kv fp8`, 135K context): loaded and READY, but the 128K prefill had not finished after ~365 s, so the test was stopped (incomplete; decode/acceptance at 128K not measured) |

Warm-prefix reuse:
- Ours: an edited-tail prompt (last 27 tokens changed) reuses 7,676 / 8,192 and 32,252 / 32,768 tokens, TTFT 0.66 s / 0.81 s; at 128K 130,812 / 131,072, TTFT 1.0 s.
- Strata: the same edited-tail prompt got **no reuse** (RESUME not emitted, full re-prefill 8.8 s at 8K, 40 s at 32K): the serve loop reuses only when the previous prompt plus generated tokens is an exact prefix of the new prompt. A true appended turn works: RESUME 8255 of 8295 tokens, TTFT 0.21 s (205 ms prefill). (Strata documents an opt-in `STRATA_CACHE_MESSAGE_BOUNDARY=1` checkpoint for edited final messages; not tested.)

Correctness:
- Strata spec-on vs spec-off (greedy, same weights): identical for 128 tokens on 1 of 4 prompts; first divergence at token 7 (8K a), 18 (8K b) and 8 (32K b). The "100% lossless" claim in `ink_spec.cuh` therefore did not hold bitwise in these runs; the divergences look like near-tie logits (the same token ids differ by one position), but I did not verify that.
- Strata vs ours (different quantization): all 128 tokens matched on 32K a (spec on and off) and on 8K a with spec on; Strata spec-off diverged from ours at token 7 on 8K a. Token ids and tokenization are consistent (same ids, same first tokens), so the prompts are comparable.

## Design inspection (source)
- Speculation: DSpark drafter on GPU (taps from layer visits 5/23/41/48), block-8 chain or W=2..4 multi-branch Markov beam tree, M up to 16/24/32 nodes verified in one CUDA-Graph-captured 48-visit pass (`forward_tree_verify`) with exact greedy tree walk. The verify pass never mutates committed state: per-node K/V and the 4 ShortConv streams are written to ~20 MB staging buffers, then **two fused kernels** commit only the accepted path across all 48 visits (`k_ink_commit_kv_all48`, `k_ink_commit_sconv_all48`) and compact the accepted taps into the drafter's KV. This is the state-rollback design that blocked our own DSpark attempt (sliding-window rings, ShortConv histories, global KV). Accounting: `drafts_offered` counts every drafted tree node, `drafts_accepted` counts accepted tokens, so acceptance is not the chain-acceptance rate usually quoted; tokens per step is the more useful number (not logged by the engine; 128 tokens took 152 offered nodes / 98 accepted at 8K run a).
- Why decode is slower without speculation: Strata's non-spec decode on SQ-15 is 30.7 vs our 55.1 tok/s (-44%); with speculation its tree verify of up to 32 nodes costs little more than one token, giving 2.5x speedup at 8K (77 tok/s) and 2.2x at 32K, which puts it 40% / 21% above our non-speculative decode.
- Attention/KV: 40 sliding-window visits use a ring (`cap_pos` 1024 + chunk), 8 global visits hold full context; KV fp16 or fp8 E4M3 with per-head-token scales; fused relative-position bias + per-query sink in the flash kernel; split-K flash decoding and a GQA-3 kernel that reads K/V once for the three Q heads; prefill attention uses cuBLAS FP16 on sm_75 with an explicit score/softmax step (`k_ink_pf_softmax_rel_sink`), which I suspect (not measured) explains the slow 128K prefill and higher VRAM (peak 18.7-20.6 GB vs our 16.1 GB at 8K-32K).
- Hybrid mode (cold experts in host RAM via a CUDA-graph "doorbell"), dual-GPU expert parallelism, vision: not tested, not applicable to a single 22 GB card with a resident model.

## Verdict
- Strata does **not** beat our custom engine on this card for prefill (-18% at 8K, -21% at 32K), non-speculative decode (-44%), VRAM (+0.6 to +4.5 GB) or 128K behaviour (fp16 KV does not fit; fp8 KV loads but was slower than our 227 s cold prefill; incomplete).
- It **does** win decode with DSpark on: 77 / 61 tok/s vs our 55 / 50 (+40% / +21%) at 8K / 32K, from a drafter we do not have. Quality and quantization differ (SQ-15), and bitwise spec-on/off equality was not confirmed.
- Transferable to our 256K+ work: (1) the staged tree-verify + 2-kernel commit design for exact state rollback (MIT code in `engine/src/inkling2/cuda/ink_spec.cuh`, `ink_dspark.cuh`), the most valuable piece because it solves the DSpark integration we parked and reuses the official drafter taps; (2) the 8-visit global / 40-visit ring KV layout and fp8 per-head-token-scale KV if we need more context in 22 GB; (3) the GQA-3 K/V-sharing and split-K decode ideas (we already rejected GQA packing for prefill FA). Not transferable: sm_120a FP4, hybrid doorbell offload.
- Not evaluated: chat-style prompts, sampled decoding, Strata's own HTTP layer, multi-slot batching, SQ-10 (smaller, closer to our 11.7 GB), repeat runs.

## Limits and incomplete items
One run per cell; raw code-continuation prompts (acceptance on chat or agent text may differ); Strata 8K/32K driver had to be rerun after two harness bugs (first-pass JSONs kept as `*_firstpass_buggy.json`); Strata 128K fp8 incomplete; decode throughput includes the first token for ours and excludes it for Strata; Strata used the default micro-batch/chunk.
