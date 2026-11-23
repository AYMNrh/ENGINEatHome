# PLAN.md — Qwen3.8-27B-OBLITERATED Q4_K_M on RTX 2080 Ti (175 W), OpenCode-agent throughput

Method: `inference-optimisation` skill (ceiling first; one change per run; control/candidate/control; raw ledger; rejected ledger).
Date: 2026-11-23. Host: homelab, llama.cpp build 12519 (98d67c8d5), GPU limit 175 W (checked by the gate before every run).
Production gateway stopped for the duration (backup of config: `../obliterated-bench/config.json.preserved`, verified byte-identical afterwards); restored `active`, lazy/unloaded.
No production config, kernel or model file was changed. Raw data: `ledger.jsonl` (170 turn rows), `ledger_side.jsonl`, `raw/`, `configs/`.

## Workload (what "agent throughput" means here)
`scripts/agentsim2.py`: one growing OpenCode-like conversation. Turn 0 = cold prefill of START tokens of real C++/CUDA source; each later turn appends the previous
answer + ~1,500-token tool result + an instruction (code / explain / JSON tool plan / diff / bullets / tests), max 640 new tokens, thinking off.
Metrics per turn: cold/tail prefill tok/s, TTFT, decode tok/s (server timings), draft_n / accepted, step time, GPU util/mem-util/clock/power (300 ms nvidia-smi stream), loop flag.
Two sampling modes: `greedy` (temp 0) and `prod` (temp 1.0, top_p 0.95, top_k 20 = the Swift production sampler). **agg_tps = completion tokens / decode seconds over all turns.**
Noise measured: identical-config repeats differ by 0.1-0.2 % (28.15 vs 28.12; 35.05 vs 34.98; 33.13 vs 33.15), so effects >1 % are real **for this harness** (single machine, fixed prompts).

## Ceilings (THEORY.md)
Measured bytes: main blocks 14.778 GB + LM head 1.043 GB per token (embedding is a row lookup); MTP block 0.263 GB. KV q4_0 = 18,432 B/token (16 attention layers).
Bandwidth 547 GB/s (measured earlier on this card, Inkling work). Practical no-MTP decode ceiling 34 tok/s @4K, 32 @64K.

| Metric | Measured | Ceiling | % |
|---|---|---|---|
| no-MTP decode 4K / 16K / 52K / 64K | 23.4 / 21.0 / 17.0 / 16.0 | 34 / 34 / 32 / 32 | 68 / 62 / 53 / 50 |
| MTP d2 decode 65K (greedy, old prompts) | 24.1 | 65 | 37 |
| prefill 4K / 64K | 540 / 387 | 733 / 600 | 74 / 65 |

Memory-controller utilisation during decode is only 40-48 % with SM "util" 97-98 %: **decode is kernel/latency-bound, not bandwidth-bound.** Step-time arithmetic:
no-MTP step 47-48 ms @16K, 58.6 ms @52K; MTP adds ~15 ms per drafted token (bandwidth cost of a draft step is ~2.4 ms) -> the draft/verify path is the inefficiency.

## Experiment log (all numbers agg_tps unless noted)
| id | change | result | verdict |
|---|---|---|---|
| probe | per-request `speculative.n_max` override | ignored (eff depth 3.8 in all) | rejected: depth needs relaunch |
| probe | replay same prompt after longer one | cache_n 25 -> full re-prefill (recurrent state cannot roll back without checkpoint) | noted |
| s1/s7 | no-MTP 16K greedy controls | 20.6 / 21.16 | baseline |
| s2/s6 | MTP d2 16K greedy | 35.05 / 34.98 (+70 %), acc 0.82 | |
| s3 | d1 | 30.81 | worse |
| s4/l7 | d3 | 35.71 / 35.88 | +2 % vs d2 |
| s5 | d4 | 36.20 | +3 % vs d2, acc 0.67 |
| p1-p4 | **prod sampling** 16K: base 20.6; d1 29.63; d2 33.13/33.15; d3 29.93 | depth 2 best at prod sampling (greedy sweep would have mis-picked 3-4) | |
| l6 | d4 + p-min 0.6 (greedy) | 38.76 (+8 % vs d3) | accepted lever |
| y1-y3 | prod: d4 p-min .6 34.46; d4 p-min .75 32.87; **d3 p-min .6 35.96** | d3+p-min 0.6 beats d2 by +8.5 % | accepted |
| x1-x3 | 52K prod: base 16.98; d2 28.13; d3 30.28 | MTP +66..78 % at depth | |
| c1 | `--cache-ram 0` + short side request (title call) | next turn re-prefills 20,645 tokens = 44.1 s TTFT (vs 4.3 s) | **problem** |
| c2/c3 | `--cache-ram 8192` or `--parallel 2 --kv-unified` | tail only, 4.25 s; decode unchanged | accepted |
| x4 | `--cache-ram 8192` at 52K | turn after side request 5.9 s TTFT (cold 52K prefill = 131.6 s) | confirmed |
| l1/z2 | KV q8_0 | 16K +1.7 %; 52K +4.6 % (29.43), 18.4 GiB | modest |
| l2/z1 | KV f16 | 16K +7 %; 52K **+10.7 %** (31.13; step 77.9 vs 90.7 ms), 20.1 GiB | accepted (VRAM) |
| l4 | ub 256 / b 1024 | prefill +6 %, decode flat | small |
| l5/z3 | ub 512 / b 2048 | prefill **+10 % cold, +8 % tail**, decode flat (28.31 vs 28.15), +0.13 GiB | accepted |
| l3/l3b | FA off + f16 KV | l3 failed to start (q4_0 draft KV needs FA); l3b 32.4 vs ~35.8 (-9 %), 21.5 GiB | rejected |
| w1-w4 | **stack** d3, p-min .6, KV f16, ub512/b2048, cache-ram 8192 @52K prod vs d2 q4_0 control | 33.70 vs 28.15 / 28.12 controls (**+19.7 %**), prefill +10 %, 20.4 GiB; q8 variant 30.21 (+7.3 %) | **candidate best** |
| w5 | stack f16 @65K greedy, same prompt as the 24.10 run | **29.50 tok/s**, prefill 404, acc 0.766, needle OK, peak 20.4 GiB, mean 171 W | |

## Rejected / failed ledger (do not repeat without a new reason)
- Per-request `speculative.n_max`: server ignores it (this build).
- FA off: -9 %, +3 GiB VRAM; q4_0/q8_0 KV and draft KV need FA.
- Depth >= 4 without p-min: no gain at prod sampling (acc 0.67 greedy / lower with sampling).
- Fixed d3 at prod sampling @16K: -10 % vs d2 (but +8 % at 52K) -> use p-min instead of a fixed depth.
- ubatch 256: prefill gain only half of ubatch 512's.
- nsys 2023.4.4 CUDA trace (graphs on and off): `.qdstrm` captured but QdstrmImporter fails ("Wrong event order", EventCollection.cpp:1048). Not a model problem; needs a newer nsys or ncu windows. **No kernel-level profile was obtained.** Observation only: disabling CUDA graphs cost 0 % on no-MTP decode (20.96 vs ~21.0) and ~-7 % on MTP d3.
- Quantization alternatives (IQ4_XS 15.4 GB / Q3_K_M 13.5 GB on HF): NOT tested. With memory-controller util at 40-48 % a smaller file cannot give a proportional win; defer until a kernel profile says weight bytes matter. IQ4_XS would also need to be checked for the blk.64 MTP head.

## Not verified (say so plainly)
- Greedy MTP vs no-MTP outputs: byte-identical at 4K/16K (d2, earlier session); at other depths turn-0 text hashes differ across depths (numeric M=1 vs M=k difference), logits at the divergence point were not inspected.
- Quality of f16-vs-q4_0 KV: f16 is the higher-precision option, not measured for task quality. No soak, no full needle depth sweep on the stacked config (only one 65K midpoint needle, greedy).
- Real OpenCode sessions (tool schemas, thinking on / xhigh) not replayed; acceptance depends on content (ranged 0.55-0.97 per turn).
- Stack VRAM 20.4 GiB of 22 GiB at 65,536 context: ~1.6 GiB free; not soak-tested for 1+ hour or with the dashboard/other processes on the GPU.
- Whole experiment ran 1 slot, no concurrency. w1 (control) overlapped a nice-19 nsys import on the host CPU for a few seconds; w4 (control b) did not, and agrees to 0.1 %.

## STATUS / plateau
Config levers are not yet at a plateau on the draft path; the remaining gap is in the draft/verify path and long-context attention (needs kernel work, out of scope for this phase).
Ranked next steps:
1. (config, ~0 risk) adopt `--cache-ram 8192` (or `-np 2 --kv-unified`) — removes 40-130 s stalls after any side request. Needs your approval to touch production.
2. (config) `--spec-draft-n-max 3 --spec-draft-p-min 0.6`, `-ub 512 -b 2048`, KV f16 if the 1.6 GiB VRAM margin is acceptable, else q8_0 (+7 %).
3. (profiling, no kernel edits) get a per-kernel decode/verify breakdown at 16K/52K with ncu windowed capture or a newer nsys; decide among: draft-step latency (15 ms vs 2.4 ms ideal), q4_0 flash-attn dequant, SSM state copies in verify.
4. (kernel) only then: Turing-specific fused draft step / q4_0 attention path.

## Follow-up: production integration evidence (2026-11-23)

The optimisation study above was turned into a production profile; the evidence is in:

- `isolation/`: client-isolation matrix for llama.cpp issue #27148 (light and 12K-token scale, per configuration), the cache-restore probes with the server's own cache decisions, and `pool-consumption-evidence.log` (why a prompt sharing a long prefix consumes a cached conversation).
- `qualification/`: staging-gateway qualification (F16 and Q8_0 KV), the production switching/Swift-cache run, and the deliberately oversized replay that reproduced the pool-consumption case.
- `opencode/`: real OpenCode 1.18.16 runs against the staging and production gateway.

Write-up: [docs/obliterated-profile-and-ram-cache.md](../../docs/obliterated-profile-and-ram-cache.md).
