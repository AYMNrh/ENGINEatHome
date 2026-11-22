# Inkling 2 Mini long-context qualification, 128K to 256K (2026-11-22) - definitive report
Baseline **7d3e5b1** (branch relpos-gpu, binary bin-rp2), fallbacks e4a7ea6 and e0be55a unchanged, no kernel or production change. RTX 2080 Ti 22 GB (175 W), UD-IQ3_XXS, **Q8_0 K and V** (never lowered), FA on, one slot, prefix cache on, real docker sandbox shell, temperature 0.
Raw data: `v06-longctx/results/` (ctx_results_*.json, server logs, vram.log, summary), harness `pilot_ctx.py` (`--vary` randomizes needles and decoys), `graph_audit.py`.

## Verdict
| claim | result |
|---|---|
| **Capacity** (loads and runs at ctx 262144, ub 256) | **Yes.** Loaded 19,824 MiB, peak 19,846 MiB (2.7 GB free), KV 4,352 + 106 MiB, compute buffer 4,299 MiB. ub 512 does NOT load at 262144 (cudaMalloc of a 7.57 GiB compute buffer fails); ub 512 loads at 204800 (20.5 GB). |
| **Retrieval** at 242K actual input | **Yes**, 3/3 (fresh randomized needle 9582 found among 3000 / 9700 decoys, service name and checksum seed recalled from the first file after tool use, current rules file identified over a never-adopted draft and the superseded 2024 rules). |
| **Reasoning / multi-file coding** at 242K | **Yes**, hidden tests 7/7, current-rules compliance (legacy/ edited, 4 spaces, `# fixed` comments) true, two bugs in two distant files. |
| **Usable agent performance** | **Qualified for agent work with a cached prefix; not interactive from cold.** Cold prefill of 242K = 556 s (9.3 min, 437 tok/s). Once cached, steps are 1.2-2.5 s TTFT with 26.7 tok/s decode. |
Largest context that is reliable (all checks pass) and interactive once warm: **242,134 actual input tokens, 246.7K occupied after the session in the 262,144 window (about 15K left for generation and tool calls)**. Largest context that is interactive **including** a cold start (about 3 min): about 128K. No claim above 246.7K occupancy was tested.

## Measured (ub 256, ctx 262144, fresh seed 4242, randomized content; tokenizer counts from the server `/tokenize`)
| | 128K run (126,908 tok, 181 files) | 256K-class run (242,134 tok, 334 files) |
|---|---|---|
| cold prefill (first call) | 220.3 s, 584 tok/s | 556.1 s, 437 tok/s |
| fix turn total (time to solution) | 278.6 s, 6 calls, 5 tools (1 expected failing test run) | 641.1 s, 5 calls, 5 tools, all ok |
| hidden tests / rules | 7/7 / ok | 7/7 / ok |
| warm TTFT during the fix turn (median) | 1.59 s | 2.49 s |
| cached TTFT, follow-up turns | 0.85-0.92 s | 1.24-1.34 s |
| cache reuse | 131,071 / 131,125 prompt tokens cached on the first follow-up | 246,015 of 246,069 cached |
| decode | 36.3 tok/s | 26.8 tok/s |
| retrieval / recall / rules | 3/3 | 3/3 |
| peak VRAM (nvidia-smi, 2 s poll) | 19,852 MiB | 19,846 MiB |
| 4-turn session wall | 303.7 s | 674.8 s |
Failures: none of model origin, none of harness origin (no truncated calls, no timeouts, no transport errors). The single `cmd_nonzero` is the pre-fix test run.
**Stability:** 0 `sched_graph_realloc` lines (no scheduler re-plans), no CUDA errors or aborts in the server log, scheduler graph 11,740 nodes / 2 splits as reserved; all model calls of both sessions ran on the same plan. CUDA-graph reuse is not logged by this build, so "stable" means: no crash, no re-plan, identical plan across all calls, no stale-pointer symptoms.
**Time per solved task:** 242K cold: 641 s (about 0.09 tasks/min); 128K cold: 279 s (0.22/min); any follow-up edit inside a warm 242K session costs about 2.5 s TTFT plus decode, so the economics are one cold prefill per repository snapshot followed by cheap turns.

## ubatch effect (same binary, same harness; earlier ub 512 run used another seed and ctx 204800)
| | ub 512 (ctx 204800) | ub 256 (ctx 262144) |
|---|---|---|
| 128K cold prefill | 167.9 s (767 tok/s) | 220.3 s (584 tok/s), **+31 %** |
| decode at 128K | 35.8 tok/s | 36.3 tok/s (same) |
| warm TTFT / cached TTFT at 128K | 1.44 s / 0.85-0.91 s | 1.59 s / 0.85-0.92 s (same) |
| loaded VRAM | 20,542 MiB (ctx 204800) | 19,824 MiB (ctx 262144) |
| ctx 262144 | does not load (OOM, 7.57 GiB compute buffer) | loads |
Correctness at both settings: identical task outcomes. (Known: ub 256 vs ub 512 logits can differ slightly in the baseline itself; no effect on any pass/fail here.)

## Limits and honesty notes
- One run per size and one seed (4242); retrieval/recall/rules turns were 1 sample each, so this is a qualification, not a statistical reliability figure.
- The three follow-up turns are short and rely on the prefix cache; a cold re-prefill after a cache eviction costs the full 9.3 min at 242K.
- Occupancy reached 246.7K of 262,144 tokens; I did not fill the last 15K and made no claim for it.
- GPU time used for this round: 994 s (16.6 min of 30). Swift restored; `/healthz` ok and a real inference request (17*3) returned 51.

## Unresolved graph-integration risks (see v0.6 docs)
Fused relbias-in-FA remains rejected: prefill and decode op sequences diverge (node 1131, 280 nodes with different sources), causing scheduler re-plans; exact hidden read not located. `graph_audit.py` (equal op sequence/sources, 0 live-range overlaps, 0 re-plans) is the gate any future fusion must pass on CPU first. Without the fusion, 256K at ub 256 costs 19.8 GB and 9.3 min cold prefill; the dense per-head mask is what keeps ub 512 out of 256K.

## Earlier 128K / 185K qualification (ub 512, ctx 204800, seed 31)
127,671 tok: cold 167.9 s (767 tok/s), 7/7, rules ok, 3/3, warm TTFT 1.44 s, decode 35.8, peak 20,570 MiB. 185,099 tok: cold 285.7 s (652 tok/s), all pass, warm TTFT 1.85 s, decode 30.5, peak 20,572 MiB. 0 re-plans.
