# Strata IQ3_S: 64K Hermes coding-agent results

7/18 strict agent passes (38.9%); 10/18 held-out patch-grade passes (55.6%).

All 18 attempts were graded. Eleven hit the 40-turn cap; three of those had correct patches but did not finish normally. No episode timed out or failed infrastructure setup.

RTX 5070 Ti 16GB, Ryzen 7 7800X3D, 64GB DDR5. IQ3_S, Strata 0.1.59, INT8 KV, MTP spec 4 / min-p 0.5, 65,536-token capacity, 1500 MiB VRAM reserve, default expert-tail skip 7 enabled (approximate), full PLE, thinking off. Windows native localhost API; pinned Hermes and offline Docker grading.

Six frozen SWE-bench Verified tasks, three trials each, 40-turn limit. Actual context varies by episode; no filler was injected. A correct patch without normal agent completion remains a strict failure. The original 8 GiB physical-memory reserve is evaluated separately; native Strata admission permits this requested diagnostic, and no promotion gate is passed.

| Task | Trial | Agent pass | Patch pass | Seconds | Exit |
|---|---:|---|---|---:|---|
| psf__requests-1142 | 1 | False | True | 115.0 | max_iterations_reached(40/40) |
| psf__requests-5414 | 1 | False | False | 104.8 | max_iterations_reached(40/40) |
| pytest-dev__pytest-10081 | 1 | True | True | 79.8 | text_response(finish_reason=stop) |
| pytest-dev__pytest-6202 | 1 | True | True | 300.3 | text_response(finish_reason=stop) |
| sphinx-doc__sphinx-10323 | 1 | False | True | 278.3 | max_iterations_reached(40/40) |
| sphinx-doc__sphinx-10449 | 1 | False | False | 150.1 | max_iterations_reached(40/40) |
| psf__requests-1142 | 2 | False | False | 139.7 | max_iterations_reached(40/40) |
| psf__requests-5414 | 2 | False | False | 165.2 | max_iterations_reached(40/40) |
| pytest-dev__pytest-10081 | 2 | True | True | 121.9 | text_response(finish_reason=stop) |
| pytest-dev__pytest-6202 | 2 | True | True | 82.8 | text_response(finish_reason=stop) |
| sphinx-doc__sphinx-10323 | 2 | False | True | 258.0 | max_iterations_reached(40/40) |
| sphinx-doc__sphinx-10449 | 2 | False | False | 134.0 | max_iterations_reached(40/40) |
| psf__requests-1142 | 3 | False | False | 111.3 | max_iterations_reached(40/40) |
| psf__requests-5414 | 3 | False | False | 97.1 | max_iterations_reached(40/40) |
| pytest-dev__pytest-10081 | 3 | True | True | 126.2 | text_response(finish_reason=stop) |
| pytest-dev__pytest-6202 | 3 | True | True | 312.4 | text_response(finish_reason=stop) |
| sphinx-doc__sphinx-10323 | 3 | True | True | 312.3 | text_response(finish_reason=stop) |
| sphinx-doc__sphinx-10449 | 3 | False | False | 128.3 | max_iterations_reached(40/40) |

## Measured rates

Rates use native engine timings and include all logged requests in each actual-context bucket, including startup tool/stream probes. These are accepted output tokens, not draft-token rates.

| Actual prompt bucket | Decode tok/s | Fresh prefill tok/s | Requests |
|---|---:|---:|---:|
| 0-8K | 110.35 | 1510.31 | 94 |
| 8-32K | 118.37 | 626.10 | 555 |
| 32-48K | 109.72 | 735.06 | 3 |

Successful tasks/hour: 8.35. Median attempt latency: 131.1 seconds. Tool calls/error results: 637/81. Compactions: 0.

Measured agent-window GPU energy: 72.30 Wh; mean/peak power: 85.5/257.1 W. Power coverage against the full campaign window: 96.2%. Peak GPU temperature: 58.2 C.

Peak sampled VRAM: 14.97 GiB; minimum free VRAM: 670 MiB. Peak native process RSS: 47.83 GiB. Minimum host-available physical RAM: 8.1 MiB, below the original 8 GiB reserve criterion; this profile does not pass that memory gate.

Resource counters and power coverage are in the accompanying JSON. The agent measurement stream excludes model startup; overlapping runtime energy is not added. GPU Wh is device-wide GPU-only, including other applications; CPU and wall power are unavailable. Native process RSS can include mapped/shared weights. Tool errors include failed shell/test commands. Cache reuse uses native per-request counters, without adding duplicate episode totals.

## Separate filled-context speed probe

Three requests each had 63,971 actual input tokens: 102.3 cold, 131.9 cached, 132.5 cached tok/s. Outputs: 718/610/731 tokens; latency: 27.3/9.2/10.1 s. All three retrieval-canary and Python-syntax sanity checks passed. These are not Hermes task passes.

## Findings and limits

Earlier forced mmap and clamped resident-budget configurations measured ~20 and 8.7 tok/s respectively. Restoring automatic arena placement and the original INT8/MTP/default-skip settings raised measured speed; multiple settings changed, so this does not isolate one optimization. Windows automatically grew its page file during full-arena runs; its earlier 67.2 GiB commit limit was a point-in-time observation. No manual page-file change was made.

Windows grading writes patches with LF and keeps Git stderr warnings out of patch contents. A known-correct fix passed all six held-out preflight checks before this matrix. The first launch stopped at its C: disk reserve before any task; results/workspaces were placed on a separate data drive and the unchanged eight-GiB disk reserve passed. Raw failed-preflight evidence is retained locally and excluded from model quality.

Expert-tail skipping 7 changes computation; these scores do not establish equivalence to full-expert inference. This run does not validate 128K or one-hour compaction/resume stability.

[Machine-readable measurements](results/strata-iq3-hermes18-20261109.json)
