# OpenCode + Strata IQ3_S: full 64K task matrix

**16/18 strict agent passes (88.9%); all 18 patches passed held-out grading.** Six frozen SWE-bench Verified issues, three fresh attempts each. The earlier hard-task pilot is separate and is not added to these totals.

Subsequent failure inspection found a custom-tool timeout-unit bug: values such as 120 were interpreted as milliseconds despite the agent intending seconds. The two capped runs contain repeated premature command timeouts. These historical strict completion results are harness-affected; raw scores and independent patch grades are preserved. A corrected explicit-seconds tool is being tested separately.

| Issue | Strict passes | Patch passes |
|---|---:|---:|
| Requests 1142 | 3/3 | 3/3 |
| Requests 5414 | 3/3 | 3/3 |
| Pytest 10081 | 3/3 | 3/3 |
| Pytest 6202 | 1/3 | 3/3 |
| Sphinx 10323 | 3/3 | 3/3 |
| Sphinx 10449 | 3/3 | 3/3 |

Both strict failures reached the 100-step boundary on Pytest 6202 despite passing patch grading. There were no overall agent timeouts; premature shell timeouts were subsequently found as described above. All failures and prior results are preserved, with no retries. OpenCode + 1K remains the preferred measured candidate; the historical strict pass-count criterion of 15/18 is met with the harness qualification, but the original physical-memory reserve and untested retrieval milestones prevent overall promotion. A later separate [corrected stability and expert-tail comparison](strata-iq3-efficiency-20261110.md) validates compaction/process recovery without rescoring this matrix.

Measured native accepted decode: **97.6 tok/s**, including reasoning. Fresh prefill: **833.8 tok/s**, excluding reused tokens. Capacity 65,536; maximum observed prompt 57,294 tokens, with no injected filler. Median episode latency 253.8 seconds; 11.11 strict passes per episode-hour. Cache reuse 97.42%. There were 723 tool calls and 34 OpenCode error states; this error counter can omit nonzero shell exits serialized inside successful custom-tool responses. Compaction count is unavailable, not zero.

GPU-only device-wide energy: 146.74 Wh, mean 101.83 W / peak 267.47 W. Agent telemetry covers its 5,188.02-second window fully; startup is excluded, and overlapping runtime telemetry is not added. CPU and wall power are unavailable. Peak GPU temperature 63.84 C; mean utilization 48.70%. Peak sampled VRAM 14.85 GiB, native RSS 48.40 GiB (can include shared/mapped memory). Minimum available Windows physical RAM 26.57 MiB, below the original 8 GiB reserve.

Settings unchanged from the preferred pilot: OpenCode 1.18.16, high effort, requested thinking budget 1,024 tokens, 8,192 output cap, 100 steps, 1,200-second agent limit, temperature 0 / seed 42. Strata 0.1.59, IQ3_S, INT8 KV, MTP4/min-p 0.5, full PLE, automatic full expert arena, 1,500 MiB VRAM reserve, default approximate expert-tail skip 7. Windows-native OpenCode edits a disposable frozen workspace and uses its offline Docker environment for shell commands and grading. Model-first startup and preserved socket-directory repair were retained. Personal configurations and installations are unchanged.

| Earlier / current run | Strict passes | Patch passes |
|---|---:|---:|
| Hermes, thinking off | 7/18 | 10/18 |
| Hermes, thinking on | 13/18 | 16/18 |
| OpenCode, 1K thinking | 16/18 | 18/18 |

This comparison does not isolate the harness: OpenCode also had larger step/time/output limits than Hermes. More patches passing does not establish full-expert equivalence, broader model performance, filled-64K retrieval, compaction/resume stability, or 128K readiness.

Official runtime stop returned ok:true/stopped; the owned Strata process was verified gone at completion. Later experiments are recorded separately in the linked comparison. Raw evidence: bench/results/strata-iq3-opencode18-20261110.

[Sanitized settings and measurements](results/strata-iq3-opencode18-20261110.json)
