# Strata IQ3_S: thinking-only 64K pilot

**Rank 1 — provisional best measured 64K coding-agent configuration.** Keep this thinking-on profile as the reference for subsequent Strata experiments. Ranking is based on this frozen six-task pilot, not a completed 18-attempt validation or a matched ranking against both 27B models. The memory reserve gate remains failed.

Thinking enabled: **6/6 strict passes and 6/6 held-out patch passes**, with no 40-turn caps or episode timeouts. The preserved thinking-off first trial on the same six tasks scored **2/6 strict and 4/6 patch passes**. This is a six-task pilot, not an 18-attempt validation or evidence of a general success rate.

Only `enable_thinking` changed in the inference/agent settings. Both runs retain the existing 1024-token reasoning-budget request field, 4096 output-token cap, 40-turn cap, deterministic sampling, pinned Hermes revision, offline oracle, and frozen task manifest. Backend configuration comparison differs only in its output log path. IQ3_S, Strata 0.1.59, INT8 KV, MTP4/min-p0.5, automatic arena, full PLE, 1500 MiB VRAM reserve and default approximate expert-tail skip7 remain unchanged. Reasoning content is present in the saved pilot histories; this was not just a requested flag.

| Frozen task | Baseline strict / patch (trial 1) | Thinking strict / patch | Thinking seconds | API calls |
|---|---|---|---:|---:|
| Requests 1142 | Fail / Pass | Pass / Pass | 179.8 | 26 |
| Requests 5414 | Fail / Fail | Pass / Pass | 94.7 | 15 |
| Pytest 10081 | Pass / Pass | Pass / Pass | 125.8 | 15 |
| Pytest 6202 | Pass / Pass | Pass / Pass | 104.8 | 17 |
| Sphinx 10323 | Fail / Pass | Pass / Pass | 407.3 | 38 |
| Sphinx 10449 | Fail / Fail | Pass / Pass | 396.9 | 37 |

The baseline failure histories include repeated identical terminal calls, investigation of failures also present on the unchanged repository, and unfinished exploration. Three of the original 18 attempts produced correct patches but hit the cap; the other eight strict failures also had failing patches. Raising the cap alone would not establish a fix. No stopping change was needed to obtain this pilot result.

| Measured value | Thinking pilot |
|---|---:|
| Successful tasks per episode-hour | 16.50 (matched baseline first trial: 7.00) |
| Total episode time | 1309.25 s (matched baseline: 1028.39 s) |
| Median episode latency | 152.81 s |
| Tool calls / error results | 153 / 18 (matched baseline: 202 calls) |
| Compactions | 0 |
| Actual maximum prompt | 42,699 tokens; capacity 65,536 |
| Accepted decode, <8K / 8–32K / 32–48K actual prompt | 106.13 / 105.26 / 103.32 tok/s |
| Fresh prefill, corresponding buckets | 1523.25 / 854.96 / 546.48 tok/s |
| GPU energy | 40.20 Wh |
| Mean / peak GPU power | 109.73 / 261.95 W |
| Peak sampled VRAM | 15.02 GiB |
| Minimum free VRAM | 613 MiB |
| Peak native process RSS | 47.84 GiB |
| Minimum Windows available physical RAM | 148.52 MiB |
| Peak GPU temperature | 55.82 C |

Accepted engine output counts include reasoning; these are not visible-answer-only tokens/s. Cache reuse is measured from native request counters, without adding duplicate episode usage. GPU energy is device-wide GPU-only, with 100% of the agent telemetry window covered and 95.58% coverage relative to campaign start; overlapping runtime telemetry is not added. CPU/wall energy is unavailable. RAM figures refer to the Windows host; RSS can include mapped/shared weights. Tool errors include failing commands/tests.

The original 8 GiB physical-memory reserve still fails. Expert-tail skipping remains approximate. No 128K promotion, filled-64K agent-quality gate, compaction/resume stability or full-expert parity is established. Task selection was frozen, but this is a sequential comparison rather than randomized interleaving; single-trial variability remains unknown.

Raw evidence: `bench/results/strata-iq3-thinking-pilot-20261110/` (`events.jsonl`, histories, request traces, reports, telemetry, configuration and source snapshot). The run completed at 2026-11-09 23:59:54 UTC; official runtime stop succeeded and no owned model remains loaded. Reviewed on 2026-11-10 after the inherited deadline of 06:47:15 UTC; no further GPU job was launched or budget reset. Existing 18-attempt baseline and saved base/Huihui comparison state are preserved.

Next experiment when a new run window is authorized: repeat this exact thinking-on configuration for the full frozen 18-attempt matrix. Keep stopping and memory settings fixed to establish repeatability before another variable changes.

[Ranked configuration and sanitized measurements](results/strata-iq3-thinking-pilot-20261110.json)
