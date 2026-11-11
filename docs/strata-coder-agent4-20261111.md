# Coder: four matched 64K coding attempts

Completed 11 November 2026, 00:03 Paris. Strata runtime officially stopped afterward; no native Strata process remained when checked.

| Measurement | Expert-pruned Coder | Original IQ3_S / tail7 |
|---|---:|---:|
| Strict passes | 2/4 | 2/4 |
| Passing patches | 3/4 | 2/4 |
| Accepted decode, including reasoning | 107.32 tok/s | 103.34 tok/s |
| Fresh prefill | 1,181.92 tok/s | 812.93 tok/s |
| Maximum actual input | 31,539 tokens | 35,158 tokens |

Same four-case manifest: Pytest6202 three trials and Sphinx10323 one trial. OpenCode1.18.16/high effort/requested reasoning1024/output8192/max100steps/1200seconds/temp0/seed42; Strata0.1.59/INT8KV/MTP4/minp0.5/fullPLE/expert-tail skip7 (approximate), 65,536 context capacity. These are actual agent prompts, with no filler.

| Task | Time | Strict | Held-out patch |
|---|---:|---|---|
| Pytest trial1 | 20m03s | FAIL: timeout | FAIL: 6/7 checks |
| Sphinx | 5m01s | PASS | PASS: 21/21 checks |
| Pytest trial2 | 6m31s | PASS | PASS: 7/7 checks |
| Pytest trial3 | 20m04s | FAIL: timeout | PASS: 7/7 checks |

The last attempt produced a correct patch but did not finish within the agent limit. Raw traces show long baseline-test waits, including `sleep 100`; workflow latency remains a limitation. Historical preempted, ungraded attempts are preserved separately and excluded from these totals.

148 tool calls, 10 native tool error states, 6 shell-command failures and 2 shell timeouts were recorded. These counters have distinct scopes and must not be added together. Cache reuse was 96.07%; compaction count is unavailable.

Agent-only telemetry: 44.26 GPU Wh over 3,101.25 covered seconds (100% sampled coverage), mean51.37W/peak265.14W, peak55.61C, mean14.46% GPU utilization. Peak sampled VRAM15.12GiB, native peak RSS32.60GiB, minimum available Windows physical RAM14.33GiB. Energy is device-wide GPU-only, excluding CPU/wall energy; native RSS may contain shared or mapped pages. Parent runtime telemetry overlaps and was not added.

Four non-interleaved attempts cannot establish general quality superiority. Decode and prefill comparisons follow different generated trajectories, input lengths and cache histories. The Coder run retained more host RAM than the earlier original-model run; this does not validate longer-context agent quality, full attention parity or the complete promotion gates.

Raw evidence: `bench/results/strata-coder-agent-final-20261110/{status,comparison,events,runtime-stop}.json*`, per-case patches/grades/traces and `telemetry.jsonl`. Baseline: [original efficiency comparison](strata-iq3-efficiency-20261110.md). Next authorized experiment: focused-test/time-awareness/patch-preservation guidance as a separate controlled change; not part of these frozen results.
