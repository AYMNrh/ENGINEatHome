# 64K OpenCode: corrected stability and expert-tail comparison

Corrected one-hour stability passed. Disabling approximate expert-tail skipping did not improve observed strict completion on this small coding sample. Agent full-suite baseline checks, long sleeps and source-fix restoration failures dominated the failures. Keep the tail7 profile as the provisional speed reference, with its approximation explicit; neither profile passes the complete 64K gates.

| Metric | Tail7 approximate | Tail0 skipping OFF |
|---|---:|---:|
| Strict passes | 2 | 1 |
| Patch passes | 2 | 2 |
| Accepted decode incl. reasoning (tok/s) | 103.3 | 81.9 |
| Fresh prefill (tok/s) | 812.9 | 900.9 |
| Maximum actual prompt | 35158 | 38400 |
| Median task seconds | 752.4 | 1203.6 |
| Strict passes/episode-hour | 2.42 | 0.92 |
| Coding-window GPU Wh | 44.55 | 50.27 |
| Tool errors | 8 | 9 |
| Shell timeouts | 1 | 3 |

Each profile ran Pytest6202 three times and Sphinx10323 once. Tail7: Pytest1/3 strict+patch, Sphinx1/1. Tail0: Pytest0/3 strict,1/3 patch; Sphinx1/1. All five strict failures reached the 1200-second agent limit. The successful Sphinx control took301.0s with tail7 and317.7s with tail0. These are sequential, non-interleaved trials with differing generated trajectories; timing and pass differences are descriptive, not general causal estimates.

Corrected stability:3717.7s,103review turns,557completed tools,11recorded compactions; explicit-compaction count increase, exact marker/invariant recall,389/389restart message IDs, real tool execution after both checkpoints, unchanged patch and21/21held-out tests passed. Native decode109.5tok/s/freshprefill1509.5,maxactual58712. GPU169.8Wh,164.5Wmean270.6Wpeak,68.6Cpeak,100%sampled-window power coverage. Original aborted soak is excluded from stability success and matched coding energy.

- Two issues, four attempts per profile; no general model ranking or full-expert equivalence established.
- Decode includes reasoning; fresh prefill excludes cached tokens. Prompt distributions differ.
- GPU-only device-wide energy, CPU/wall unavailable; child streams are sequential. Startup gaps excluded.
- Matched coding energy excludes both stability runs; whole campaign includes aborted invalid soak.
- Physical RAM reserve failed; native RSS can include mapped/shared pages. No 128K promotion.
- Stability is multi-turn review of an existing patch, not filled-64K retrieval or new coding throughput.
- All failures preserved. Timeout-unit and persistent-tool-permission harness bugs fixed; old soak invalid.

Model officially stopped after each stage; final native process absence verified. No new GPU stage or Git publication. Next proposed experiment is bounded focused-test/time-awareness/patch-preservation guidance; it has not run.

[Measurements](results/strata-iq3-efficiency-20261110.json)
