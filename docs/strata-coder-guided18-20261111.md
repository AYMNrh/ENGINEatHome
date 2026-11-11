# Guided Coder: 18-attempt 64K validation

**18/18 strict passes and 18/18 independently graded patches.** Six selected frozen SWE-bench Verified issues, three attempts each; not a full SWE-bench score. The separate four-task guidance pilot passed4/4 and is not added to these totals.

Same Coder inference settings as the saved unguided four-task baseline: Strata0.1.59, INT8KV, MTP4/minp0.5, fullPLE, full256-expert arena, reserve1500MiB and approximate expert-tail skip7. OpenCode1.18.16/high effort/requested thinking1024/output8192/max100steps/1200seconds/temp0/seed42. Guidance adds focused tests, explicit time budgeting, short waits and final patch preservation; no scored attempts were retried.

| Measurement | Result |
|---|---:|
| Strict passes / held-out patch passes | 18/18 / 18/18 |
| Agent timeouts / shell timeouts | 0 / 0 |
| Total task time / median task latency | 73.96min / 202.29s |
| Accepted decode, including reasoning | 110.31tok/s |
| Fresh prefill | 1,265.47tok/s |
| Maximum actual input / configured capacity | 43,633 / 65,536 tokens |
| Prompt cache reuse | 96.41% |
| Tool calls / native tool error states | 566 / 40 |
| Shell command failures | 23 |
| GPU energy / mean / peak power | 117.55Wh / 95.31W / 266.98W |
| Peak temperature / mean GPU utilization | 58.81C / 42.21% |
| Peak sampled VRAM / native RSS | 15.27GiB / 32.00GiB |
| Minimum available Windows physical RAM | 13.16GiB |

All4,440.03 sampled agent seconds have GPU-power coverage. Energy is device-wide GPU-only; CPU/wall energy is unavailable. Native RSS can include mapped/shared pages. Tool errors and shell failures have distinct scopes and are not added together. Compaction count for these coding attempts is unavailable.

These prompts contain actual agent work without filler; this is not filled64K retrieval validation. Historical original IQ3_S OpenCode18 had16 strict/18 patch passes but different model/quantization and a known historical timeout-unit issue. The new guidance-only pilot is a closer comparison: baseline2/4 strict and3/4 patch in51.65min versus guided4/4 in16.01min, with exact model configuration matching apart from log location. Small non-interleaved experiments do not establish a general quality gain.

The 18-task runtime officially unloaded before the separately authorized one-hour Coder stability stage began. Stability and128K agent work are not included in these results;128K remains gated. All evidence is local under `bench/results/strata-coder-workflow-20261111/guided18`, including config, definitions, traces, patches, grades, request timings, telemetry and runtime-stop.
