# Coder at near-filled 64K context

All three requests used **63,971 actual input tokens** at 65,536 capacity. The repository-text fixture is exactly the same as the saved original IQ3_S filled-64K fixture. Thinking was OFF for this synthetic speed/retrieval probe; this is separate from the thinking-on coding campaign.

| Request | Accepted decode tok/s | Fresh prefill tok/s | Reused input | Output tokens | API seconds |
|---|---:|---:|---:|---:|---:|
| Cold | 119.8 | 3,840.7 | 0 | 575 | 21.51 |
| Cached 1 | 132.4 | 2,812.1 | 57,344 | 916 | 9.32 |
| Cached 2 | 145.4 | 2,886.7 | 57,344 | 651 | 6.81 |

Three retrieval markers at 10%, 50% and 90% depth were returned correctly in all three requests; all three generated implementations compiled. These checks do not execute the generated implementation or establish agent-task quality or full-context retrieval accuracy. The fixture repeats repository text as disclosed filler. Warm requests reuse a prefix and read the remaining 6,627 tokens; their prefill rate is for fresh tokens only.

Peak sampled VRAM was 14.52 GiB; native peak RSS was 30.54 GiB (may include shared/mapped pages); minimum available Windows physical RAM was 22.16 GiB during this short probe. Expert arena: 23.42 GiB. GPU-only device-wide energy: 2.56 Wh, mean 120.97 W / peak 269.19 W, peak temperature 57.27 C. All 76.19 sampled seconds were covered, including startup. CPU/wall energy is unavailable. Overlapping runtime/agent telemetry is not summed.

Settings: updated ISTA-DASLab GSQ-RCO Coder IQ1_M (256 experts/layer, 10 active; retained weights 3.5 bpw), Strata 0.1.59, INT8 KV, MTP4/min-p 0.5, full PLE, automatic full expert arena, 1,500 MiB reserve, default approximate expert-tail skip 7. The model's own 256-expert cache profile was used. No attention truncation or context reduction was added to fit. KV streams between GPU and host; this is a hybrid Strata run on an RTX 5070 Ti 16GB with 64GB host RAM.

Earlier original IQ3_S measured 102.3/131.9/132.5 decode tok/s and 3175.5/3281.6/3283.3 fresh prefill tok/s on the same fixture. There is only one cold request and two cached requests per model; generation lengths and cache reuse differ. These observations are not a general speed/quality equivalence claim or 128K promotion.

Official runtime stop was confirmed before the four thinking-on coding attempts resumed separately. Their validation is still in progress and is not included in this score. The earlier user-interrupted coding attempt remains ungraded and excluded, with its raw workspace/trace retained.

[Sanitized measurements](results/strata-coder-filled64k-20261110.json)
