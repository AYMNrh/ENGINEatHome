# Final Swift routing comparison — stop, retain stock

**Verdict:** the default IQ4-only routing patch gained **0.8834%** median decode
throughput in the preregistered short screen, below the **5%** stop/go threshold.
No more routing experiments, new filled-context evaluation, or expensive patch
quality evaluation were launched. Stock Swift remains the validated baseline.

## Frozen protocol and stock reproducibility

- RTX 2080 Ti 22GB, 175 W cap throughout; all weights and Q4 KV on GPU.
- Swift 1.5 Qwen3.8-27B GSQ-RCO IQ3_S MTP checkpoint SHA256:
  `9aecf1cd41b2cb2f32a74e0d889e33855ebef43b26f43b43feb5720239e677e5`.
- llama.cpp revision `98d67c8d536dc584897c4adb2510509796517222`;
  published pinned routing diff is the sole experimental code change.
- 131,072 capacity, Flash Attention, CUDA Graphs, MTP3, batch 512, microbatch 128.
- Seed 42, temperature 1.0, top-p .95, top-k 20, xhigh thinking,
  no separate reasoning-token budget. Core runtime arguments were copied from
  the passing reference; binary/library paths, ports and aliases differ as needed.
- The existing v4 dependency fixture was frozen, including rendered prompt,
  165 input IDs and five held-out cases. A fresh unchanged stock replay passed
  **5/5**, finished naturally at **6,698 output tokens**, and matched the previous
  passing run's completion length. This establishes two passing observations of
  the same fixture; it does not attribute earlier progress to any one of the
  three changes that originally created v4.

## Short performance screen

**Actual input: 4,096 tokens. Allocated capacity: 131,072.** These figures are
not newly filled-128K performance. Both profiles used identical prompt IDs,
sampling, seed, warmup, 256-token timing windows and runtime settings.
Pair order was stock/patch, patch/stock, stock/patch. Natural EOS was honored;
all six measured outputs reached 256 tokens. Pre-sampling numerical capture ran
after timing, so its synchronization overhead is excluded from throughput.

| Metric | Stock | IQ4-only routing, threshold 2 |
| --- | ---: | ---: |
| Median decode | 35.2009 tok/s | 35.5119 tok/s |
| Median cold prefix prefill | 527.8771 tok/s | 526.9727 tok/s |
| Median cached client TTFT | 0.1815 s | 0.1775 s |
| Median measured request wall time | 7.4262 s | 7.3594 s |
| Sampled peak device VRAM | 15.7715 GiB | 15.7734 GiB |

Paired decode gains were **1.0362%, 0.8657%, 0.8285%**. All pairs favored the
patch slightly, but the **0.8834% median gain** did not meet the predefined
meaningful-gain requirement. This is a small observed gain, not a claim that
the difference is necessarily random noise.

The conditional 129,024-input-token phase and expensive patch coding-quality
test were **skipped by design**. No new filled-128K conclusion or patch-quality
pass is claimed. Earlier filled-128K results remain separate historical evidence.

## Numerical observations

The short-screen greedy traces matched **32/32 token IDs**, and probability
instrumentation reproduced each profile's uninstrumented trace. At the identical
teacher-forced eight-token prefix, both profiles selected the same next token.
Top-two relative-logit margins were **3.60235** stock and **3.94577** patched;
the largest difference among 32 shared top-token log probabilities was **0.62152**.

These are pre-sampling log probabilities, equivalent to logits up to a shared
normalization constant, **not raw absolute logits**. The one-token teacher-forced
probe also differs in evaluation batch from MTP verification. Matching output
does not establish numerical identity, and this does not rule out every
implementation bug. With no useful speed gain, further numerical investigation
of this patch is stopped rather than expanded.

## What we learned

1. Stock supports the measured long-context retrieval workload. Token drift alone
   should not be treated as task-quality failure.
2. The earlier shared coding failure combined an ambiguous fixture, greedy
   sampling and a 16K total-output limit. The revised fixture succeeds, but the
   earlier three simultaneous changes do not establish a single causal fix.
3. Freeze and reproduce a passing evaluator before expanding an experiment matrix.
   A cheap screening gate prevented another costly, low-value sweep here.

## Next recommendation — not implemented

Profile stock cold prefill, then consider a narrowly scoped microbatch comparison
of **128 versus 256/512**. Prior filled-128K cold TTFT is about **461 seconds**, versus
about **22 seconds** for 512-token decoding: roughly **95%** of that cold request
is prefill, with approximately **5.7 GiB** free VRAM. Larger microbatches could
amortize launches and improve matrix utilization, but the benefit is unmeasured
and must retain matched quality checks.

This targets **cold-start latency**, not necessarily warm coding-agent throughput;
cached TTFT is already about 0.27 seconds. No microbatch change, new profiling
run, kernel optimization, or broader benchmark was implemented.

## Evidence and shutdown

Homelab: `/home/marvin/llm/experiments/results/final-stock-routing-128k-v1/attempt-1-1795097614455775842/`.
The attempt contains preregistration, frozen source checksum, JSON, CSV, prompts,
native responses, telemetry, runtime arguments, numerical captures and seven
successful server-shutdown records. No GPU inference process remained afterward;
the 175 W cap remained set. Original source/builds/results were preserved.

Local copies: `bench/results/final-stock-routing-128k-v1/attempt-1-1795097614455775842/`.
At comparison completion the executor's approved scope contained only this
completed comparison and the hourly supervisor was report-only. Both benchmark
scheduling paths were subsequently disabled for the [production serving
deployment](production-serving.md); their files and results are preserved.
Nine targeted automation checks passed (two new).
