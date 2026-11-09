# Huihui 64K expanded benchmark

Measured 2026-11-08 on RTX 5070 Ti 16GB and 64GB host RAM using the llama.cpp reference executor, build 12110 / cd959bae9. This measures the reference backend, not an independent QAH inference implementation.

Model: Huihui Qwen3.8-27B abliterated GSQ-RCO IQ3_S MTP, SHA-256 `eea0638e283433e27b0edbd409b552be467a75ecc777d91c51db554c0a644c19`.

Capacity 65,536 tokens; long prompts exactly 65,279 tokens. MTP depth 3, all 66/66 stored-layer/output placement slots offloaded, main q4_0 KV and draft q8_0 KV, batch/microbatch 512/128, GPU sampling, one slot, native Jinja template, thinking disabled, temperature zero and generation seed 42. Full-history dense attention and native recurrent layers; no context shifting or positional override. KV quantization is lossy. Input embeddings remain CPU mapped. Startup fit-margin gate passed; sampled device headroom subsequently became tight.

## Results

26 requests: six coding tasks twice, six distinct retrieval cases with six cached repeats, two forced decoding probes. Both repeated coding outputs were identical per task, so these repetitions measure timing rather than additional task diversity.

| Coding task (short prompt at 64K capacity) | Functional passes | Median output tok/s | Median HTTP seconds |
| --- | --- | --- | --- |
| Cache expiry fix | 2/2 | 118.51 | 2.46 |
| CSV multi-function edit | 2/2 | 112.08 | 1.87 |
| Lazy event merging | 2/2 | 93.07 | 7.07 |
| Interval merging | 2/2 | 103.04 | 3.65 |
| Dependency ordering | 0/2 | 96.45 (failed outputs) | 6.80 |
| JSON merge patch | 2/2 | 101.71 | 4.16 |

Total coding correctness: **10/12 executions, 5/6 distinct tasks**. Dependency ordering consumes prerequisite iterators during node discovery, then iterates them again when building edges; the one-shot iterator check fails. Its generated source also skips self-edges, so cycle handling is not fully correct. Do not count its output speed as completed coding performance. Generated Python was screened and checked using the retained subprocess validator; this is not a security sandbox.

| Near-full-context measurement | Result |
| --- | --- |
| Cold exact-key retrieval | 6/6 correct |
| Cached retrieval | 6/6 correct |
| Needle positions | 10%, 50%, 90%, each with two seeds |
| Cold retrieval HTTP median | 72.10 s |
| Cached retrieval HTTP median | 0.381 s |
| Actual cached input | 65,275 tokens per cached request |
| Forced 128-token decoding | 2/2 completed, no truncation |
| Decode output rates | 49.41 and 52.75 tok/s |
| Decode output median | **51.08 tok/s** |
| Decode HTTP elapsed, including cold prefill | 74.43 and 75.50 s |
| Decode prefill rates | 914.42 and 898.76 tok/s |
| Draft acceptance during decode | 78/143 and 83/130 |
| Device-wide sampled VRAM peak | 15,865 MiB (15.49 GiB) |
| Server peak resident memory | 20,687 MiB (20.20 GiB) |

The VRAM counter includes other applications, is sampled, and is not an exact engine allocation peak. Device free-memory samples reached only a few hundred MiB; no sustained-memory-stability claim is supported. Resident host memory can include mapped weights and cached recurrent checkpoints.

Retrieval uses repeated varied archive filler and a single exact key per prompt. Coding checks use short prompts, so they do not establish coding correctness over a filled 64K repository. Forced decoding explicitly ignores EOS for throughput and has no code-correctness score. Two decoding probes cannot establish reliable tail latency. The earlier single 84,992-capacity result of 56.21 tok/s was faster than this run; differences in prompt, MTP acceptance and machine state prevent a controlled context-scaling or regression claim. No MTP-off control was run here.

## Reproduction and evidence

```powershell
py -3.9 bench/compare.py --models huihui --contexts 65536 --depths 3 --trials 2 --extended-coding --needle-depths 0.1 0.5 0.9 --tasks coding retrieval decode --output bench/results/huihui-64k-expanded.jsonl
py -3.9 bench/summary.py bench/results/huihui-64k-expanded.jsonl --output bench/results/huihui-64k-expanded-summary.json
```

Local raw evidence: `bench/results/huihui-64k-expanded.jsonl`; grouped summary: `bench/results/huihui-64k-expanded-summary.json`. Run ID `47501660-226c-4e7f-b3a2-c12b344eceff`. Corresponding archived sources, full model audit, server log and generated modules are retained beside the results. All 26 requests have status `ok`; functional failures are separately retained as `quality_pass=false`. No publication or Git push was performed for this run.

Publication update: [sanitized evidence](../bench/evidence/huihui-64k-expanded.jsonl) and [summary](../bench/evidence/huihui-64k-expanded-summary.json) are included in ENGINEatHome. Private paths are replaced and duplicate response output removed; original measurement hashes remain historical provenance. [Publication hashes and scope](../bench/evidence/publication.json) identify the transformed artifacts. Full logs and original source snapshots remain local.
