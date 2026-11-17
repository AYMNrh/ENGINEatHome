# SenseNova-Loop-9B 256K quality evaluation

## Phase 1 · Exact retrieval

**6 of 6 exact matches** across all six requested positions. Actual prompts contained 261,630 to 261,632 tokens of a 262,144-token context (99.80%).

| Needle depth | Token offset (zero-based) | Actual input | Mode | Reused tokens | Expected and observed | Score | TTFT (s) | Prefill (tok/s) | Decode (tok/s) |
|---:|---:|---:|---|---:|---|---|---:|---:|---:|
| 5.00% | 13,082 | 261,630 | cold | 0 | `WAJH-ZX6C-PAGS` | Pass | 244.593 | 1,069.9 | 25.8 |
| 20.00% | 52,326 | 261,632 | cached | 260,602 | `YBHK-CHDJ-WMMG` | Pass | 1.954 | 542.8 | 26.7 |
| 40.00% | 104,653 | 261,630 | cached | 260,604 | `ZPUM-9ER2-5EFX` | Pass | 1.837 | 577.7 | 26.1 |
| 60.00% | 156,979 | 261,630 | cached | 260,604 | `T2SA-LA76-4KB6` | Pass | 1.827 | 579.2 | 26.2 |
| 80.00% | 209,306 | 261,631 | cached | 260,604 | `H65C-HXGE-7UR3` | Pass | 1.838 | 577.4 | 26.2 |
| 95.00% | 248,550 | 261,632 | cached | 260,604 | `FRGN-BN9J-3QP8` | Pass | 1.829 | 580.7 | 26.6 |

The cold request took 244.593 s to first token at 1,069.9 fresh-input tok/s and 25.8 decode tok/s. The five cached requests had median TTFT 1.837 s, 577.7 fresh-input tok/s, and 26.2 decode tok/s. Each cached request reused 260,604 tokens and evaluated 1,026 to 1,028 new tokens. Cached prefill throughput uses only freshly evaluated tokens. Decode times cover short exact answers.

## Method and failure analysis

- Model and runtime were SenseNova-Loop-9B Q4_K_M on the RTX 5070 Ti with Q8 K and V, microbatch 1,024, and the validated CUDA library. The library SHA-256 matched the preserved snapshot. The server reported 57/57 layers offloaded.
- Each request queried one approved shipment record in a shared synthetic archive, with same-shipment draft and revoked entries as distractors. The first request was cold and the five follow-ups requested prefix reuse. Depth varies across the six questions and cache mode differs between the first and later questions.
- Temperature was 0, seed was 42, and the model's chat template disabled thinking. The exact-match scorer trimmed surrounding whitespace. All responses matched, including the hyphens, and ended at EOS. A single synthetic archive and one query per depth give a screening result. Reliability on other corpora, multi-hop reasoning, and code understanding remains unmeasured.

## Recommendation

Advance to the multi-hop and repository screens after approval. The retrieval result warrants those screens. Use their results to decide whether to schedule a full coding-agent benchmark.

[JSON results](results.json) · [Questions, expected answers, and token suffixes](fixtures.json) · [Shared prompt text](shared-prefix.txt) · [Shared prefix token IDs](shared-prefix-tokens.json) · [Runner](phase1.py) · [Engine log](engine.log.gz)
