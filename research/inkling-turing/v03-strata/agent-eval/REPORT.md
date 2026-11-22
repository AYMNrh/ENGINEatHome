# Inkling-Turing (e0be55a, UD-IQ3_XXS) vs Swift 1.5 - 4-task pilot (2026-11-22)
Harness pilot.py (own minimal agent loop, 6 tools, temp 0, seed 0, 135 s/task, 14 turns max). Results: results/*_results.json, per-turn data and transcripts in results/. GPU time: Swift 243 s (incl. 40 s cold load), Inkling 157 s (8 s load), about 6.7 min total.
Swift: production config via gateway (MTP3, Q4 KV, thinking forced on, xhigh). Inkling: llama-server bin-hk, q8_0 KV, 16K ctx, default template reasoning.
| task | Swift | Inkling |
| bugfix | pass 8/8, 1st-attempt, 17.9 s, 5 turns, 581 tok | pass 8/8, 1st-attempt, 15.7 s, 7 turns, 711 tok |
| feature | pass 7/7, retry needed, 45.2 s, 8 turns, 1727 tok | pass 7/7, 1st-attempt, 31.2 s, 9 turns, 1603 tok |
| multifile | pass 9/9 but hit 14-turn limit, 68.8 s, 2683 tok | pass 9/9, 1st-attempt, 45.9 s, 9 turns, 2421 tok |
| explore | correct, 70.5 s, 9 turns, 2536 tok | correct, 54.0 s, 8 turns, 2899 tok |
Totals: Swift 202 s / 7527 tok / 36 turns / 7 bad of 49 calls; Inkling 147 s / 7634 tok / 33 turns / 3 bad of 47 calls.
Decode tok/s: Swift 38-43 (MTP3 acceptance 5410/6282 = 86%), Inkling 56-57. Max prefill tok/s (turns with >=100 new tokens): Swift 340-430, Inkling 670-880. Median TTFT 0.38-0.47 s Swift, 0.52-0.54 s Inkling; first-turn TTFT 0.8-1.8 s vs 0.4-1.0 s.
Prefix reuse (cached/(cached+new) over turns 2+): Swift 0.92-0.98, Inkling 0.89-0.95.
