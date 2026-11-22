# Long-context readiness (2026-11-22) - baseline 7d3e5b1 (bin-rp2), q8_0 KV, ctx 204800, ub 512, np 1, FA on
GPU time used: about 11 of 30 minutes (one session 647 s + 7 s graph dump). Swift restored and healthy after both.
Raw: pilot/results_ctx/ (ctx_results.json, *.toollog.jsonl, *.transcript.jsonl, server_ub512.log).

## Phase 2 results (4-turn sandboxed sessions: fix 2 bugs in 2 files with 7 hidden tests + rules, decoy retrieval, recall, contradictory rules)
| size (tokenizer count) | cold prefill | prefill tps | fix turn | hidden tests | rules | retrieval / recall / rules | warm TTFT (median) | cached follow-up TTFT | decode tps | VRAM loaded / peak |
|---|---|---|---|---|---|---|---|---|---|---|
| 127,671 tok (183 files) | 167.9 s | 767 | 218 s, 7 calls, 6 tools | 7/7 | ok | 3/3 | 1.44 s | 0.85-0.91 s | 35.8 | 20,542 / 20,570 MiB |
| 185,099 tok (258 files) | 285.7 s | 652 | 364 s, 6 calls, 5 tools | pass | ok | 3/3 (see json) | 1.85 s | about 1.15 s | 30.5 | 20,542 / 20,572 MiB |
Failures: none of model origin; one non-zero command exit in the 128K fix turn was a normal test failure before the fix (tool category cmd_nonzero), no harness failure, no truncated calls, no timeouts.
Cached prefix reuse worked (cache_first 131,680 of 131,734 tokens on the follow-ups).
Largest context qualified: 185K actual occupancy is reliable and interactive for agent turns (about 2 s per warm step, 30 tok/s decode); the only slow step is the one-off cold prefill (4.8 min at 185K, 2.8 min at 128K).
256K full depth: approved separately (30 GPU minutes, ub 256); see the 256K section once run.

## Phase 1: relbias-FA graph discrepancy (offline, CPU and one 7 s GPU dump; tool: graph_audit.py, dumps in diag/)
- The llama-level graph is NOT where the +122 comes from: reserve, real prefill and decode graphs all have the same node count (CPU 11,164, CUDA 11,740) for fused and unfused. 11,862 is the node count of the scheduler graph after split/copy insertion (sched_graph_realloc logs sched->graph), 122 nodes more than 11,740 added by ggml_backend_sched.
- The fused variant breaks topology equality between the prefill and decode graphs: same node count and op histogram, but the op SEQUENCE differs from node 1131 (PAD, bias MUL_MAT, descriptor CUSTOM placed before the Q-clip chain in prefill, after it in decode) and 280 nodes have different source lists (unfused: 0). The allocator plan reserved for one graph does not fit the other, which triggers the scheduler re-plans (backend ids unchanged, graph size unchanged). This is independent of CUDA graphs, which is why disabling them did not help.
- Live-range overlap check on the real graphs (fused and unfused): 0 overlaps in 7,214/7,220 tensors, so the visible plan is consistent; the corruption comes from hidden reads (kernel reading a tensor through another tensor pointer, scheduler split copies) that the dump cannot see. Exact corrupting access still not located.
- Regression test (deterministic, no GPU): build pp/tg/real graphs, require equal node count AND equal op sequence AND equal source index lists (graph_audit.py T1/T1b/T3), plus 0 live-range overlaps (T5), plus zero sched_graph_realloc in a CPU stress. Current 7d3e5b1: passes; fused variant: fails T1b/T3.
- Safe integration strategy: same op sequence in prefill and decode (emit identical nodes in the same positions, only change what they compute), real sources only, run the audit on CPU before any GPU test.
