# PLAN / status (2026-11-24)
Done: inventory (GPU idle, 175 W, 580.178.04, nsys/ncu present, CUDA 12.4), baseline D1, profile D2, FAGQA mode comparison D3, parity probe D4.
Prior rounds (research/inkling-turing) already tried and rejected: IQ3 dequant variants, grouped-expert verify, FA tile widths, FA relbias fusion, DSpark (drafter does not load). Not repeated.
Open, not done: (1) pipelined/prefetching GQA decode kernel (target 300+ GB/s at 128K, est. +10-15%), (2) larger parity + 64K retrieval tests for FAGQA=2, (3) 242K baseline refresh, (4) cache multi-client contamination tests (nothing changed in caching this session).
Not covered this session: Phase 6 full qualification, CUDA-graph and prefix-cache studies (see prior docs), clean-build repro.
Verdict: no production change. FAGQA=2 is a measured +3-5% opt-in candidate, unqualified.
