# Inkling 2 Mini custom CUDA kernels for the RTX 2080 Ti (sm_75)

Start with **HANDOVER.md**. It covers the best configuration, results, bottlenecks, rejected approaches and next steps.

## Contents
- `patches/`: branch `turing-iq3-opt` as a `git format-patch` series on top of `realugbun/llama.cpp@inkling2-speed` commit 7669d89. `SERIES.txt` lists the commits.
- `THEORY.md`: ceilings and gap analysis.
- `progress.md`: the live log.
- `ledger.jsonl`: every experiment, including rejected ones.
- `scripts/`: patch generators, benchmark, profiling and equivalence-check scripts.

## Rebuild
```
git clone https://github.com/realugbun/llama.cpp -b inkling2-speed && cd llama.cpp && git checkout -b turing-iq3-opt 7669d89
git am ../patches/*.patch
```
The first 18 patches give `e0be55a`; `0019-relbias-node-e4a7ea6.patch` gives `e4a7ea6` and `0020-relpos-gpu-7d3e5b1.patch` gives the validated runtime `7d3e5b1`. `LLAMA_INK_RELBIAS_NODE=0` selects the original unfused chain.

Build command:
```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo && cmake --build build -j
```

## Long-context qualification and graph audit
`v06-longctx/` holds the 128K to 242K evaluation harness (`pilot_ctx.py`), raw results, `graph_audit.py` (the topology regression gate for any attention fusion) and the report. The publication summary is `docs/inkling-turing-20261122.md` in the repository root; cards are regenerated from `docs/results/inkling-turing-20261122.json` by `bench/render_cards.py`.
