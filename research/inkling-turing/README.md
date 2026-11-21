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
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo && cmake --build build -j
```
