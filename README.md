# ENGINEatHome

**RTX 5070 Ti 16GB · 64GB DDR5 · Current focus: 64K**

Building a specialized local inference engine. Current measurements use Strata and llama.cpp; the independent transformer graph and CUDA kernels are not implemented yet.

[Setup & methodology](docs/benchmark-guide.md) · [Architecture](docs/architecture-audit.md) · [Memory model](docs/memory-model.md)

[![Coder filled-64K benchmark: actual input, cold and cached decode, fresh prefill, memory and GPU energy](docs/assets/coder-filled64k.png)](docs/strata-coder-filled64k-20261110.md)

[Full report](docs/strata-coder-filled64k-20261110.md) · [Measurements JSON](docs/results/strata-coder-filled64k-20261110.json)

[![Coder near-filled 128K and 256K: cold prefill, short-output decode, retrieval and memory](docs/assets/coder-long-context.png)](docs/strata-coder-long-context-20261110.md)

[Full report](docs/strata-coder-long-context-20261110.md) · [Measurements JSON](docs/results/strata-coder-long-context-20261110.json)

[![Coding-agent quality: strict and patch passes across the Hermes and OpenCode experiments, with harness limitations](docs/assets/coding-quality.png)](docs/strata-iq3-opencode18-20261110.md)

[OpenCode report](docs/strata-iq3-opencode18-20261110.md) · [Hermes report](docs/strata-iq3-thinking18-20261110.md) · [Measurements JSON](docs/results/strata-iq3-opencode18-20261110.json)

[![Corrected one-hour stability validation and matched expert-tail skipping comparison](docs/assets/stability-and-experts.png)](docs/strata-iq3-efficiency-20261110.md)

[Full report](docs/strata-iq3-efficiency-20261110.md) · [Measurements JSON](docs/results/strata-iq3-efficiency-20261110.json)

**Snapshot: 10 November 2026.** Coder coding validation is paused. The separate 128K/256K smoke tests passed; full agent-quality gates and context promotion remain unpassed. Original IQ3_S remains the provisional coding reference. Energy figures are GPU-only; approximate expert-tail skipping is disclosed on each card.

### Earlier experiments

[Huihui 64K](docs/huihui-64k.md) · [DSpine 8K](docs/dspine-results.md) · [Thinking-budget pilot](docs/strata-iq3-opencode-hard-20261110.md) · [Model comparison](docs/model-comparison.md)

### Reproduce and inspect

[Commands & methodology](docs/benchmark-guide.md) · [Evidence manifest](bench/evidence/publication.json) · [Card renderer](bench/render_cards.py)

Model weights, binaries, credentials and raw local traces are excluded from Git. Detailed results and limitations stay linked beside each card.
