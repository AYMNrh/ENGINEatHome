# Swift 1.5 / 256K capacity with 128K input and large output headroom

RTX 2080 Ti 22 GB, Q4_0 target and draft KV, MTP3, Flash Attention, one sequence. All model weights including token embeddings and both caches are on CUDA. Batch 512, microbatch 128, draft microbatch 64. Thinking uses the checkpoint template with xhigh and reasoning-budget -1. Sampling is temperature 1, top-p 0.95, top-k 20, min-p 0, presence penalty 0, repeat penalty 1 and seed 42.

The initial rendered prompt has exactly 131,072 token IDs. The total output allowance is 130,816, with 256 tokens of additional allocation headroom. This output allowance covers both reasoning and final content. EOS is respected. The fixture asks for a complete Python log-processing CLI with validation, aggregation, tests and usage examples, following a synthetic archive.

| Request | Actual input | Reused / fresh | Output allowance | Generated | Prefill tok/s | Decode tok/s | TTFT s | End-to-end s | Natural stop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cold | 131,072 | 0 / 131,072 | 130,816 | 30,154 | 294.72 | 21.91 | 444.763 | 1820.91 | True |
| cached | 131,072 | 131,068 / 4 | 130,816 | 30,154 | 15.94 | 21.91 | 0.276 | 1376.45 | True |

## KV allocation and memory observations

Main KV reserves 4,608.00 MiB at initialization. Draft KV reserves 288.00 MiB.
Total logged KV is 4,896.00 MiB. The runtime sizes cache tensors for the configured capacity and allocates their buffers during context initialization. Filled positions increase as input and output are processed.

| Stage | Process CUDA MiB | Device used MiB | Device free MiB |
| --- | ---: | ---: | ---: |
| startup | 19,422 | 19,426 | 2,573 |
| after_short | 19,628 | 19,632 | 2,367 |
| after_tokenization | 19,628 | 19,632 | 2,367 |
| after_long | 19,666 | 19,670 | 2,329 |
| after_cached | 19,666 | 19,670 | 2,329 |
| after_tools | 19,668 | 19,672 | 2,327 |

Device memory is sampled and includes other GPU users. Process allocations are from nvidia-smi. Linux lacks a directly comparable WDDM shared-GPU-memory counter. Variation in workspace and graphs can change observed total VRAM despite a fixed KV reservation.

## Tool continuations

The model calls archive_lookup for B. The harness returns a deterministic record and digest. The model then calls verify_digest, receives a verification result, and must return all three original archive keys plus verified=true. These are local test tools. Every assistant message, including reasoning_content and tool_calls, is sent back with its corresponding tool response. Before each continuation, the rendered prompt is checked for exact prior reasoning text. A preserve_thinking=false control is also rendered to show whether the template switch matters.

| Turn | Actual input | max_tokens | Nonempty thinking | Previous traces retained | Finish |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | 131329 | 130,559 | True | 0/0 | tool_calls |
| 1 | 131684 | 130,204 | True | 1/1 | tool_calls |
| 2 | 131858 | 130,030 | True | 2/2 | stop |

Fresh reasoning across tool turns: True. Final archive/tool-result recall correct: True.

Preservation means the prior reasoning trace is present in the model input. The final recall check separately measures whether the model uses the archived facts and tool results correctly. This is one controlled conversation. The preserve_thinking=false control did not change these prompts. The embedded template keeps assistant reasoning after the last user query even with that option disabled, which covers this uninterrupted tool sequence. This result does not establish retention across a new user query or compaction.

Run state: complete. Error: None.
Original server restoration: {'status': 'ok'}.

## Reproduce

Temporarily unload the idle original server using its verified PID, then run from the benchmark workspace.

    PYTHONPATH=src python3 bench/swift_homelab.py --replace-pid <8080-pid> --live-audit live-audit --output headroom-fresh --headroom
    python3 bench/swift_headroom_report.py headroom-fresh

The controller restores the captured original command and verifies port 8080. Full prompt IDs, messages, model outputs, native timings, telemetry and tool prompt checks are retained in headroom/.
