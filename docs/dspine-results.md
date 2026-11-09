# Tencent DSpine with Huihui: first executable results

Measured 2026-11-09 on RTX 5070 Ti 16GB / 64GB host RAM, under WSL Ubuntu. DSpine3 improved the near-filled 8K decode probe by **20.0% versus MTP3** with all trunk layers on GPU. This is one probe per profile, not a statistically established or long-context improvement. No production profile was changed.

## Runtime and model

- Preferred target: Huihui Qwen3.8-27B abliterated GSQ-RCO IQ3_S MTP, SHA256 `eea0638e283433e27b0edbd409b552be467a75ecc777d91c51db554c0a644c19`.
- [Tencent's official drafter](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16): five layers, block 16, native 8192-token draft sliding window. It was trained for the stock target; this evaluation uses the abliterated target explicitly.
- [Community GGUF conversion](https://huggingface.co/aj9o9/Qwen3.8-27B-DSpine-b16-GGUF), revision `62e92414e114702e792ea7a4c37eba57b25bac8b`, Q4_K_M file 1,257,892,896 bytes, SHA256 `0c6c984ac18611288d7b8f497bc347c4921e8231b1b6abf28650481eaf35ff55`.
- [Community llama.cpp port](https://github.com/Ajay9o9/llama.cpp/tree/d493f841b20241606c47a4e0537d3c786b5cf50b), compiled locally with GCC 13.3 / CUDA 13.3 for sm120. Binary SHA256 `a9138f1984a16512bebf50de37134dfbe0dfafbb9c66023775c457bba1333248`.
- Port CLI is `--spec-type draft-dflash`; the loader detects DSpine tensors and logs `DSpine code bridge (code = 1024, rank = 512, top-k = 16)`. These results use the port, not Tencent's official BF16 SGLang runtime or an independent QAH compute implementation.
- Capacity 8192; retrieval/decode prompts exactly 7935 tokens. Main KV q4_0, draft KV q8_0, batch/microbatch 512/128, one slot, native Jinja template, thinking off, greedy sampling/seed 42, GPU sampling. Dense target attention retains full history; no context shift or positional override. KV quantization is lossy. The draft's native sliding window does not truncate target attention.

## All trunk layers on GPU: DSpine3 versus MTP3

Both profiles request `--gpu-layers 99`. Input embeddings remain CPU mapped. **DSpine's draft limit is reduced to three tokens**, rather than the model-card recommendation of 15, to preserve memory on this GPU. Its underlying drafter is still the block-16 DSpine checkpoint.

| Task | DSpine3 tok/s | MTP3 tok/s | Functional result, each |
| --- | ---: | ---: | --- |
| Cache expiry fix | 120.36 | 124.88 | Pass |
| CSV multi-function edit | 135.61 | 120.45 | Pass |
| Lazy event merge | 108.23 | 101.20 | Pass |
| Interval merge | 121.37 | 110.96 | Pass |
| Dependency order | 112.16 | 104.45 | Fail: one-shot prerequisite iterator |
| JSON merge patch | 119.64 | 107.69 | Pass |
| Forced 128-token decode near filled 8K | **95.76** | **79.80** | Both complete, no truncation; code quality unscored |

Coding prompts are short. All six coding outputs are byte-identical between these two full-GPU profiles, including the failing output. Four of the five passing coding tasks have higher backend output rates with DSpine in this single run. This does not establish general sampling/logit equivalence.

| Other measurement | DSpine3 | MTP3 |
| --- | ---: | ---: |
| Coding checks | 5/6 | 5/6 |
| Cold retrieval: 10%, 50%, 90% | 3/3 | 3/3 |
| Cached retrieval at those positions | 3/3 | 3/3 |
| Cold retrieval HTTP median | 5.695 s | 5.326 s |
| Cached retrieval HTTP median | 0.292 s | 0.231 s |
| Actual cached prompt tokens, each | 7931 | 7931 |
| Decode draft tokens accepted/proposed | 85/123 | 80/138 |
| Sampled device-wide peak VRAM | 15,571 MiB | 14,593 MiB |
| Device free immediately after load | 525 MiB | 1800 MiB |

DSpine costs 978 MiB more sampled peak VRAM here. The post-load reserve was barely above 512 MiB and is not guaranteed to persist during requests. Device counters include other applications and sampled peaks can miss transients. Linux process-RAM telemetry remains null; no RAM-efficiency claim is made. Retrieval HTTP latency was slower with DSpine despite its faster decode probe. These are separate measurements.

## Explicit partial CPU placement

All three profiles request `--gpu-layers 60`; logs show 60/66 placement with the same trunk placement. MTP additionally loads its stored predictor layer, whereas DSpine loads the external drafter. The initial trunk layers and their KV/recurrent buffers are on CPU; this profile differs from the full-GPU comparison above.

| Measurement | No speculation | MTP3 | DSpine7 |
| --- | ---: | ---: | ---: |
| Near-filled 8K decode tok/s | 20.03 | 18.48 | **29.09** |
| Coding checks | 5/6 | 5/6 | 5/6 |
| Retrieval checks, cold plus cached | 6/6 | 6/6 | 6/6 |
| Cold retrieval HTTP median | 9.650 s | 10.138 s | 10.070 s |
| Cached retrieval HTTP median | 0.870 s | 0.758 s | 0.551 s |
| Sampled device-wide peak VRAM | 13,055 MiB | 13,887 MiB | 15,574 MiB |

DSpine7 is 45.2% faster than no speculation and 57.4% faster than MTP3 on this single decode probe. Its coding output hashes differ from the no-speculation control on three of six tasks. Passing small functional checks does not prove arbitrary-output equivalence; numerical/near-tie behavior and the port's general exactness need independent investigation.

## Memory check and setup outcomes

The port's fitter cannot create the draft context independently and logs that it is fitting without the extra model. Its reported reserve therefore omits draft allocations. The harness now records this omission and checks actual device free memory after loading: unavailable counters or less than 512 MiB refuse before requests unless the existing explicit pressure override is selected. No pressure override was used for the completed runs. This post-load check cannot guarantee future workspace headroom.

The first 15-token/all-GPU attempt was cancelled during setup after this fitter omission was discovered. It has **no measured request or OOM result** and is excluded from completed-run summaries. One temporary-cache preparation retry also produced no request measurements. Linux-native temporary copies avoided slow Windows-mounted model hashing/loading during the completed tests; original weights and the existing Windows binary remain unchanged. Temporary copies were absent after the session, and no benchmark server remains running.

There is **no DSpine result at 64K, 128K or longer**, no full 15-token-block performance measurement, no repeated stability test and no long-context coding-quality claim. The 8K result cannot be extrapolated to the earlier 64K MTP benchmark. The six fixtures are synthetic; archive filler is varied but repeated, and each position uses one seed. Forced decode ignores EOS solely for throughput. The existing validator screens generated Python and runs it in a subprocess; it is not a security sandbox.

## Reproduction and local evidence

Build, model metadata and version logs are retained in ignored `build/dspine/`. Build command:

```bash
cmake -S build/dspine/llama.cpp -B build/dspine/cuda-build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build build/dspine/cuda-build --target llama-server -j 8
```

Run from the workspace inside Ubuntu; choose a Linux model directory containing the checksum-verified Huihui file. Temporary caches must be created and consumed in a continuous WSL session if `/tmp` is cleared between sessions.

```bash
python3 bench/compare.py --models huihui --model-dir /path/to/models \
  --server build/dspine/cuda-build/bin/llama-server \
  --dspine-draft build/dspine/Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf \
  --contexts 8192 --depths 3 --gpu-layers 99 --trials 1 --extended-coding \
  --needle-depths 0.1 0.5 0.9 --tasks coding retrieval decode \
  --output bench/results/huihui-dspine-8k-d3-g99.jsonl
```

For its MTP control omit `--dspine-draft`, preserving depth 3, and use a different output file. For partial CPU placement use GPU layers 60 and depths 0/3 for controls, or depth 7 with the DSpine draft. Each JSONL records exact argv, hashes, source snapshot, settings, outcomes and allocation logs.

Completed evidence: `bench/results/huihui-dspine-8k-d7-g60.jsonl` (13), `huihui-dspine-controls-8k-g60.jsonl` (26), `huihui-dspine-8k-d3-g99.jsonl` (13), `huihui-dspine-mtp-8k-g99.jsonl` (13): **65 completed requests**, 30 coding checks (25 passes), 30 retrieval checks (30 passes), five complete forced decode probes. Per-file summaries and `huihui-dspine-combined-summary.json` separate profiles. Cancelled setup remains in `huihui-dspine-8k-full.jsonl` and is excluded.

Publication update: ENGINEatHome includes [DSpine3 full-GPU evidence](../bench/evidence/huihui-dspine-8k-d3-g99.jsonl), [MTP3 full-GPU evidence](../bench/evidence/huihui-dspine-mtp-8k-g99.jsonl), [DSpine7 partial-CPU evidence](../bench/evidence/huihui-dspine-8k-d7-g60.jsonl), [partial-CPU controls](../bench/evidence/huihui-dspine-controls-8k-g60.jsonl), [combined summary](../bench/evidence/huihui-dspine-combined-summary.json), and the [cancelled setup outcome](../bench/evidence/huihui-dspine-8k-full.jsonl). Private paths are replaced and duplicate output removed; original provenance hashes are preserved. [Manifest](../bench/evidence/publication.json) hashes identify published bytes separately. Full logs, original source snapshots, generated execution directories, weights and binaries remain local.
