# SenseNova-Loop-9B Q4_K_M on the RTX 5070 Ti

The selected runtime uses the GuideboardLabs llama.cpp fork with CUDA mHC routing and merge, Q8 K and V, FlashAttention, CUDA Graphs, and a 1,024-token microbatch. The filled 256K benchmark measured **1,111.3 prefill tok/s, 29.2 decode tok/s, and 14.84 GiB peak sampled device VRAM**. Phase 1 scored **6/6 exact retrieval matches** across 5% through 95% depth. The retrieval fixture uses a synthetic archive with one query at each depth.

## Reproduce the setup

- Windows 11 with WSL2 Ubuntu, NVIDIA driver 621.14, RTX 5070 Ti 16 GB and 64 GB host RAM. The build used WSL kernel 5.15.167.4, CUDA Toolkit 13.3.73, CMake 3.28.3, GCC 13.3.0, and Python 3.12.
- Runtime fork at `fad174ecd8bf22746a626c58e67e193f095aaa46`. Apply the CUDA mHC patch from this repository. CMake converts CUDA architecture `120` to `120a` in this build. The server uses 56 layer executions and 57/57 reported GPU layer placement including the output layer. [Runtime fork](https://github.com/GuideboardLabs/llama.cpp-sensenova-loop/tree/fad174ecd8bf22746a626c58e67e193f095aaa46) · [CUDA mHC patch](../bench/sensenova/evidence/sensenova-mhc-cuda.patch)
- Model file is 5,628,723,552 bytes. SHA-256 is `983a5899fe29c67a73953f40aa0219044aae80868f2de077f6adb004697b79da`. This is the text-only community GGUF of the final release. Hugging Face denied access to the official weights during setup. [Pinned GGUF](https://huggingface.co/guideboardlabs/SenseNova-Loop-9B-GGUF/tree/5cf786036589460866c676ab515bb7f4dd5790a4)

From the repository root in WSL, run the setup script against a fresh directory. It builds the selected runtime, verifies the GGUF, and makes a local baseline library snapshot. CUDA Toolkit defaults to `/usr/local/cuda-13.3` and can be specified with `CUDA_ROOT`. [Setup script](../bench/sensenova/setup.sh)

```bash
bash bench/sensenova/setup.sh
```

The archived Python runners expect the default WSL runtime directory `~/models/sensenova-runtime` and available local port 18931. For Phase 1, copy the archived runner into a fresh output directory, then invoke it with the CUDA library directory available.

```bash
mkdir sensenova-phase1-run
cp bench/sensenova/evidence/quality-256k-phase1/phase1.py sensenova-phase1-run/
LD_LIBRARY_PATH=/usr/local/cuda-13.3/targets/x86_64-linux/lib:/usr/lib/wsl/lib \
  python3 sensenova-phase1-run/phase1.py
```

The baseline server command used by the runners is:

```bash
LD_LIBRARY_PATH=/usr/local/cuda-13.3/targets/x86_64-linux/lib:/usr/lib/wsl/lib \
  ~/models/sensenova-runtime/build-cu133/bin/llama-server \
  -m ~/models/sensenova-runtime/SenseNova-Loop-9B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 18931 --device CUDA0 --fit off -lv 5 -ngl 99 -np 1 \
  -c 262144 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 1024 --no-warmup
```

## Speed and optimization

The original fork used CPU fallback for two mHC operations. The CUDA patch moved them to the GPU and passed a short 32-token greedy comparison. The first filled 64K probe moved from **1,109.2/21.9** to **2,426.9/48.6 tok/s** (prefill/decode). That comparison also changed microbatch from 512 to 1,024. Later matched tests isolate individual experiments. [Original speed result](../bench/sensenova/evidence/results.json) · [CUDA 64K result](../bench/sensenova/evidence/tuned-65536/results.json) · [Short comparison](../bench/sensenova/evidence/mhc-cuda/results.json)

The selected baseline was measured with two warmed, cold-prompt requests per context and 256 greedy output tokens per request.

| Capacity | Actual input | Median prefill (tok/s) | Median decode (tok/s) | Median TTFT (s) | Peak device VRAM (GiB) |
|---|---:|---:|---:|---:|---:|
| 64K | 65,024 | 2,422.5 | 48.3 | 26.9 | 8.13 |
| 128K | 130,560 | 1,735.1 | 39.7 | 75.3 | 10.36 |
| 256K | 261,632 | 1,111.3 | 29.2 | 235.5 | 14.84 |

The 64K sweep found that microbatch 2,048 improved prefill by 4.6% and changed 159 of 256 output IDs per repeat. Microbatches 256 and 512 also diverged. Standard-FP32 mHC changed 136 IDs per repeat and delivered no speed gain. The baseline remained selected. [Matched 64K sweep and context baselines](../bench/sensenova/evidence/optimization/summary.csv) · [Standard-FP32 mHC patch](../bench/sensenova/evidence/optimization/mhc-fp32.patch)

At 256K, both the matched baseline and fused Q4_K MMQ J=64 completed three cold and three cached requests with all 256 output IDs matching the frozen output on every request. The baseline median cold latency was 241.41 s and J=64 took 244.14 s. This matched campaign collected top-two log probabilities. The Q8 conversion experiment was stopped after one completed cold and cached pair, both exact, so it has a partial result. [Matched 256K baseline](../bench/sensenova/evidence/optimization-256k/matched-baseline/results.json) · [MMQ J=64](../bench/sensenova/evidence/optimization-256k/mmq-j64/results.json) · [Partial Q8 conversion](../bench/sensenova/evidence/optimization-256k/q8-convert-warp4/results.json)

The 256K Nsight Systems trace attributed **170.05 s** of summed GPU kernel time to the dominant FlashAttention specialization across **3,612 launches**. The last two quarters of prefill accounted for about 76% of that kernel's time. Nsight Compute returned `ERR_NVGPUCTRPERM`, leaving achieved occupancy and DRAM bandwidth unmeasured. The source review identified K/V reuse, register pressure, and Q8 staging as possible areas for improvement. [Profiling report](../bench/sensenova/evidence/optimization-256k/profile-report.md) · [Top kernels](../bench/sensenova/evidence/optimization-256k/nsys-top5.csv)

The early FP16 packing experiment compiled the original and patched SM120a specialization with identical flags. Both used **246 allocated registers**, **217 peak live general-purpose registers**, a **16-byte stack frame**, and **zero compiler-reported spills**. Executable SASS was identical. The experiment was rejected and the branch was restored. [Patch](../bench/sensenova/evidence/optimization-256k/early-pack-compile/early-pack.patch) · [Original compiler evidence](../bench/sensenova/evidence/optimization-256k/early-pack-compile/original-resources.txt) · [Patched compiler evidence](../bench/sensenova/evidence/optimization-256k/early-pack-compile/patched-resources.txt)

## Phase 1 retrieval

The six prompts used **261,630–261,632 actual tokens**, or **99.80%** of capacity. All six approved references matched exactly despite draft and revoked distractors. Temperature was 0 and seed was 42.

| Mode | Exact matches | TTFT (s) | Fresh prefill (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|---:|
| Cold | 1/1 | 244.59 | 1,069.9 | 25.8 |
| Cached median | 5/5 | 1.84 | 577.7 | 26.2 |

Cached requests each reused 260,604 tokens. Multi-hop reasoning and code understanding remain unmeasured. Run those screens before deciding on a full coding-agent benchmark. [Phase 1 report](../bench/sensenova/evidence/quality-256k-phase1/report.md) · [JSON results](../bench/sensenova/evidence/quality-256k-phase1/results.json) · [Fixtures](../bench/sensenova/evidence/quality-256k-phase1/fixtures.json)

## Evidence archive

The archive includes raw Nsight Systems reports, CSV exports, frozen token inputs and outputs, prompts, scores, benchmark runners, historical orchestration scripts, patches, and compiler and engine logs. Historical scripts record the original workstation layout and experimental steps. Engine and build logs and SASS dumps are gzip-compressed. Nsight Systems can export the included `.nsys-rep` files to SQLite with `nsys export --type sqlite`. Model weights, generated object files, and duplicate SQLite trace exports are external to the Git archive. [Evidence manifest with SHA-256 hashes](../bench/sensenova/evidence-manifest.json)
