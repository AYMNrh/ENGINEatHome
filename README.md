# ENGINEatHome

**RTX 5070 Ti 16GB + 64GB RAM · RTX 2080 Ti 22GB + 32GB RAM · Local coding-agent benchmarks**

Local inference benchmarking and private serving with Strata and llama.cpp. The independent transformer graph and CUDA kernels remain future work. Click a section title to expand or collapse it.

**Current winner for our deployment: stock Swift 1.5 Qwen3.8-27B GSQ-RCO IQ3_S MTP.** Selected on RTX 2080 Ti 22GB with 256K context, Q4 KV, MTP3, Flash Attention, xhigh thinking and a 175 W limit. This is our serving recommendation; historical scores and failures remain unchanged. Inkling-Turing is a separately validated research engine, not the production winner.

## Results at a glance

| Model / experiment | Headline measurement | Role | Report |
|---|---|---|---|
| **Swift 1.5 Qwen3.8-27B IQ3_S MTP3** (2080 Ti) | 260,096 actual input in a 262,144 window, 19.31 tok/s decode, 4-task coding set 3/4 (4/4 on RTX 5070 Ti) | **Production** | [report](docs/swift-2080ti-20261119.md) |
| **Qwen3.8-27B-OBLITERATED Q4_K_M, MTP3** (2080 Ti, 64K) | 37.8 tok/s decode and 408.5 tok/s prefill at 63,591 tokens, MTP acceptance 90%, cached follow-up 0.51 s vs 158 s cold, 0/400 cross-client leaks | Selectable profile (Swift stays default) | [report](docs/obliterated-profile-and-ram-cache.md) |
| **Inkling-Turing** (Inkling 2 Mini IQ3_XXS, 2080 Ti) | 242,134 actual input in a 262,144 window, cold prefill 556 s, 26.8 tok/s decode, 7/7 hidden tests, 3/3 follow-up checks | Research engine | [report](docs/inkling-turing-20261122.md) |
| SenseNova-Loop-9B Q4_K_M | 256K retrieval 6/6, multi-hop 1/4 content, 0/4 strict JSON | Evaluated | [report](docs/sensenova-loop-9b.md) |
| Qwen3.8-Flash-Next IQ3_S, Strata 0.1.60 | 128K: 4K prefill chunks give 2.33x the cold prefill throughput of 1K chunks | Evaluated | [report](docs/strata-iq3s-128k-prefill-20261117.md) |
| Bonsai DSH 256K | seven 18-attempt configurations compared | Evaluated | [report](docs/bonsai-dsh-256k-20261117.md) |
| Qwen3.6 hardest task | 64K, 92K and 128K capacity compared | Evaluated | [report](docs/qwen36-hardest-context-20261114.md) |
| Dense Qwen3.8-27B IQ3_S / DFlash2 | up to 84.8 tok/s at filled 64K; DFlash2 depth 7 above 100 tok/s | Evaluated | [report](docs/dense-iq3s-20261112.md) |
| Strata Coder | guided 18/18 six-issue result; 128K capacity six strict passes (55,948 maximum actual input); one-hour 64K stability gate passed | Evaluated | [report](docs/strata-coder-guided18-20261111.md) |
| Edge0-145B Q3_K at 16K | 0/18 strict coding passes, 2/18 held-out patch passes | Evaluated | [report](docs/edge0-145b-16k-quality-20261114.md) |

Numbers come from different hardware, runtimes and harnesses; compare them only through the linked reports. Details, cards and raw results are in the collapsible sections below.

<details open>
<summary><b>Swift 1.5 Qwen3.8-27B: selected production backend</b> · 19.31 tok/s at 256K · 3/4 coding on RTX 2080 Ti</summary>

### RTX 2080 Ti 22GB, MTP3, Q4 KV, filled context

| Capacity | Actual input | Prefill tok/s | Sustained decode tok/s | Peak VRAM |
|---|---|---|---|---|
| 131,072 | 129,024 | 296.7 | 24.64 | 15.77 GiB |
| 196,608 | 194,560 | 235.8 | 17.35 | 17.49 GiB |
| 262,144 | 260,096 | 195.7 | 19.31 | 19.21 GiB |

Retrieval 6/6 at every capacity (cold and cached). Source: [measurements JSON](docs/results/swift-2080ti-overview-20261119.json).

| Coding set (4 attempts, 64K, xhigh) | Strict passes | Task time |
|---|---|---|
| RTX 2080 Ti 22GB | 3/4 | 57.8 min |
| RTX 5070 Ti 16GB | 4/4 | 34.8 min |

[![Swift 1.5 selected production winner: filled-context prefill and sustained decode](docs/assets/swift-selected-winner.png)](docs/swift-selected-20261119.md)

[![Final stock versus routing comparison: only 0.88 percent decode improvement, stock retained](docs/assets/swift-stock-routing-verdict.png)](docs/swift-final-routing-2080ti-20261119.md)

[![Swift private production API: lifecycle protection, crash recovery, metadata-only analytics and Home Assistant](docs/assets/swift-production-serving.png)](docs/production-serving.md)

[Latest recommendation and evidence](docs/swift-selected-20261119.md) · [Production setup and clients](docs/production-serving.md) · [Latest measurements JSON](docs/results/swift-serving-20261119.json) · [Serving source](serving/)

**Current selection: 19 November 2026.** Stock Swift 1.5 is our production winner. Its four-attempt coding set scored **4/4** on RTX 5070 Ti and **3/4** on RTX 2080 Ti. Guided Strata Coder retains the broader **18/18** six-issue result; Swift has not completed an equivalent 18-attempt matrix. The earlier Qwen3.6 Sphinx #10449 result remains a distinct hardest-task observation. Hardware, runtimes and harness details are linked beside each card.

<details>
<summary>RTX 2080 Ti benchmark cards</summary>

[![RTX 2080 Ti 22GB Swift 1.5 filled 128K, 192K and 256K MTP benchmarks](docs/assets/swift-2080ti-filled-mtp.png)](docs/swift-2080ti-20261119.md)

[![RTX 2080 Ti 22GB 256K capacity, 128K input and natural long generation](docs/assets/swift-2080ti-headroom.png)](docs/swift-headroom-homelab-20261118.md)

[RTX 2080 Ti report and all measurements](docs/swift-2080ti-20261119.md) · [Coding comparison card](docs/assets/swift-2080ti-coding.png) · [Raw evidence](bench/swift/evidence-manifest.json)

[![Swift 1.5 Qwen3.8-27B IQ3_S scored 4/4 on the matched 64K coding set](docs/assets/swift-qwen-coding.png)](docs/swift-qwen-20261118.md)

[Swift inference card](docs/assets/swift-qwen-inference.png) · [Benchmark report](docs/swift-qwen-20261118.md) · [Measurements JSON](docs/results/swift-qwen-20261118.json)

</details>

</details>

<details open>
<summary><b>Qwen3.8-27B-OBLITERATED Q4_K_M and RAM prompt cache</b> · 37.8 tok/s at 64K · cached follow-up 0.51 s</summary>

[![Obliterated Q4_K_M 64K profile: F16 versus Q8_0 KV decode, prefill and VRAM](docs/assets/obliterated-profile.png)](docs/obliterated-profile-and-ram-cache.md)

[![RAM prompt cache: cold versus cached restore times for Obliterated and Swift](docs/assets/obliterated-ram-cache.png)](docs/obliterated-profile-and-ram-cache.md)

[![Qualification: 0 of 400 contaminated responses, OpenCode task, recovery and switching](docs/assets/obliterated-qualification.png)](docs/obliterated-profile-and-ram-cache.md)

Selectable in Home Assistant (select never switches, Apply is explicit); Swift remains the default and Obliterated never auto-starts. 65,536 context only, F16 KV, Q8_0 validated fallback. Swift now runs `--cache-ram 8192 --no-cache-idle-slots`. The upstream isolation bug (llama.cpp #27148) is unfixed: 0 of 400 is evidence, not proof, and the Inkling profiles are unchanged.

[Report](docs/obliterated-profile-and-ram-cache.md) · [Measurements JSON](docs/results/obliterated-20261123.json) · [Research evidence](research/obliterated-q4km-2080ti/README.md)

</details>

<details open>
<summary><b>Inkling-Turing: long-context CUDA engineering (research engine)</b> · 242,134 actual input · 7/7 hidden tests · 3/3 checks</summary>

Inkling 2 Mini (UD-IQ3_XXS) on the RTX 2080 Ti 22 GB at 175 W, Q8_0 KV, with a llama.cpp fork carrying sm_75 kernels and a GPU-side relative-position path. Validated runtime `7d3e5b1`; fallbacks `e4a7ea6` and `e0be55a`. Swift 1.5 remains the selected production backend.

### Long-context qualification (runtime 7d3e5b1, Q8_0 KV)

| Run | Capacity | ubatch | Actual input (snapshot) | Context after session | Cold prefill | Prefill tok/s | Decode tok/s | Peak VRAM | Hidden tests | Follow-ups |
|---|---|---|---|---|---|---|---|---|---|---|
| 128K | 204,800 | 512 | 127,671 | 132,416 | 167.9 s | 766.7 | 35.8 | 20,570 MiB | 7/7 | 3/3 |
| 185K | 204,800 | 512 | 185,099 | 189,949 | 285.7 s | 652.5 | 30.5 | 20,572 MiB | 7/7 | 3/3 |
| 128K | 262,144 | 256 | 126,908 | 131,783 | 220.3 s | 583.7 | 36.3 | 19,852 MiB | 7/7 | 3/3 |
| **242K** | 262,144 | 256 | **242,134** | 246,714 | **556.1 s** | 437.3 | 26.8 | **19,846 MiB** | 7/7 | 3/3 |

| Cache behaviour at 242K | Value |
|---|---|
| Cold prefill (empty cache) | 556.1 s (9.3 min) |
| Cached follow-up TTFT | 1.24 to 1.34 s |
| Warm TTFT during the fix turn (median) | 2.49 s |
| Prefix reuse | 246,015 of 246,069 tokens |
| Scheduler re-plans | 0 |

| Memory engineering | Before (e4a7ea6) | After (7d3e5b1) |
|---|---|---|
| Host upload per 32K prefill | 2,333 MB | 13.2 MB |
| CPU fill per 32K prefill | 1,288.5 ms | 2.0 ms |
| Loaded VRAM, 256K at ub256 | 20,080 MiB | 19,824 MiB |

**Measured limits.** Largest qualified actual input is **242,134 snapshot tokens** (246,714 in context after the session) in a 262,144-token window at ub256; it is **not** a 256K actual-input run. ub512 does not load at 262,144.

**Limitations.** One successful run per context size, so this is a qualification and not a reliability estimate; a controlled synthetic-repository task, not SWE-bench; cold prefill at 242K takes 9.3 minutes, so interactive use needs a warm prefix cache; the ub512 and ub256 runs differ in capacity, seed and content.

**What was done.** The relative-position bias became one graph node and was then computed from cell and token positions on the GPU, removing the host uploads and making 256K fit at ub256 (the unfused build did not fit). A Flash Attention fusion of the bias was **rejected**: unit tests were bit-exact but model runs were unstable, and a graph audit shows prefill and decode op sequences diverge. Patches, the audit and the regression gate are published.

[![Inkling-Turing overview: 242,134 actual input tokens in a 262,144-token window, 7/7 hidden tests, 3/3 follow-up checks](docs/assets/inkling-turing-overview.png)](docs/inkling-turing-20261122.md)

<details>
<summary>All Inkling-Turing cards</summary>

[![Inkling-Turing long-context decode: 35.8, 30.5 and 26.8 tok/s at 128K, 185K and 242K](docs/assets/inkling-turing-decode.png)](docs/inkling-turing-20261122.md#results)

[![Inkling-Turing cold prefill throughput by context size and microbatch](docs/assets/inkling-turing-prefill.png)](docs/inkling-turing-20261122.md#microbatch-effect)

[![Inkling-Turing context intelligence: two-file fix, retrieval, recall and contradictory rules](docs/assets/inkling-turing-intelligence.png)](docs/inkling-turing-20261122.md#what-was-measured)

[![Inkling-Turing memory engineering: host uploads, VRAM and the rejected Flash Attention fusion](docs/assets/inkling-turing-memory.png)](docs/inkling-turing-20261122.md#memory-engineering)

[![Inkling-Turing cache and agent responsiveness: cold 9.3 minutes, cached follow-ups about one second](docs/assets/inkling-turing-cache.png)](docs/inkling-turing-20261122.md#cache-and-responsiveness)

</details>

[Full report and methodology](docs/inkling-turing-20261122.md) · [Measurements JSON](docs/results/inkling-turing-20261122.json) · [Raw results and logs](research/inkling-turing/v06-longctx/results/) · [Long-context readiness report](research/inkling-turing/v06-longctx/LONGCTX_READINESS.md) · [Research bundle, patches and rebuild](research/inkling-turing/README.md) · [Rejected experiments](research/inkling-turing/rejected/) · [Graph audit](research/inkling-turing/v06-longctx/graph_audit.py)

</details>

<details>
<summary><b>SenseNova-Loop-9B Q4_K_M</b> · 256K retrieval 6/6 · multi-hop 1/4 content, 0/4 strict JSON</summary>

| Check | Result |
|---|---|
| 256K retrieval | 6/6 |
| Multi-hop reasoning, content | 1/4 |
| Multi-hop reasoning, strict JSON | 0/4 |

[SenseNova-Loop-9B Q4_K_M report](docs/sensenova-loop-9b.md) · [Quality card](docs/assets/sensenova-phase2-quality.png) · [Timing card](docs/assets/sensenova-phase2-timing.png)

</details>

<details>
<summary><b>Qwen3.8-Flash-Next GSQ-RCO IQ3_S on Strata 0.1.60</b> · 4K prefill chunks give 2.33x the throughput of 1K chunks at 128K</summary>

[![Qwen3.8-Flash-Next GSQ-RCO IQ3_S at 128K on Strata 0.1.60. 4K prefill chunks deliver 2.33 times the cold prefill throughput of 1K chunks.](docs/assets/strata-iq3s-128k-prefill.png)](docs/strata-iq3s-128k-prefill-20261117.md)

[Prefill report](docs/strata-iq3s-128k-prefill-20261117.md)

</details>

<details>
<summary><b>Bonsai DSH 256K</b> · coding quality and runtime across seven 18-attempt configurations</summary>

[![Coding quality and runtime across seven 18-attempt configurations](docs/assets/bonsai-coding-comparison.png)](docs/bonsai-dsh-256k-20261117.md)

[Bonsai DSH 256K report](docs/bonsai-dsh-256k-20261117.md) · [All 18 attempt times](docs/assets/bonsai-dsh-attempts.png) · [Bonsai vs. Ornith context plots](docs/assets/bonsai-ornith-context.png) · [Measurements JSON](docs/results/bonsai-dsh-256k-20261117.json)

</details>

<details>
<summary><b>Qwen3.6 hardest task</b> · result at 64K, 92K and 128K capacity</summary>

[![Qwen3.6 hardest-task result at 64K, 92K and 128K capacity](docs/assets/qwen36-hardest-context.png)](docs/qwen36-hardest-context-20261114.md)

[Qwen3.6 hardest-task and context-allocation report](docs/qwen36-hardest-context-20261114.md) · [Measurements JSON](docs/results/qwen36-hardest-context-20261114.json)

</details>

<details>
<summary><b>Dense Qwen3.8-27B IQ3_S, DFlash2 and matched coding</b> · up to 84.8 tok/s at filled 64K · DFlash2 depth 7 above 100 tok/s</summary>

<details>
<summary>Matched coding ranking and supervision pilot</summary>

[![Strata Coder ranks first on four matched live coding attempts; Codex-supervised Qwen is partial](docs/assets/dense-iq3s-supervision.png)](docs/dense-iq3s-supervision-20261112.md)

[Matched coding ranking and supervision pilot](docs/dense-iq3s-supervision-20261112.md) · [Measurements JSON](docs/results/dense-iq3s-supervision-20261112.json)

</details>

<details>
<summary>Live coding and retrieval (Dense DFlash2 vs Strata Coder)</summary>

[![Dense DFlash2 vs Strata Coder live coding, with matched 64K retrieval](docs/assets/dense-iq3s-live.png)](docs/dense-iq3s-live-20261112.md)

[Live coding and retrieval report](docs/dense-iq3s-live-20261112.md) · [Measurements JSON](docs/results/dense-iq3s-live-20261112.json)

</details>

<details>
<summary>DFlash2 depth sweep and OpenCode replay</summary>

[![DFlash2 depth 7 exceeds 100 tok/s on both filled 64K fixtures](docs/assets/dense-iq3s-dflash2-depths.png)](docs/dense-iq3s-dflash2-20261112.md)

[![DFlash2 depth 7 versus MTP5 on reconstructed OpenCode history](docs/assets/dense-iq3s-dflash2-replay.png)](docs/dense-iq3s-dflash2-20261112.md#opencode-history-replay)

[DFlash2 sweep and replay report](docs/dense-iq3s-dflash2-20261112.md) · [Measurements JSON](docs/results/dense-iq3s-dflash2-20261112.json)

</details>

<details>
<summary>Dense IQ3_S speed and context fit</summary>

[![Dense Qwen3.8-27B IQ3_S filled 64K decode, up to 84.8 tokens per second](docs/assets/dense-iq3s-64k.png)](docs/dense-iq3s-20261112.md)

[![Dense Qwen3.8-27B IQ3_S context fit at 64K, 128K and 256K](docs/assets/dense-iq3s-context.png)](docs/dense-iq3s-20261112.md#context-fit-with-mtp-off)

[Dense IQ3_S speed report](docs/dense-iq3s-20261112.md) · [Measurements JSON](docs/results/dense-iq3s-20261112.json)

</details>

</details>

<details>
<summary><b>Strata Coder</b> · guided 18/18 · 128K capacity six strict passes · one-hour 64K stability gate passed</summary>

<details>
<summary>128K capacity: six strict passes, 55,948 maximum actual input</summary>

[![Coder with 128K capacity: six strict passes, 55,948 maximum actual input](docs/assets/coder-agent128.png)](docs/strata-coder-agent128-20261111.md)

[128K-capacity coding report](docs/strata-coder-agent128-20261111.md) · [Measurements JSON](docs/results/strata-coder-agent128-20261111.json)

</details>

<details>
<summary>One-hour 64K stability gate</summary>

[![Coder passed the one-hour 64K stability gate: verified tool use, compaction, restart recall and held-out grading](docs/assets/coder-stability-handoff.png)](docs/strata-coder-stability-handoff-20261111.md)

[Stability report](docs/strata-coder-stability-handoff-20261111.md) · [Measurements JSON](docs/results/strata-coder-stability-handoff-20261111.json)

</details>

<details>
<summary>Guided Coder 18/18, faster matched tasks, restart recall failed</summary>

[![Rank 1 for measured coding: guided Coder 18/18, faster matched tasks, restart recall failed](docs/assets/coder-guided18.png)](docs/strata-coder-guided18-20261111.md)

[Coding report](docs/strata-coder-guided18-20261111.md) · [Stability verdict](docs/strata-coder-workflow-verdict-20261111.md) · [Rankings & measurements](docs/results/strata-coder-workflow-20261111.json)

</details>

<details>
<summary>Filled-64K benchmark</summary>

[![Coder filled-64K benchmark: actual input, cold and cached decode, fresh prefill, memory and GPU energy](docs/assets/coder-filled64k.png)](docs/strata-coder-filled64k-20261110.md)

[Full report](docs/strata-coder-filled64k-20261110.md) · [Measurements JSON](docs/results/strata-coder-filled64k-20261110.json)

</details>

<details>
<summary>Near-filled 128K and 256K</summary>

[![Coder near-filled 128K and 256K: cold prefill, short-output decode, retrieval and memory](docs/assets/coder-long-context.png)](docs/strata-coder-long-context-20261110.md)

[Full report](docs/strata-coder-long-context-20261110.md) · [Measurements JSON](docs/results/strata-coder-long-context-20261110.json)

</details>

<details>
<summary>Coding-agent quality across Hermes and OpenCode</summary>

[![Coding-agent quality: strict and patch passes across the Hermes and OpenCode experiments, with harness limitations](docs/assets/coding-quality.png)](docs/strata-iq3-opencode18-20261110.md)

[OpenCode report](docs/strata-iq3-opencode18-20261110.md) · [Hermes report](docs/strata-iq3-thinking18-20261110.md) · [Measurements JSON](docs/results/strata-iq3-opencode18-20261110.json)

</details>

<details>
<summary>Corrected stability validation and expert-tail skipping</summary>

[![Corrected one-hour stability validation and matched expert-tail skipping comparison](docs/assets/stability-and-experts.png)](docs/strata-iq3-efficiency-20261110.md)

[Full report](docs/strata-iq3-efficiency-20261110.md) · [Measurements JSON](docs/results/strata-iq3-efficiency-20261110.json)

</details>

</details>

<details>
<summary><b>Earlier experiments</b></summary>

[Huihui 64K](docs/huihui-64k.md) · [DSpine 8K](docs/dspine-results.md) · [Thinking-budget pilot](docs/strata-iq3-opencode-hard-20261110.md) · [Model comparison](docs/model-comparison.md)

[Ornith 9B experimental speed tests](docs/ornith-9b-testing-20261113.md) · DFlash, EAGLE-3 and parallel batching

[Edge0-145B Q3_K at 16K](docs/edge0-145b-16k-quality-20261114.md) · fast inference, 0/18 strict coding passes and 2/18 held-out patch passes

</details>

<details>
<summary><b>Setup, methodology and reproduction</b></summary>

[Setup & methodology](docs/benchmark-guide.md) · [Architecture](docs/architecture-audit.md) · [Memory model](docs/memory-model.md)

[Commands & methodology](docs/benchmark-guide.md) · [Evidence manifest](bench/evidence/publication.json) · [Card renderer](bench/render_cards.py)

Model weights, binaries, credentials and raw local traces are excluded from Git. Detailed results and limitations stay linked beside each card.

</details>
