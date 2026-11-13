# Ornith 1.5 9B experimental speed tests

13 November 2026. Local Ornith 1.5 9B Uncensored Q5_K_M on an RTX 5070 Ti 16 GB, with llama.cpp b12110 (`cd959bae9`), CUDA, Flash Attention, Q8_0 key/value caches, batch 2,048 and microbatch 512. The target GGUF publisher is unverified. These are synthetic speed tests. Coding quality and retrieval accuracy remain unmeasured.

## Baseline

Three `llama-bench` repetitions per test. Each decode test generated 128 tokens. Prefill and decode were timed separately.

| Actual input | Prefill (tok/s) | Decode (tok/s) |
|---:|---:|---:|
| Short | 6,780.45 at 512 tokens | 117.66 from an empty context |
| 64,000 | 4,627.12 | 91.99 |

## Speculative decoding at 64K

One run per cell, with 64,000 fresh input tokens and 512 generated tokens on each of two synthetic coding prompts. Rates are in tokens per second. Acceptance is accepted draft tokens divided by proposed draft tokens.

| Method | Code decode | Edit decode | Code prefill | Edit prefill | Code acceptance | Edit acceptance |
|---|---:|---:|---:|---:|---:|---:|
| Drafting off | 89.75 | 90.07 | 4,660.96 | 4,649.88 | — | — |
| DFlash depth 4 | 128.76 | 139.54 | 4,355.60 | 4,368.99 | 333/709 | 346/654 |
| DFlash depth 8 | 127.53 | **145.40** | 4,360.94 | 4,355.32 | 351/1,268 | 370/1,115 |
| DFlash depth 12 | 124.49 | 136.03 | 4,358.22 | 4,363.09 | 352/1,890 | 365/1,722 |
| EAGLE-3 depth 4 | 51.76 | — | 3,593.65 | — | 84/1,698 | — |

Depth 8 averaged 136.46 decode tok/s across the two prompts and peaked at 9.34 GiB sampled device VRAM. The cold edit request finished in 18.24 s with depth 8 versus 19.47 s with drafting off. Greedy DFlash output first diverged from the control at tokens 34 and 23 on the two prompts, respectively. Exact output parity failed on both probes.

EAGLE-3 used a Qwen3.5-9B head with the Ornith target. The completed code run accepted 4.95% of proposals. Its post-request Windows GPU counter showed 532 MiB shared usage, above the 512 MiB test threshold. The edit run was skipped and fully dedicated residency remains unverified. The result applies to this pairing. The verification batch size was held fixed.

[Ornith DFlash Q4_K_M revision](https://huggingface.co/abenzerps/Ornith-1.5-9B-DFlash-GGUF/tree/4ba29c7a17fb17ba07deb79e25ebdeb00fd01e01) · [Qwen3.5-9B EAGLE-3 Q8_0 revision](https://huggingface.co/dobriak/Eagle3-Qwen3.5-9B/tree/ceaaaed6a006ba661a133b05225ab9439009cc8b)

## Parallel requests

`llama-batched-bench`, drafting off, independent prompts and 512 generated tokens per sequence. Rates are aggregate across the batch unless labeled per sequence.

| Sequences | Decode at 512 input (tok/s) | Decode at 64,000 input (tok/s) | 64K decode per sequence (tok/s) | 64K prefill (tok/s) | 64K prefill time (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 117.50 | 91.76 | 91.76 | 4,651.19 | 13.76 |
| 2 | 222.69 | 144.45 | 72.23 | 4,642.24 | 27.57 |
| 4 | 396.78 | 201.89 | 50.47 | 4,624.91 | 55.35 |

```powershell
llama-batched-bench -m <target.gguf> -c 262144 -ngl 999 --fit off -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 -t 8 -npp 512,64000 -ntg 512 -npl 1,2,4
```

In the separate live API test, four cold 64K requests finished in 67.41 s with drafting off. Four-slot DFlash exceeded the shared-memory threshold at startup (558 MiB shared) and was stopped before generation.

| Two-slot live API | Cold batch time (s) | Cached continuation time (s) | Cached aggregate output (tok/s) |
|---|---:|---:|---:|
| Drafting off | 35.51 | 7.40 | 138.41 |
| DFlash depth 8 | 40.38 | 7.94 | 128.91 |

Each cached continuation reused 64,511 tokens and evaluated one fresh token per slot. Aggregate output includes elapsed client time. Each configuration has one sample. The runtime was stopped after testing.
