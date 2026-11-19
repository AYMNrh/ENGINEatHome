# Swift 1.5 Qwen3.8-27B / filled Q4 MTP on RTX 2080 Ti 22 GB

Q4_0 target and draft K/V, full CUDA weights including the token embedding, Flash Attention, batch 512, microbatch 128, draft microbatch 64 and one sequence. Thinking enabled with verified xhigh instructions and unrestricted reasoning. Sampling uses temperature 1, top-p 0.95, top-k 20, min-p 0, seed 42, presence penalty 0 and repeat penalty 1.

Each prompt leaves 2,048 tokens of headroom and allows 1,792 total output tokens for retrieval. Three signed active records at 10%, 50% and 90% depth have revoked distractors. The response must match the predefined three-field JSON object. Input IDs are saved and checked to be identical across depths. Remaining text repeats a synthetic archive.

| Capacity | MTP | Actual input | Cold prefill tok/s | Cold decode tok/s | Sustained decode tok/s | Peak GPU GiB | Min free MiB | Retrieval cold/cached |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 131,072 | 0 | 129,024 | 314.1 | 11.80 | 11.81 | 14.32 | 7,339 | pass/pass |
| 131,072 | 3 | 129,024 | 296.7 | 27.74 | 24.64 | 15.77 | 5,851 | pass/pass |
| 131,072 | 5 | 129,024 | 296.0 | 29.19 | 21.53 | 16.06 | 5,553 | pass/pass |
| 196,608 | 0 | 194,560 | 249.8 | 9.57 | 9.56 | 15.71 | 5,915 | pass/pass |
| 196,608 | 3 | 194,560 | 235.7 | 20.83 | 17.35 | 17.49 | 4,091 | pass/pass |
| 196,608 | 5 | 194,560 | 235.5 | 18.56 | 16.23 | 17.78 | 3,791 | pass/pass |
| 262,144 | 0 | 260,096 | 207.4 | 8.10 | 8.08 | 17.10 | 4,491 | pass/pass |
| 262,144 | 3 | 260,096 | 195.7 | 18.88 | 19.31 | 19.21 | 2,331 | pass/pass |
| 262,144 | 5 | 260,096 | 195.3 | 19.89 | 19.51 | 19.50 | 2,033 | pass/pass |

| Capacity | MTP | Cold TTFT s | Cached TTFT s | Reused / fresh | Cached fresh prefill tok/s | Accepted / proposed drafts | Sustained speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 131,072 | 0 | 410.738 | 0.260 | 129,020 / 4 | 17.0 | â€” / â€” | 1.00x |
| 131,072 | 3 | 434.945 | 0.263 | 129,020 / 4 | 16.8 | 358 / 457 | 2.09x |
| 131,072 | 5 | 435.941 | 0.262 | 129,020 / 4 | 16.8 | 373 / 679 | 1.82x |
| 196,608 | 0 | 779.047 | 0.312 | 194,556 / 4 | 14.5 | â€” / â€” | 1.00x |
| 196,608 | 3 | 825.323 | 0.308 | 194,556 / 4 | 14.7 | 332 / 534 | 1.82x |
| 196,608 | 5 | 826.044 | 0.313 | 194,556 / 4 | 14.4 | 360 / 741 | 1.70x |
| 262,144 | 0 | 1,253.872 | 0.356 | 260,092 / 4 | 13.0 | â€” / â€” | 1.00x |
| 262,144 | 3 | 1,329.068 | 0.351 | 260,092 / 4 | 13.2 | 374 / 408 | 2.39x |
| 262,144 | 5 | 1,331.752 | 0.348 | 260,092 / 4 | 13.3 | 405 / 529 | 2.42x |

One cold retrieval and one exact replay per profile. Accuracy covers three records per prompt. A separate replay forces 512 output tokens for sustained decode. Those outputs are excluded from retrieval scoring. Memory is sampled device-wide. Dedicated process allocation and CUDA layer/cache logs are checked before and after inference. Linux lacks a directly comparable WDDM shared-memory counter. Output lengths and sampled reasoning can differ across depths, so fixed-length decode and natural-response latency are reported separately. This is one sample per profile, without confidence intervals.


## Measured choices

- 131,072 MTP3 led sustained decode at 24.64 tok/s. MTP0 completed cold retrieval fastest at 435.41 s.
- 196,608 MTP3 led sustained decode at 17.35 tok/s. MTP0 completed cold retrieval fastest at 811.05 s.
- 262,144 MTP5 led sustained decode at 19.51 tok/s. MTP0 completed cold retrieval fastest at 1292.80 s.
The measured long-context candidate is 262,144 Q4_0 with MTP5. Coding-agent quality with MTP remains unmeasured.
At 256K, MTP5 measured 1.1% more sustained throughput than MTP3 and used 298 MiB more peak GPU memory. With one sample, that small speed difference does not establish a repeatable advantage. For everyday cached work, use 128K Q4_0 with MTP3. For the longest context, 256K Q4_0 with MTP3 offers nearly the same measured decode speed as MTP5 with additional VRAM headroom. MTP off had the shortest cold retrieval latency at every capacity.

Run state complete.

## Reproduce

Run from the benchmark workspace with the original endpoint idle.

    PYTHONPATH=src python3 bench/swift_homelab.py --replace-pid <8080-pid> --live-audit live-audit --output filled-mtp-fresh --filled-mtp
    python3 bench/swift_filled_mtp_report.py filled-mtp-fresh

Runtime version: 1.0.0-dev (build 12519, commit 98d67c8d5)
built with GNU 15.2.0 for Linux x86_64

Model SHA256 9aecf1cd41b2cb2f32a74e0d889e33855ebef43b26f43b43feb5720239e677e5. Launch commands and estimates are in provenance.json and each result.json. The original command is restored and recorded in restoration.json.
