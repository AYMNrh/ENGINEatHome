# Swift 1.5 Qwen3.8-27B / RTX 2080 Ti 22 GB

| Profile | Actual input | Cold prefill tok/s | Cold decode tok/s | Cached TTFT / reused tokens | Sustained decode tok/s | Cold peak VRAM GiB | Retrieval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Live 128K Q8 | 120,000 | 345.6 | 11.39 | 0.249 s / 119,996 | â€” | 16.12 | pass |
| 131,072 q8_0 MTP0 | 120,000 | 322.7 | 11.38 | 0.256 s / 119,996 | 11.38 | 16.32 | pass |
| 131,072 q4_0 MTP0 | 120,000 | 323.9 | 12.14 | 0.259 s / 119,996 | 12.14 | 14.32 | pass |
| 196,608 q8_0 MTP0 | 195,584 | 247.6 | 8.46 | 0.308 s / 195,580 | 8.46 | 18.71 | failed or incomplete |
| 196,608 q4_0 MTP0 | 195,584 | 248.7 | 9.52 | 0.301 s / 195,580 | 9.53 | 15.71 | pass |
| 262,144 q8_0 MTP0 | 261,120 | 206.0 | 6.87 | 0.355 s / 261,116 | 6.86 | 21.10 | pass |
| 262,144 q4_0 MTP0 | 261,120 | 206.7 | 8.04 | 0.355 s / 261,116 | 8.06 | 17.10 | pass |
| 131,072 q4_0 MTP0 | 64,000 | 416.9 | 15.16 | 0.221 s / 63,996 | 15.14 | 14.32 | pass |
| 131,072 q4_0 MTP3 | 64,000 | 396.3 | 31.33 | 0.219 s / 63,996 | 32.23 | 15.89 | pass |
| 131,072 q4_0 MTP4 | 64,000 | 396.3 | 28.32 | 0.223 s / 63,996 | 31.82 | 16.04 | pass |
| 131,072 q4_0 MTP5 | 64,000 | 395.9 | 29.51 | 0.223 s / 63,996 | 34.38 | 16.19 | pass |
| 5070 Ti 65,536 Q4 MTP0 | 64,512 | 1086.1 | 37.61 | 0.095 s / 64,508 | â€” | 14.44 | pass |
| 5070 Ti 92,160 Q4 MTP0 | 91,136 | 893.5 | 32.55 | 0.130 s / 91,132 | â€” | 14.85 | pass |
| 5070 Ti 65,536 Q4 MTP3 | 64,512 | 581.3 | 69.52 | 0.099 s / 64,508 | â€” | 15.32 | pass |

Cold requests require zero reused tokens. Sustained decode forces 512 output tokens. VRAM is sampled device-wide. The 5070 Ti used a different llama.cpp build. Its MTP3 run used Q4 draft KV and 128/32 batch; Homelab MTP uses Q8 draft KV and 512/128 batch.

- 262,144 q8_0 MTP0 â€” dedicated-memory guard: PID 42991, used bytes [22649241600], required 21044249231, GPU {'used': 22653435904, 'free': 414187520, 'utilization_percent': 99.0, 'temperature_c': 47.0, 'power_watts': 200.79, 'power_limit_watts': 200.0}
- 196,608 q8_0 reached its 512-token total output allowance during thinking. Retrieval remains unverified.

Largest profile that passed retrieval and the residency guard: 262,144 capacity with q4_0 and 261,120 actual input tokens.


| MTP depth | Sustained accepted / proposed drafts | Cold accepted / proposed drafts |
| ---: | ---: | ---: |
| 3 | 364/438 | 135/168 |
| 4 | 382/513 | 94/148 |
| 5 | 403/535 | 99/160 |

## Coding

- RTX 5070 Ti 4/4 strict and 4/4 patches in 34.80 minutes.
- RTX 2080 Ti 3/4 strict and 3/4 patches in 57.83 minutes.

| Issue | Trial | Strict pass | Patch pass | Time min | Tools | Timed out |
| --- | ---: | --- | --- | ---: | ---: | --- |
| pytest-dev__pytest-6202 | 0 | True | True | 7.87 | 30 | False |
| sphinx-doc__sphinx-10323 | 0 | False | False | 20.00 | 35 | True |
| pytest-dev__pytest-6202 | 1 | True | True | 15.20 | 57 | False |
| pytest-dev__pytest-6202 | 2 | True | True | 14.76 | 44 | False |

Sphinx reached the 20-minute limit. Its retained patch was graded offline after a Windows-only cleanup command failed on Linux. The two remaining Pytest runs used the corrected process-group cleanup.
Retokenizing captured traces from 2 attempts estimates 16,673 reasoning tokens.

Run state failed. Original endpoint restoration is recorded in restoration.json.
Coding continuation state complete.

## Reproduction

Use bench/swift_homelab.py with the verified checkpoint and exact llama-server build in provenance.json. Each profile directory records the launch arguments, prompt hash, rendered effort controls, output IDs, timings, and sampled telemetry. The live audit records the original 120K cold and cached requests. The coding files retain the frozen task manifest hash, OpenCode traces, patches, and individual grader outcomes.

From the Homelab benchmark workspace, regenerate this report with python3 bench/swift_homelab_report.py results-20261118 --coding-resume coding-resume-20261118.
