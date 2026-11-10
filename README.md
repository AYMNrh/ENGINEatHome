# QwenAtHome Engine

New specialized runtime, starting with independent GGUF inspection, memory planning and reproducible reference-backed execution. llama.cpp is an external reference and initial execution component. QAH does not yet contain an independent transformer graph or CUDA kernels. Existing ~84K context is the baseline to reproduce; 128K and 256K are investigation targets, 512K+ an experimental extension.

Requires Python 3.9+ (standard library only), a compatible llama-server binary, the GGUF, and NVIDIA driver tools for GPU telemetry. On this machine use `py -3.9`; the Python Store 3.10 launcher is inaccessible.

```powershell
py -3.9 -m unittest discover -s tests -v
py -3.9 bench/audit.py --model 'F:\llama.cpp\models\Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S-mtp.gguf' --server 'F:\llama.cpp\b12110-cuda13\llama-server.exe' --checksum --output bench/results/audit-iq3.json
py -3.9 bench/run.py --model 'F:\llama.cpp\models\Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S-mtp.gguf' --server 'F:\llama.cpp\b12110-cuda13\llama-server.exe' --contexts 8192 --trials 3 --tasks retrieval code decode --depths 0.1 0.5 0.9 --output bench/results/reference-f16.jsonl
py -3.9 bench/summary.py bench/results/reference-f16.jsonl --output bench/results/reference-f16-summary.json
```

For a controlled long-context probe, explicitly select lossy KV and CPU weight placement:

```powershell
py -3.9 bench/run.py --model 'F:\llama.cpp\models\Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S-mtp.gguf' --server 'F:\llama.cpp\b12110-cuda13\llama-server.exe' --contexts 8192 32768 86016 131072 --trials 1 --generate 128 --key-type q4_0 --value-type q4_0 --gpu-layers 64 --tasks retrieval code decode --depths 0.5 --no-reuse --output bench/results/context-probes.jsonl
```

This one-trial command is a probe, not a statistically repeatable result. Use 3+ trials to estimate medians and nearest-rank p95 (small-sample p95 is only the maximum). Default runner matrix covers 8K, 32K, 84 Ki tokens (86,016), 128K, 256K and 512K; unsupported positional lengths and known all-GPU budget violations are recorded without allocating. These are total capacity sizes. Actual prompt length reserves generation plus one backend token; exact token-array length and SHA-256 are recorded.

Cold requests disable prompt reuse. Warm-prefix requests repeat the exact token array and compare deterministic outputs to cold; inspect `timings.cache_n` for actual reuse, not `tokens_cached` (which is state length after the request). Recurrent backends may fully reprocess even identical prompts. The tool-turn fixture is a controlled message-suffix/cache locality probe; it is not full OpenAI tool-call conformance or an actual coding-agent task.

`decode` forces the requested output length by explicitly ignoring EOS and measures sustained decode near filled context. That choice is logged and **only applies to the throughput probe**. Quality fixtures honor EOS and use strict exact-match answers. Repeated, varied filler is disclosed in results; retrieval, a small three-file dependency tracing task and forced decoding do not establish real coding-agent tasks/minute.

Each server is isolated on loopback, uses one slot, disables automatic fit, context shifting, speculation and host prompt-cache/checkpoints, and is terminated after its trial. Timeouts/failures are outcomes. `--allocation-only` loads and warms up the model, records allocation details and makes no prefill/decode/quality claim. Check `reference_argv`, actual generation settings and verbosity-4 allocation logs for precise residency and behavior. No existing server is modified. Results are local ignored artifacts under `bench/results/`; summary never turns failed/unknown quality into success.

Completed-request rows include model and binary checksums, reference build/commit, source hash, repository commit (null while unborn), dirty status, exact prompt size/hash, generation, seed/sampling, KV types, placement flags, phase timings, sampled memory peaks and raw-log path. Allocation/gate rows explicitly omit unmeasured request phases. See `bench/schema.json`, `docs/architecture-audit.md`, `docs/memory-model.md`, `docs/benchmark-results.md`, and `docs/progress.md`.

## Preferred-model comparison

Huihui abliterated is the user-preferred candidate. `bench/compare.py` checks both model checksums against the retained QwenAtHome baseline and compares matched settings. It reuses that repository's three executable coding checks with the native chat template. Generated Python is AST-screened and run in separate `python -I` processes with an eight-second timeout; the filter is not a security sandbox.

```powershell
py -3.9 bench/compare.py --contexts 32768 --depths 0 3 --trials 3 --tasks coding --output bench/results/compare-coding-32k.jsonl
py -3.9 bench/compare.py --contexts 32768 84992 --depths 3 --trials 1 --tasks retrieval decode --output bench/results/compare-context.jsonl
py -3.9 bench/summary.py bench/results/compare-coding-32k.jsonl --output bench/results/compare-coding-32k-summary.json
```

These comparison runs enable fit on/512 MiB target, GPU sampling, Jinja, prefix checkpoints and MTP as selected. Target batch/microbatch is 512/128; main KV is q4_0 and draft KV q8_0. `--draft-ubatch` controls only the draft's physical microbatch. Cold coding explicitly disables prefix reuse. Warm retrieval reports actual cached input tokens. Non-streaming coding TTFT is null; HTTP elapsed time is measured. The summary separates MTP depth and speculation settings.

A failed reference fit-margin check is refused before prefill. `--allow-memory-pressure` explicitly overrides that safeguard for a deliberate pressure experiment; it does not change attention or positional semantics. The original `bench/run.py` remains the MTP-off baseline runner. Results and limits are in `docs/model-comparison.md`.

Expanded Huihui 64K run (six coding tasks twice, six retrieval cases plus cached repeats, two decode probes):

```powershell
py -3.9 bench/compare.py --models huihui --contexts 65536 --depths 3 --trials 2 --extended-coding --needle-depths 0.1 0.5 0.9 --tasks coding retrieval decode --output bench/results/huihui-64k-expanded.jsonl
```

`--extended-coding` adds interval merging, dependency ordering and JSON merge patch checks. `--needle-depths` varies placement within repeated filler; summaries separate these depths. Coding uses short prompts at the configured capacity; retrieval/decode use capacity minus 257 prompt tokens. Two repetitions do not provide reliable tail-latency estimates.

## DSpine experiment

`--dspine-draft PATH` enables the pinned community DSpine port through its `draft-dflash` mode (DSpine tensors are auto-detected). It requires the Q4_K_M draft SHA-256 recorded by the harness and a DSpine-capable binary; the original Windows b12110 server does not support it. Without this flag, existing MTP runs are unchanged. Draft depths up to 15 are accepted for DSpine; `mtp_depth` is null for active DSpine rows and `draft_tokens` records the selected block length.

For Linux/WSL, run the harness inside Ubuntu with Linux paths, including `/mnt/f/llama.cpp/models` for the existing weights. Example from the workspace root after building the port:

```bash
python3 bench/compare.py --models huihui --model-dir /mnt/f/llama.cpp/models \
  --server build/dspine/cuda-build/bin/llama-server \
  --dspine-draft build/dspine/Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf \
  --contexts 8192 --depths 3 --trials 1 --extended-coding \
  --needle-depths 0.1 0.5 0.9 --tasks coding retrieval decode \
  --output bench/results/huihui-dspine-8k-d3-g99.jsonl
```

Use the same fork, context, KV and placement for matched MTP/no-speculation controls. Draft weights and extra recurrent verification states cost memory; a fit refusal is an outcome, not a throughput result. DSpine's own sliding window does not shorten target-model attention. Linux process-RAM telemetry is currently unavailable in the shared monitor and remains null.

## Git repository

This private repository contains the ENGINEatHome source, tests, experiment configuration and documentation. Selected sanitized synthetic benchmark evidence is published under `bench/evidence/`. Model weights, llama.cpp binaries, raw local benchmark results and the nested `.publication/` checkout are excluded. The `.publication/` checkout belongs to a separate QwenAtHome repository and is not part of this source tree.

## Latest measurements

**Current provisional preferred configuration: OpenCode + Strata IQ3_S, 1K thinking, 64K capacity, approximate expert-tail skipping 7.** The [full six-issue, three-trial matrix](docs/strata-iq3-opencode18-20261110.md) scored **16/18 strict passes and 18/18 passing patches**, at 97.6 accepted output tok/s including reasoning. Both strict failures reached 100 steps with passing patches; subsequent inspection found premature shell timeouts caused by an ambiguous timeout-unit contract. These historical completion scores are harness-affected and remain unchanged. Actual prompts reached 57,294 tokens. [Saved settings and measurements](docs/results/strata-iq3-opencode18-20261110.json).

The [corrected stability and expert-tail comparison](docs/strata-iq3-efficiency-20261110.md) passed a 62-minute review session with 557 completed tool calls, 11 recorded compactions, successful recall/process restart, preserved patch and 21/21 grading tests. With explicit-seconds shell timeouts, four coding attempts per profile scored **2/4 strict and patch passes with skipping ON**, versus **1/4 strict and 2/4 patch passes with skipping OFF**. Coding decode was 103.3 versus 81.9 tok/s; coding-window GPU energy 44.6 versus 50.3 Wh. Long baseline test runs and unrestored fixes dominated failures. This small, sequential sample establishes neither general quality equivalence nor filled-64K retrieval accuracy. The original RAM reserve still fails; no 128K promotion. [Comparison measurements](docs/results/strata-iq3-efficiency-20261110.json).

The preliminary [thinking-budget comparison](docs/strata-iq3-opencode-hard-20261110.md) ranked 1K first (2/2, fastest), 4K second (2/2, slower) and no explicit cap third (1/2), on one issue repeated twice per setup. Those small-sample rankings are not a broad quality comparison. [Pilot benchmark card](docs/assets/strata-opencode-64k-card.png).

The broader [Hermes thinking-on validation](docs/strata-iq3-thinking18-20261110.md) scored 13/18 strict passes and 16/18 correct patches, versus 7/18 and 10/18 with thinking off. OpenCode also used larger step/time/output limits, so this does not isolate the harness effect. The physical-RAM reserve gate still fails, and expert-tail skipping 7 remains approximate. Neither result establishes 128K promotion.

| Experiment | Near-filled-context decoding | Scope |
| --- | --- | --- |
| [Huihui MTP3 at 64K](docs/huihui-64k.md) | 51.08 tok/s median, two runs | 26 requests; coding 10/12, retrieval 12/12 |
| [Huihui DSpine3 versus MTP3 at 8K](docs/dspine-results.md) | 95.76 versus 79.80 tok/s | Single decode probe each; five profiles, 65 requests total |

DSpine3 used all trunk layers on GPU and about 1 GiB more sampled peak VRAM than MTP3. Its draft limit was reduced to three tokens. These DSpine numbers are for 8K only; no DSpine result exists at 64K or longer. The reports retain failures, memory limits and exact settings. [Sanitized evidence manifest](bench/evidence/publication.json) records original and published artifact hashes; original logs and source snapshots remain local.
