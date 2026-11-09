# Progress

## 2026-11-08 — Initial audit and implementation

- Repository was empty: unborn `master`, no tracked files, submodules, configs or baseline implementation. No user files overwritten; no commits or pushes authorized by the direct request.
- Read the execution plan in full. Implementing Phase 0 and Phase 1 first; later phases remain gated by measured evidence.
- GPU accessible: RTX 5070 Ti, 16,303 MiB reported by nvidia-smi, driver 621.14. Toolkit 12.6.85. Existing reference executable: F:/llama.cpp/b12110-cuda13/llama-server.exe, build 12110, commit cd959bae9, CUDA sm_120 present in historical startup log.
- Located three Qwen3.8-27B GGUF weight variants. Existing reference log indicates qwen35 hybrid architecture, 64 trunk + 1 MTP, 16 full-attention and 48 recurrent layers. Independent GGUF parser and memory estimator being validated against the actual file.
- Python 3.10 Store launcher is inaccessible; Python 3.9.5 works. Implementation uses standard library only.
- RAM/CPU CIM inspection denied by host permissions; historical reference log reports 7800X3D and 64,548 MiB RAM. Current physical RAM and PCIe bandwidth must be measured separately.
- Existing ~84K / ~70 tok/s baseline is user-reported, not reproduced. 128K/256K/512K quality and speed are unmeasured.

## Milestone 1 — Executable inventory and model-specific planner

- Implemented `src/qah/model.py`: bounded little-endian GGUF v2/v3 reader, metadata and tensor directory validation, supported quantization block sizes, alignment/overlap/file-bound checks, per-layer encoded sizes and streaming SHA-256. Rejects unsupported architectures/shards/encodings instead of guessing.
- Actual IQ3_S model: 866 tensors, SHA-256 `eea0638e283433e27b0edbd409b552be467a75ecc777d91c51db554c0a644c19`. Metadata/tensor families agree on 64 trunk layers (16 dense/48 recurrent) plus one MTP block; metadata context limit 262,144.
- Implemented `src/qah/memory.py`, `bench/audit.py`, architecture audit, memory formulae and an ADR. Phase 0 gate passed with explicitly documented workspace/graph/placement unknowns.
- Windows `GlobalMemoryStatusEx` bypasses the unavailable CIM query without escalation: live physical RAM 67,684,352,000 bytes (64,548.8 MiB). PCIe under inference reports Gen5 x16, versus idle Gen1 x16; transfer bandwidth remains unmeasured.
- `py -3.9 -m unittest discover -s tests -v`: 3 tests pass (GGUF bounds, hybrid memory/gates, truthful summary scoring). No added dependencies.

## Milestone 2 — Reference-backed execution and benchmark harness

- Implemented isolated reference lifecycle/streaming, memory sampling, exact token-count fixtures, reproducible seeds, cold/warm/tool-prefix phases, forced-length decode probe, dependency-tracing task, JSONL schema and median/p95 summary.
- Sampling, EOS override for throughput only, lossy KV, positional limits, fit behavior, requested placement and actual backend terminal settings are recorded. New runs snapshot source/fixtures/configs because repository HEAD is still unborn.
- Initial 8K f16 smoke: needle exact-match passed, cold/warm deterministic outputs matched, but warm input-cache count was zero. Tool-prefix prototype failed strict quality; do not count it as working tool support.
- Initial throughput smoke exposed a capacity-boundary issue in the harness: reserve an additional token and reject truncated throughput results. Fixed before context probes; the original smoke's truncated decode samples are diagnostic only.
- Controlled q4_0 probes use 64 GPU layers, one trial, midpoint needle and 128 generated throughput tokens. At 8K/32K retrieval passed; code-task strict answer scoring failed (returned Python code rather than integer). Preserve failures; do not loosen scoring.
- Reference default verbosity omits allocation details. Subsequent runs explicitly request verbosity 4; earlier logs cannot establish precise tensor residency.
- A broad matrix was interrupted after one cold sample to prioritize bounded context probes. It is incomplete, not a completed benchmark.

## Long-context probe update

- Completed 84 Ki-token capacity probes: exact midpoint retrieval passed; dependency tracing failed strict scoring; forced 128-token decode completed without truncation at about 26.83 backend decode tok/s. Cold TTFT was about 81 seconds. These are single-trial, lossy-KV, partly CPU-placed reference measurements, not engine optimization claims.
- At 131,072 capacity, 130,943 prompt tokens successfully prefilled and midpoint retrieval passed exact match. TTFT about 164.97 seconds, 15 predicted tokens. Other 128K checks still running at this entry.
- Added allocation-only runs to distinguish load/warmup from prefill, sustained decode and quality. GPU gates exclude uncertain shared tensor placement, and retain the 1 GiB reserve.
- Verbose allocation evidence showed an important backend placement detail: `--n-gpu-layers 64` offloads 64/66 placement slots, with 6.23 MiB recurrent state on CPU (two recurrent layers). Placement includes the stored MTP block and output slot, even with speculation disabled. Corrected the all-trunk GPU gate to require 66 slots; earlier q4 allocation gate rows using 64 are diagnostic and superseded by final all-GPU (`99`) checks. Never infer actual residency from the requested layer number alone.

## Milestone 3 — Completed initial measurements and truth table

Done:

- 12 full context probes completed: retrieval, dependency tracing and forced 128-token decode at 8K/32K/84 Ki/128K. Needle passed at each midpoint; code task failed at every length. All revised forced decodes completed without truncation.
- 128K: exact prompt 130,943 tokens; decode TTFT 164.922 s; backend decode 22.679 tok/s for 128 forced tokens, one trial, q4_0 K/V, 64 requested placement slots (two recurrent trunk layers CPU placed). Sampled device-wide peak 15,792 MiB includes other apps. This is a measured reference probe, not validated useful 128K or an optimization.
- Three 8K f16 reference trials per task: retrieval 3/3, code 0/3, forced decode 3/3. Median forced-decode rate 52.348 tok/s; nearest-rank p95 52.665. Median TTFT 4.040 s. No comparison to the user-reported ~70 tok/s baseline is established because model/config/placement differ.
- Final frozen-source allocation run (`allocation-verified-q4.jsonl`): 128K q4 all-trunk GPU load/warmup succeeded, 256K all-trunk GPU budget refused, 512K positional limit refused. Earlier all-GPU allocation run has the same outcomes; verified run captures the final gate implementation.
- 256K q4 with 48 requested GPU slots loaded and warmed up successfully: 1,152 MiB KV CPU + 3,456 MiB KV GPU, matching 4,608 MiB predicted. Full 256K prefill/decode/quality remain untested. CPU placement includes four dense layers and fourteen recurrent layers.
- Captured detailed backend allocation logs, process/device memory, source snapshots, model/binary checksums, seed/sampling settings, JSONL and median/p95 summaries. Created `docs/benchmark-results.md` with evidence links and limitations.
- Final validation: `py -3.9 -m unittest discover -s tests -v` (3 pass); `py -3.9 -m compileall -q src bench tests` (pass); JSONL required fields/phase/status checks and supported source-snapshot hashes (pass). No dependencies, commits, pushes, custom kernels, approximate attention or positional overrides added. All owned reference processes stopped.

Blocked/unmeasured gates:

- Phase 0 passed. Phase 1 foundation is executable, but useful-context quality, long-context repeatability and real tool/agent performance gates have **not** passed. Strict code fixture failure also occurs at 8K f16; investigate task/template/reasoning configuration before attributing it to context or KV quantization.
- Full 256K tests need an explicit CPU-placement experiment and runtime budget; 512K needs an independently evaluated positional-extension strategy. Allocation alone is not support.
- Independent graph/logit parity, MTP, rollback, prefix DAG, exact host KV tiering, API/tool conformance, PCIe transfer measurement and optimizations remain unimplemented/unmeasured. No validated production hardware profiles exist.

Next:

1. Validate representative coding-agent fixtures and model template/reasoning settings, retaining strict scoring and the current failure evidence.
2. Repeat 128K at multiple needle depths/seeds with safe headroom, including feasible f16 comparisons and measured q8/q4 quality deltas.
3. Run full 256K prefill, sustained decode and quality using the measured partial CPU placement; measure host compute/transfer costs.
4. Only after baseline gates pass, implement independent forward/logit checks and recurrent snapshot/rollback semantics before replacing execution components.

## Baseline identified from QwenAtHome repository — 2026-11-08

Inspected https://github.com/AYMNrh/QwenAtHome (README, launch profiles, methodology, retained coding harness, Q3/Huihui reports and raw Q3 results). The local ENGINEatHome repository has no remote and is a separate workspace; nothing was cloned, merged, committed or pushed.

- The historical ~84K profile is **Qwen3.8-27B-UD-Q3_K_XL.gguf**, SHA-256 `8c2a45ff85e7674ca185ec8eb6cdeab0e617ed9d8018caed0b64380eb2a67a5e`, allocation **84,992** (not the 86,016 probe capacity used here).
- Published launcher: all GPU placement, q4_0 main K/V, draft-mtp up to 3 tokens, q8_0 draft K/V, batch 512/microbatch 128, backend sampling, Jinja chat template, fit on/512 MiB target, one slot. Our initial probes used a different model, disabled MTP, microbatch 512 and partial CPU placement; they do not reproduce this baseline.
- Retained raw Q3 results: short chat 67.833 tok/s with a 34-token prompt, 413 generated tokens; thinking chat 65.122 tok/s with a 74-token prompt, 512 generated tokens (hit output cap). Near-filled cold prompt 83,916 tokens: 236.022 s prefill, 39.025 decode tok/s, 237.381 s wall; resumed 84,000 tokens: 83,965 cached, 35 newly evaluated, 46.867 decode tok/s, 1.439 s wall. Both retrieval checks passed. These are historical measurements, not newly reproduced results or sustained 128-token filled-context benchmarks.
- The Huihui file we inspected matches the separate published Huihui SHA-256 exactly, but that historical profile was 32K, q8 KV and MTP. The repo records two successful edit tasks and a failing fresh-code task; its executable coding checks are more representative than our integer-only dependency fixture. They use chat completions with enable_thinking=false and up to 1,600 generated tokens.
- Next baseline work must checksum/audit Q3_K_XL and reproduce its exact 84,992-token profile before comparing a new runtime or MTP/context changes. Preserve all earlier probe results as measurements of their actual configurations. Do not interpret the ~70 tok/s short-chat figure as filled-84K speed.

Sources: [Q3 launcher](https://github.com/AYMNrh/QwenAtHome/blob/main/profiles/Start-Qwen38-Q3-85K.cmd), [Q3 report](https://github.com/AYMNrh/QwenAtHome/blob/main/docs/reports/Qwen38-Q3-85K-report.md), [raw Q3 results](https://github.com/AYMNrh/QwenAtHome/blob/main/data/Qwen38-Q3-85K-results.json), [methodology](https://github.com/AYMNrh/QwenAtHome/blob/main/docs/methodology.md).

## Matched model comparison started — user prefers abliterated Huihui

- User requested testing both Q3_K_XL and Huihui, with Huihui as the preferred model. Preference recorded; no model replacement or deployment performed.
- Added `bench/compare.py` and retained the user's published three executable coding fixtures (`bench/fixtures/upstream_focused_checks.py`, upstream blob 4860d114b71f43954b077476bab853535ac3428f). These are existing checks, not an expanded generated test suite. Generated modules are AST-screened and executed in isolated Python subprocesses with the original eight-second timeout; that screening is not a security sandbox.
- Reference adapter now supports explicit MTP depth, microbatch, GPU sampling and prefix/checkpoint settings while retaining previous defaults (MTP off, cache off, fit off). Comparison uses native Jinja chat formatting with thinking disabled, seed 42, up to 1,600 output tokens, q4 main K/V, q8 draft K/V, all GPU placement, batch 512/microbatch 128, GPU sampling, fit on/512 MiB target and original cache settings.
- Running three coding trials per model with MTP 0 and 3 at 32K capacity. Identical fixed seed is intentional for deterministic repeatability and MTP output comparisons, not a diverse task accuracy estimate. Full-prefix retrieval/decode probes are separate from short coding requests.
- Summary grouping now separates MTP depth/speculation; regression added to the existing summary test, with no extra test functions. No silent mixing of MTP-off and MTP-on speeds.

### Matched coding milestone completed

- 36 requests completed: two models, MTP depths 0/3, three existing coding checks, three repeats. Both model checksums match the published baseline. No coding prompt cache reuse (cache_n=0). All repeated outputs within each profile/task were identical, so repeats measure timing variability rather than task diversity.
- Huihui MTP3: bugfix 3/3 at median 125.94 tok/s; CSV edit 3/3 at 115.26; fresh code 3/3 at 96.92. Median HTTP task times 2.08 / 1.75 / 6.77 s respectively.
- Q3_K_XL MTP3: bugfix 3/3 at 123.86 tok/s; CSV edit 3/3 at 116.66; fresh code 0/3 (TypeError: list is not an iterator). Failed fresh outputs still report speed but are never counted as completed coding tasks.
- Without MTP, both pass edits and fail fresh code. Huihui bugfix/CSV medians 50.72/50.49 tok/s; Q3 47.35/47.97. For Huihui, the edit outputs are identical with MTP on/off, while fresh-code output differs and passes only with MTP3. Do not claim general deterministic/logit equivalence: per-layer/logit comparisons and near-tie behavior are unresolved.
- Raw results: `bench/results/compare-coding-32k.jsonl`; MTP-separated summary: `bench/results/compare-coding-32k-summary.json`. These small synthetic tasks support the user's Huihui preference, not a broad model-quality ranking.
- Long-context comparison is underway at 32,768 and 84,992 capacity, MTP3, identical q4 main/q8 draft cache configuration. Native-template midpoint retrieval and a 128-token parser-generation throughput probe are distinct tasks. An interrupted pilot accidentally used a short-answer retrieval prompt for forced decoding; excluded it and restarted with the long-code prompt (`compare-context-mtp3-v2.jsonl`).

### Comparison complete — preference retained

- Huihui MTP3 completed 32K and 84,992-capacity cold/warm retrieval plus forced 128-token parser generation. At 32K: 73.59 decode tok/s; at 84,992: 56.21 tok/s. Near-85K exact prompt was 84,735 tokens; cold retrieval took 106.67 s, warm 0.44 s. All four needle checks passed. These are single probes, not long-context statistical validation or full code-quality checks.
- Q3 MTP3 completed the same 32K checks, with 75.39 parser decode tok/s. Its near-85K prefill slowed severely; last progress was 47,616 tokens at 300.36 server-reported prefill seconds, including roughly 57 s for the last 512-token segment. Stopped the memory-pressure attempt and retained a cancelled outcome, explicitly not an OOM or completed benchmark. No full near-85K Q3 decode rate is claimed.
- Tested one targeted workspace change: draft microbatch 4 instead of 128. Actual draft reservation only fell 372.75 -> 333.27 MiB (39.48 MiB), insufficient to resolve the margin problem. Huihui cold/warm retrieval repeated successfully; remaining throughput experiment was interrupted without adopting or claiming a speed optimization.
- Added a comparison startup gate: if the reference fitter reports failure to meet its requested 512 MiB reserve, refuse before prefill unless `--allow-memory-pressure` is explicit. A fresh Q3/84,992/draft4 run produced `refused_budget`, confirming the gate. This says the profile cannot meet its reserve in this run, not that historical 84K Q3 results or other placements are impossible.
- Fixed MTP allocation parsing: main and draft KV entries previously overwrote each other. New parser preserves allocation events and sums coexisting model/KV/state buffers, while treating compute reservations as repeated/last values. One regression test added (four total test functions). Earlier raw timings/VRAM samples/logs remain intact; `bench/results/compare-buffer-allocations.json` reparses complete logs for corrected buffer accounting.
- Outcome: retain **Huihui abliterated** as preferred. MTP3 at 32K is the fastest tested candidate on these fixtures; 84,992 remains experimental with a few hundred MiB VRAM headroom. Existing 128K MTP-off/partial-CPU results remain separate. No persistent server, weights, sampling algorithm, attention mask or deployment was changed.
- Report: `docs/model-comparison.md`; exact commands in README. Model preference persisted in `configs/experiments.json`. Full 128K/256K agent quality, safe headroom, independent logit parity and general MTP output equivalence remain open.
- Final validation: `py -3.9 -m unittest discover -s tests -v` (4 pass), `py -3.9 -m compileall -q src bench tests` (pass), required JSONL fields/counts and archived-source hash checks (49 rows pass). All owned llama-server processes stopped. No commits or pushes.

## Git publication completed — 2026-11-08

- User authorized commit/push. Published to `AYMNrh/QwenAtHome` main, verified remote HEAD `0dcdf864b5481c1251cfc97dd037ca5a73ac0ba4` (engine addition `9fe7589`, LF/hash correction `0dcdf86`). Existing reports/profiles/history preserved; new work is under `engine/`, with root README/privacy additions.
- Published source, tests, reports and an explicit allowlist of 14 sanitized synthetic evidence artifacts plus their publication manifest. Private paths, full logs, source-snapshot working files, generated-module execution directories, weights and binaries remain local. Generic documentation commit identity matches repository policy.
- Publication copy uses portable MODEL_DIR/LLAMA_SERVER environment defaults. Four tests, compilation, relative report links, privacy scans, and all 14 committed evidence-blob hashes passed. A Windows newline hash mismatch was corrected in a follow-up commit before pushing; no history rewrite or force push.
- Clean publication checkout retained at `.publication/repo` for future publication, ignored by this original prototype workspace. Original local benchmark files/source layout were preserved; the original workspace repository remains separate from the published checkout.

## Huihui 64K expanded benchmark — 2026-11-08

- User requested 64K and more than three tests. Running six distinct functional coding tasks twice (three retained checks plus interval merging, dependency ordering and JSON merge patch), six near-full-context needle cases (two seeds at 10%, 50%, 90%) with six cached repeats, and two forced 128-token decode probes: 26 requests total.
- Capacity 65,536, long prompts 65,279 tokens, MTP3, all trunk layers on GPU, q4 main/q8 draft KV, batch/microbatch 512/128, native template, thinking disabled. Existing fit-margin gate retained. Coding prompts are short; no claim of filled-context coding correctness.
- Added optional extended fixture/depth selection; original upstream fixture unchanged. Extra fixture hash and retrieval depth are recorded, source archived before execution. Four existing harness tests passed. Evidence: `bench/results/huihui-64k-expanded.jsonl`.
- Completed all 26 requests: coding 10/12 executions (5/6 distinct tasks), cold retrieval 6/6, cached retrieval 6/6, forced 128-token decode 2/2 complete. Dependency ordering failed the one-shot iterator check in both identical outputs. Decode 49.41 / 52.75 tok/s, median 51.08; cold/cached retrieval median 72.10 / 0.381 s. Sampled VRAM peak 15,865 MiB and server peak resident memory 20.20 GiB; memory margin is tight. Report: `docs/huihui-64k.md`. No independent-runtime or filled-context coding correctness claim.
- Final validation: compilation passed; all 26 evidence rows, outcome counts, exact long-prompt lengths, archived source contents/current source hash and extra fixture checksum verified. Owned reference server exited; no llama-server processes remain. No commit or push for this run.

## Tencent DSpine evaluation — 2026-11-09

- User requested Tencent DSpine. Official Qwen3.8-27B draft weights now exist (five layers, block 16, draft sliding window 8192); official runtime is BF16 SGLang. Existing Windows b12110 server lacks DSpine support. Found the community GGUF port `Ajay9o9/llama.cpp` branch `dspine`, pinned commit `d493f841b20241606c47a4e0537d3c786b5cf50b`, isolated under ignored `build/dspine`.
- Existing WSL Ubuntu has GCC 13.3 and CUDA 13.3; building for sm120. Initial CMake detected the Windows CUDA environment; corrected toolkit root explicitly. No existing binary or model replaced. Q4_K_M draft download pinned HF revision `62e92414e114702e792ea7a4c37eba57b25bac8b`, expected SHA256 `0c6c984ac18611288d7b8f497bc347c4921e8231b1b6abf28650481eaf35ff55`, 1,257,892,896 bytes.
- Plan: start with 8K correctness/throughput against no-speculation and MTP on the same fork/runtime, using preferred Huihui target. Draft was trained against stock target, so abliterated/quantized target acceptance and correctness must be measured. Full-block verification needs 16 recurrent rows (~2.34 GiB), versus four for MTP3; do not assume old 64K fit.
- Built the sm120 CUDA server successfully with GCC 13.3/CUDA 13.3, CMake toolkit root explicit. Fork untouched; WSL initially reports CRLF checkout files as dirty while Windows Git reports no source diff. Build/version and configuration logs retained under `build/dspine`; NTFS timestamp skew warnings occurred, binary successfully runs requests.
- Added explicit DSpine draft selection/checksum verification to the comparison harness and one adapter regression test (five existing tests now pass). Found the fork fitter omits external-draft memory when it cannot create a standalone DFlash context. Cancelled the first 15-token/all-GPU attempt before requests; retained cancellation, not an OOM or benchmark result. Added actual post-load 512 MiB VRAM reserve check for external drafts, retaining null/low-free refusal and recording fitter omissions.
- Windows-mounted hashing/loading was slow. Temporary copies under Linux `/tmp/qah-dspine-20261109` are prepared and consumed in a continuous WSL session; one earlier cache-preparation attempt was unavailable on the next invocation and produced no measurements. Original weights unchanged. Target/draft checksums are verified for each run.
- DSpine7 with explicit `--gpu-layers 60` completed 13 requests at 8K: coding 5/6, retrieval 6/6, forced 128-token decode 29.09 tok/s; cold/warm retrieval median 10.07/0.551 s. Matched no-spec/MTP controls use the same fork, placement flags, KV, fixtures and seed. Logs confirm 60/66 placement for all variants; target CPU state scales with rollback copies. Additional DSpine3/MTP3 fully-GPU comparison queued sequentially to test a practical lower-memory option.
- Completed five profiles, 65 requests: all request executions successful; 25/30 coding checks pass (same dependency-iterator failure per profile), 30/30 retrieval checks pass, five 128-token decode probes complete. Partial CPU no-spec/MTP3/DSpine7 decode: 20.03/18.48/29.09 tok/s. Full-GPU MTP3/DSpine3 decode: 79.80/95.76 tok/s (20.0% gain in one probe); full-GPU coding output hashes match on all six tasks, five pass. DSpine3 cold/warm retrieval 5.695/0.292 s versus MTP 5.326/0.231 s, so decode gain is not a universal latency improvement.
- Full-GPU DSpine3 sampled peak VRAM 15,571 MiB versus MTP3 14,593 MiB; DSpine had just 525 MiB free after loading. No general lossless-sampling, memory stability, full 15-draft, or 64K+ DSpine claim. Report: `docs/dspine-results.md`. All benchmark servers stopped; temporary Linux model cache absent after session. Existing preferred production/configuration selection unchanged; no commit or push.
- Final checks passed: five harness tests, compilation, whitespace check, all 65 evidence outcomes/counts, exact 7935-token long prompts, five complete decode probes, common reference binary checksum, archived/current source agreement, and actual post-load draft reserve checks. Per-profile and combined summaries saved beside raw results. Benchmark source SHA256 `9f72a723880bcd371a9b36918b5035563d704b836a0bede013375fa5c230f796`.

## Latest results publication — 2026-11-09

- User requested pushing latest results. Current workspace remote is `AYMNrh/ENGINEatHome`; separate QwenAtHome checkout remains unchanged. Prepared DSpine harness changes, both latest reports, and an allowlist of 12 sanitized evidence artifacts plus manifest/README under `bench/evidence/`: 91 completed requests and one separately retained cancelled setup.
- Private paths/machine names replaced and duplicate outputs removed; numeric measurements, correctness outcomes and historical provenance hashes preserved. Full logs, original source snapshots, generated execution directories, weights, binaries and unrelated workspace content remain excluded. Five tests, compilation, whitespace checks, 12 original/published artifact hashes, all outcome counts, preserved numeric fields, privacy scans and report links passed before committing. No force push or history rewrite.
