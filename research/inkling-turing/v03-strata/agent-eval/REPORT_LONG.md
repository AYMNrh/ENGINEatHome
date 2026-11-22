# Long-context coding pilot, final (2026-11-22)
Inkling-Turing (HEAD e0be55a, UD-IQ3_XXS, q8_0 KV, flash attention) vs production Swift 1.5 (gateway, MTP3, Q4 KV, thinking forced on at xhigh). RTX 2080 Ti 22 GB, temperature 0, seed 0.
Harness: `pilot_long.py` (+ `pilot.py` agent loop, 6 tools). Raw data: `results_long/*_results.json`, per-turn timings inside each result, transcripts `*.transcript.jsonl`, server logs `inkling_server_*.log`, session logs `../sessions/`.

## Method
- Deterministic synthetic Python repository, put in the first user message (as a "repository snapshot") and on disk for tools. Files are in a fixed order, so a larger variant shares the prefix of a smaller one.
- Task 1 (needle/dependency): the retry-budget constant sits in the *first* file (`settings.py`); the consuming function is in a module ~25% into the 32K part; a deprecated decoy constant (`RETRY_BUDGET_MS_LEGACY = 5000`) is later. Success = right value + consumer module + function.
- Task 2 (bug fix): `weighted_mean` divides by `len(values)` instead of the weight sum; fix via tools, graded by 7 hidden tests.
- Needle seeds: stage 1 seed 7 / value 7340; stage 2 seed 11 / value 9181 (different text from the first token, so no cache could be reused).

## Actual token counts (first-turn prompt, cached + new)
| Variant | Swift 1.5 | Inkling-Turing |
|---|---|---|
| nominal 32K | 32.3K | 35.9K |
| nominal 64K | 64.2K | 71.6K |
| nominal 128K (stage 2, 170 modules) | 117.7K | 131.3K |
The same text costs Inkling about 12% more tokens. "128K" is nominal: Swift saw 117.7K, Inkling 131.3K (it needed a 139K context).

## Cache conditions
- Stage 1: per model one server/gateway session, tasks in order 32K needle, 32K bug, 64K needle, 64K bug. The 32K bug task and the 64K needle task reused the 32K prefix (slot prefix reuse), so the 64K needle row measured prefill of only the new ~32K/36K tokens at 32K depth. The 64K bug task reused the 64K prefix.
- Stage 2: fresh seed, empty cache for both: genuinely cold 117.7K / 131.3K-token prefill. The bug task then reused the whole prompt (warm second turn).
- Swift: `--cache-ram 0`, `--ctx-checkpoints 4`, slot prefix reuse worked (99.6-99.9%). Inkling: default slot reuse (99.7-99.9%).

## Results
| Task | Result (both models) | Cold TTFT Swift / Inkling | Prefill tok/s Swift / Inkling | Decode tok/s Swift / Inkling | Wall Swift / Inkling | Turns | Bad calls* |
|---|---|---|---|---|---|---|---|
| 32K needle | correct / correct | 72.8 s / 45.6 s | 444 / 790 | 34.6 / 49.9 | 86.9 s / 49.7 s | 2 / 1 | 0 / 0 |
| 32K bug | 7/7 / 7/7 | warm 0.7 s / 0.7 s | - | 35.7 / 49.6 | 24.4 s / 19.1 s | 4 / 6 | 0 / 1 |
| 64K needle | correct / correct | 101 s / 59 s (new half only) | 317 / 610 | 29.9 / 43.5 | 114.7 s / 64.9 s | 1 / 1 | 0 / 0 |
| 64K bug | 7/7 / 7/7 | warm 0.8 s / 1.0 s | - | 30.1 / 43.3 | 29.0 s / 21.8 s | 4 / 6 | 0 / 1 |
| 128K needle (cold) | correct / correct | 409 s / 232 s | 289 / 568 | 24.0 / 35.7 | 424 s / 243 s | 1 / 2 | 0 / 0 |
| 128K bug (warm) | 7/7 / 7/7 | 1.1 s / 1.4 s | - | 26.0 / 35.7 | 32.8 s / 30.6 s | 4 / 6 | 0 / 1 |
*"Bad calls" are calls rejected by the harness policy, not malformed calls; see `REPORT_REAL.md` (root cause: all were valid JSON rejected by the run_command shell-operator rule).
Swift's 32K needle was marked 4/5 by my checker only because its answer mentioned the decoy value; the answer was correct.

## Incomplete / failed
- First Inkling stage-1 launch (`-c 139264 -ub 512`): CUDA out-of-memory at load (14.3 GB compute buffer); 243 s of GPU time lost, no data. Relaunched with `-c 81920 -ub 256` (stage 1) and `-c 139264 -ub 256` (stage 2). The micro-batch is a runtime flag, but it differs from the `-ub 512` used in kernel benchmarks.
- No task hit its timeout. Swift's 128K cold needle used 424 s of a 430 s cap.
- No 256K run (excluded by instruction).

## Limitations
- One run per cell, temperature 0: no variance estimate; differences of a few seconds are noise.
- Synthetic repository; two task types; the answer to the bug task is localised, so the long context mostly tests retrieval and prefix handling, not multi-file reasoning at depth.
- Different tokenizers, so "same text" is not the same token count; per-token rates should be compared with that in mind.
- Prefix sharing between variants means cold-prefill numbers exist only for the first task of each model at 32K and for the fresh-seed 128K prompt.
- Swift ran through the gateway, Inkling directly against llama-server, so gateway overhead (small) is only in Swift's wall times.
- Inkling's thinking was left at the chat-template default; Swift's was forced on at xhigh (production setting), so reasoning effort differs.
