# Coder workflow campaign: coding passed, restart recall failed

The campaign stopped at its stability gate. No128K agent attempts were started. Official stability runtime-stop returned `ok:true`, and native Strata was verified absent. Earlier scores and raw failures are preserved.

| Stage | Outcome |
|---|---|
| Guided four-task comparison | 4/4 strict, 4/4 patch; baseline2/4 strict, 3/4 patch |
| Six issues times three attempts | 18/18 strict and18/18 patch |
| Stability | Failed after1,865.76seconds (31.10minutes), before the full hour |
|128K agent work | Skipped: stability gate failed |

Stability completed75 API turns, including72 repository-review turns. Saved pre-restart history includes281 completed tool calls and4 tool error states. Explicit compaction plus marker/invariant recall and actual tool execution passed around minute15. At the minute30 process restart, all452 message IDs were retained; before/after history JSON is identical. The subsequent tools-disabled recall returned `{"marker":"unknown","invariants":[]}` instead of the random session-only marker and three ordered invariant IDs.

Of six saved compaction summaries before restart, the first four contain the expected marker and `preserve_patch`; the latest two contain neither. This points to summary/context retention as a likely cause rather than loss of persisted history. It does not establish whether summary generation, active-context selection or another request-state detail caused the failed recall. Restart tool execution, final patch-preservation comparison and final held-out grading were not reached; they remain unvalidated.

Further offline audit: the first marker-free summary was created25m26s into the session, and the next marker-free summary at29m40s, both before restart. The former is only714characters and contains an ordinary review of code/error handling; the latter is12,028characters of repository-review summary with neither the random marker nor invariant IDs. The failed recall request used14,741 fresh input tokens, despite452 persisted messages. Thus the information was already missing from the latest summaries before the restart check; restart timing alone cannot establish restart as the cause. The exact serialized API input was not captured, so summary selection remains an inference requiring a focused transport trace to prove.

Failed-stability telemetry:87.84 GPU Wh over1,867.75covered seconds,169.31W mean/278.88W peak,68.64C maximum,90.33% mean GPU utilization. Sampled peak VRAM15.12GiB, native RSS32.02GiB, minimum available Windows physical RAM13.25GiB. Energy is device-wide GPU-only; CPU/wall energy is unavailable. RSS may include mapped/shared pages. The agent stream alone is used; parent runtime telemetry overlaps and is excluded.

Nonoverlapping agent-stage energy: guided4 24.63Wh, guided18 117.55Wh, failed stability87.84Wh; combined230.02Wh. Loading gaps and CPU/wall energy are not included. The failed stability stage must not be blended into coding-task efficiency.

Next proposed investigation: inspect the OpenCode active-context/summary sequence and preserve critical session constraints across summaries, then run a separately labeled focused restart-recall regression before another full stability attempt. No automatic GPU retry or128K promotion is performed.

Evidence: `bench/results/strata-coder-workflow-20261111/status.json`, `guided18/comparison.json`, `stability/runtime-stop.json`, and `stability/stability/{result,definition,turn-after-restart-recall,history-before-restart,history-after-restart}.json`. All artifacts remain local; no Git publication was requested.
