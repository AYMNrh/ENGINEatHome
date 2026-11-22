# v0.3 real-repository coding pilot and tool-failure root cause (2026-11-22)
Harness: `pilot_real.py` (agent loop `pilot.py`, same 6 tools, temperature 0, seed 0, 180 s/task, 30 turns, max 6000 tokens/turn). Raw data: `results_real/*.json`, transcripts, `inkling_server.log`, `../sessions/130326-k_preal.log`.

Tasks are four real upstream fix commits. The parent commit is the workspace (no .git), the commit's tests are the hidden grading. Each was validated on CPU (parent fails, real fix passes):
- **boltons_addlist** (a1ea1030ed): `OrderedMultiDict.addlist` with one-shot iterators. Needs a fix in `boltons/dictutils.py` and in `boltons/urlutils.py` (multi-file; I added a hidden urlutils test).
- **boltons_barrel** (e00c10a809, listutils part only): `BarrelList` bounds/insert/pop bugs, must match list semantics (8 tests).
- **cachetools_setitem** (39b31bc9b6): `Cache.__setitem__` over-evicts when growing a key (312 tests).
- **humanize_tzaware** (a47a89ea91): naturalday/naturaldate with tz-aware datetimes (383 tests; a stub `_version.py` is generated).

Pass = the whole target test set is green at the end. Swift ran through the gateway (production config); Inkling ran on llama-server `bin-hk`, `-c 32768`, q8_0 KV. Swift's 4th task was skipped by its session budget and run separately afterwards (label `swift_h`). This was not the SWE-bench/Docker harness (no images offline); the tasks are real commits run natively in a venv.

| task | Swift 1.5 | Inkling-Turing |
|---|---|---|
| boltons_addlist (2 files) | tests green (157/157) but run ended by the 180 s timeout; 18 turns, 24 calls, 4 refused | FAIL 155/157; 30-turn limit at 98 s; 0 edits written; 11 of 31 calls refused by the harness policy |
| boltons_barrel | FAIL 5/8, timeout, 1 edit | FAIL 5/8; cut by the 6000-token turn cap after 17.8K chars of reasoning with no action; 0 edits |
| cachetools_setitem | FAIL 305/312, timeout (the final turn alone ran 149 s rewriting a ~400-line file) | FAIL 306/312; 30-turn limit; 1 edit; 13 of 30 calls refused |
| humanize_tzaware | FAIL 382/383, timeout, 2 edits | PASS 383/383 (fix in place), ended by an HTTP 500 at 168 s (see root cause) |

Totals: tests fully green at the end 1/4 vs 1/4, no first-attempt successes. Decode 37-40 tok/s (Swift) vs 52-54 (Inkling). Median TTFT 0.4-2 s vs 0.6-0.7 s. Later-turn prefix reuse 87-90% vs 76-96%. GPU time: Swift 545 s + 181 s, Inkling 571 s.

**Reading:** this harness cannot rank the models. 22% of Swift's calls (12 of 54) and 37% of Inkling's (32 of 87) were refused by my run_command policy (no shell operators, no sed/git), which pushed both into rewriting whole files or wasting turns. Treat the four rows as inconclusive within these limits; only the throughput numbers are reliable.

## Root cause of Inkling's "malformed tool calls"
1. **Earlier pilot (13 calls flagged "bad": Swift 7, Inkling 6).** All were valid JSON with valid tool names and arguments. Replaying them showed each was refused by my own `run_command` policy (heredoc, `&&`, pipes, `$`, redirects, `rm`). It was not a model, template or parser fault; my counter was mislabelled `bad_calls`. Across all 107 assistant turns of the earlier pilot runs (55 Inkling, 52 Swift): 0 invalid JSON, 0 unknown tools, 0 tool markup leaked into content.
2. **Real-repo run: one genuine malformed call** (humanize, turn 18), HTTP 500 `Failed to parse tool call arguments as JSON ... missing closing quote`. That turn ended with `finish_reason=length` after only 370 tokens because the context was full (max context used 32,398 of `-c 32768`). llama-server then parsed the truncated arguments strictly and returned 500 instead of returning the truncated message with `finish_reason=length`. Origin: the context cap (my setting) plus server strictness on truncated output. The model output was cut, not wrong. The right fix is a larger context or client handling of length-finished tool calls, not loosening the parser (that would hide a truncation).
3. **Real model behaviour differences.** Inkling treats `run_command` as a shell: 18 chained/piped commands in 87 calls plus sed/git, although the tool description says "no shell operators"; Swift did the equivalent about 4 times in 41 calls. Inkling also spent a whole 6000-token turn reasoning before acting (barrel task). Neither is a malformed call.

**Unverified:** I did not rerun with a larger context to confirm no further malformed output, and I did not dump the rendered chat-template prompt or the raw model text.
**Candidate fixes (none applied):** a sandboxed real shell or a stricter argv-array schema for `run_command`; at least 65K context for agent sessions; client-side handling of `finish_reason=length`.
