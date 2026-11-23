# Qwen3.8-27B-OBLITERATED profile and RAM-backed prompt caching

Deployed and verified 2026-11-23 on the homelab RTX 2080 Ti (22 GB, 175 W). One model occupies the GPU at a time; **Swift stays the default**.
Everything below is measured on the real runtime; where a number comes from earlier research its original conditions are kept.

| Decision | Outcome |
|---|---|
| Swift RAM prompt cache | **Applied**: `--cache-ram 8192 --no-cache-idle-slots` (all other Swift settings byte-identical) |
| Obliterated profile | **Deployed, status `qualified` for 65,536 context only**; never the default, never auto-activated |
| Obliterated KV precision | **F16** (all checks passed, 1.6 GiB VRAM free at 63.6K); **Q8_0 validated as the fallback** |
| Inkling profiles | **Not modified**. They run the runtime-default RAM cache; see [Inkling exposure](#inkling-exposure-not-changed) |
| Client isolation | 0 contaminated responses in 400 test requests; the upstream bug is **still open and unfixed in our runtime** — residual risk documented below |

## 1. The Obliterated profile

**Model** `qwen3.8-27b-obliterated` (display name "Qwen3.8-27B-OBLITERATED Optimized"), source <https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED>. The existing checkpoint is used; nothing was downloaded again.

| Check | Result |
|---|---|
| File | `~/llm/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`, 16,810,705,952 bytes |
| SHA256 | `1f74330b211a8253c96f1bf586cba6eb56d37117c97ed9e6eec18c198a4e7fe5` (equals the Hugging Face LFS object id; re-verified by `add_profiles.py` on every run) |
| Architecture | `qwen35`, `block_count 65`, `nextn_predict_layers 1`; `blk.64` is a full MTP layer (`nextn.eh_proj`, `enorm`, `hnorm`, `shared_head_norm`, Q4_K/Q6_K), same layout as Swift's |
| Runtime | the stock llama.cpp build already used by Swift (`98d67c8d5`, build 12519); `draft-mtp` initialises and drafts (`draft_n` > 0 on every request) |
| Chat template | **the embedded template is a minimal ChatML with no tool support** (it never mentions `tool`), so assistant tool calls and tool results would be silently dropped. The profile therefore pins the Qwen-style template extracted from the Swift GGUF (`serving/templates/qwen35-tools.jinja`, SHA256 checked before every load) via `--chat-template-file`. Tool calling over SSE and the real OpenCode client were tested with it |

**Launch settings** (the measured line from `research/obliterated-q4km-2080ti/configs/w2_stack_f16_52k.json` plus gateway plumbing and the template/isolation flags):

```text
--model Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf --ctx-size 65536 --parallel 1 --n-gpu-layers 99 --flash-attn on
--cache-type-k f16 --cache-type-v f16 --batch-size 2048 --ubatch-size 512 --threads 8 --fit off --no-context-shift
--cache-ram 8192 --no-cache-idle-slots --ctx-checkpoints 4 --jinja --chat-template-file qwen35-tools.jinja --no-warmup
--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.6 --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 --spec-draft-n-ubatch 64
--host 127.0.0.1 --port <backend> --alias qwen3.8-27b-obliterated --no-webui --log-disable      (175 W power limit is enforced by the gateway)
```

**Thinking**: the template accepts `enable_thinking` and the Swift effort levels. The profile exposes `none, low, medium, xhigh` and **defaults to `none`** because every speed measurement was taken with thinking off; change it on the dashboard or per request. The effect of the levels on this model's behaviour was not measured.

**Context**: 65,536 only. Larger contexts were not tested and are not claimed.

### KV precision: F16, with Q8_0 as the validated fallback

Both were run through the same qualification on the staging gateway (same 63,591-token prompt, thinking off, temperature 0):

| | F16 (selected) | Q8_0 |
|---|---|---|
| Prefill / decode at 63,591 tokens | 408.5 / 37.8 tok/s | 403.8 / 33.5 tok/s |
| Cached follow-up TTFT (63,656 cached) | 0.51 s | 0.60 s |
| Loaded VRAM / peak (soak) | 20,820 / 20,876 MiB | 19,150 / 19,174 MiB |
| Free VRAM at the 63.6K peak | **1,680 MiB** | 3,378 MiB |
| Soak (depth ~48K, side requests) | 12 min, 141 requests, 0 errors, VRAM drift +26 MiB, median decode 39.3 tok/s | 8 min, 100 requests, 0 errors, median decode 36.0 tok/s |
| Failed-load recovery, tool calling, interleaved restore | pass | pass |

F16 was chosen because it showed no memory instability and keeps more than the 1 GiB margin set before testing (the gateway's own post-load guard is 512 MiB; Inkling Balanced already runs with a similar ~1.5 GiB margin). To fall back to Q8_0:

```bash
cd ~/llm/serving && ~/llm/serving-venv/bin/python add_profiles.py --obliterated-kv q8_0 --obliterated-measured <measured.json> && sudo systemctl restart engineathome-api.service
```

### Existing performance evidence (preserved with its conditions)

| Measurement | Value | Conditions |
|---|---|---|
| 65K decode, original | 16.01 tok/s | no MTP, Q4_0 KV, ub 128, random-token prompt, greedy, 4 prompt sizes, single run each |
| 65K decode, MTP depth 2 | 24.10 tok/s | same prompt; acceptance 65.9% |
| 65K decode, optimised stack | 29.50 tok/s | same prompt and 120-token answer; F16 KV, depth 3, p-min 0.6, ub 512; prefill 404 tok/s; peak 20.4 GiB; needle found |
| 52K agent-simulator decode | 33.70 vs 28.15 tok/s control | production sampling (temp 1.0, top_k 20), 4 turns, controls bracket the candidate (28.15 / 28.12) |

The qualification numbers above (37.8 tok/s at 63.6K) use a different prompt and answer, so they are **not** an improvement over 29.50; decode speed depends on how predictable the answer is. See [research/obliterated-q4km-2080ti](../research/obliterated-q4km-2080ti/README.md) for the full ledger.

## 2. RAM-backed prompt caching

Production ran every profile of the stock runtime with `--cache-ram 0`, so any unrelated request (an OpenCode title call, another client) discarded the cached conversation and the next turn re-read the whole prompt. With the RAM cache the conversation is saved when another request takes the slot and restored when it returns.

### Verified benefit (Obliterated and Swift, measured through the gateway)

| Scenario | Cold / before | With RAM cache |
|---|---|---|
| Obliterated, ~12.6K-token conversation, follow-up turns | 22.3 s first turn | 1.2–1.4 s (97% cached) |
| Obliterated, 63,591-token conversation, follow-up | 158 s cold | **0.51 s** (63,656 of 63,681 cached) |
| Obliterated, after a title call and an unrelated 6.7K prompt from another client | ~30 s re-read | **0.2 s** (14,041 of 14,045 cached) |
| Swift, three 13–20K conversations resumed in turn (A, B, C, A) | 25–41 s each | 0.2–0.5 s each, all restored |
| Earlier research (Obliterated) | 44.1 s at ~20K, 131.6 s at 52K | 4.3 s, 5.9 s (preserved from the research report) |

A conversation restored after an unrelated request costs ~0.2–0.5 s instead of re-reading everything.

### Overhead

| | Measured |
|---|---|
| VRAM | none: the cache is host memory. Obliterated stays 20,820 MiB loaded (F16) |
| Host RAM, Swift | +2.1 GB resident after three conversations (≈52K cached tokens): 873 → 2,995 MiB; the 8 GiB cap bounds it |
| Host RAM, Obliterated | steady 4.4–4.6 GB resident after a 63.6K conversation; peak 9.6 GB during the F16 soak; host available never below 20 GB of 31 GB |
| Pool limits | at most 8,192 MiB, plus a token budget that starts at the context size (65,536) and grows only while the memory limit allows; oldest entry removed first |

### Client-isolation investigation (llama.cpp issue #27148)

**What the issue says.** The server's RAM cache can restore state from a finished, unrelated conversation into a new request, which then continues the stale conversation; `cached_tokens` stays 0 so clients cannot see it. Reported on ROCm, Metal and with `-np 2`; one reporter reproduced it with strictly sequential single-user requests on defaults. Reported mitigation: `--no-cache-idle-slots` together with `--cache-ram 0` (either alone is enough only when `kv_unified` is on).

**What our runtime contains.** Checked in the build's source and with the live issue/PR status (2026-11-23): the issue is **open**; the fix PR #27624 ("clear stale state when reusing LRU-selected slots") and #28058 are **open and unmerged**; our build (`98d67c8d5`) still has `server_prompt_cache::load()` returning `true` when no entry matches, the defect described in the report. The fix is therefore **not** present. Our topology differs from the reports (one slot, `--parallel 1`, hybrid GDN model with MTP and context checkpoints), so the bug was tested rather than assumed.

**How it was tested** (`serving/validate_cache_isolation.py`): two independent clients with unique random markers and disjoint real-code documents; five scenarios per round: strictly sequential unrelated requests, interleaved multi-turn follow-ups, concurrent requests, a title side request followed by the original conversation, and the issue's own single-user repro (a short prompt with a 1,200-token thinking generation followed by an unrelated long prompt). Every response is checked for any foreign marker and for its own marker.

| Configuration | Scale | Requests | Contaminated | Wrong answers | Cache hits (tokens restored) |
|---|---|---|---|---|---|
| Swift, `--cache-ram 0` (production before) | 2K-token docs | 48 | 0 | 0 | 0 |
| Swift, `--cache-ram 8192` | 2K | 48 | 0 | 0 | 23 (44,875) |
| Swift, `--cache-ram 8192 --no-cache-idle-slots` | 2K | 48 | 0 | 0 | 23 (44,875) |
| Swift, `--cache-ram 0 --no-cache-idle-slots` | 2K | 48 | 0 | 0 | 0 |
| Swift, `-np 2 --kv-unified --cache-ram 0 --no-cache-idle-slots` | 2K | 48 | 0 | 0 | 12 (26,754) |
| Swift, `--cache-ram 0` (production before) | 12K | 40 | 0 | 0 | 0 |
| Swift, `--cache-ram 8192` | 12K (503K prompt tokens) | 40 | 0 | 0 | 15 (178,800) |
| Swift, `--cache-ram 8192 --no-cache-idle-slots` (deployed) | 12K | 40 | 0 | 0 | 15 (178,800) |
| Obliterated, `--cache-ram 8192` (idle-slot default; see note below) | 12K | 40 | 0 | 0 | 14 (178,519) |

Light scale is shown because the issue warns that lightweight payloads give false negatives; the 12K rows are the real-scale check. One Obliterated response named a file (`mmvq.cu`) that was not in its own document; that is a real llama.cpp file the model knows by name while reviewing llama.cpp code, not an injected marker, and no foreign marker appeared in any response. Cache hits occur only where they should (follow-ups and the turn after a side request), never for unrelated prompts.

**Decision: applied, with hardening and an explicit residual risk.**
- Both profiles run `--cache-ram 8192 --no-cache-idle-slots`. The flag closes the issue's documented idle-slot publishing path and changed nothing measurable (identical hit counts).
- 0 contaminated responses in 400 requests (240 at 2K scale, 160 at 12K scale). 120 of the 12K-scale requests ran with the RAM cache enabled; by the rule of three that bounds the per-request rate near 2.5% at 95% confidence. This is **evidence, not proof**: the defect is unfixed upstream and timing-dependent.
- The `--no-cache-idle-slots` flag was isolation-tested on Swift (identical behaviour to the default). The exact Obliterated flag set was tested at 12K scale with the default idle-slot setting, and the deployed set (hardening flag added) was covered by the qualification runs, where no foreign marker appeared; it was not run through the full matrix.
- The gateway's three inference clients (`opencode`, `hermes`, `python`) are the owner's own tools on a private tailnet, so a leak would cross the owner's own sessions, not tenants. If that assumption stops being true, run Swift with `--cache-ram 0 --no-cache-idle-slots` (clean in the same tests, no cache benefit) or use `--parallel 2 --kv-unified` (also clean, partial benefit) and re-run the harness.
- Re-check when #27624 merges; re-run `python validate_cache_isolation.py --config-name <variant> --doc-tokens 12000 --long-generation 1200` after any runtime rebuild (the production gateway must be stopped; it uses one GPU).
- Model switching: the cache lives inside the `llama-server` process, which every switch, unload and restart terminates. Verified live: the first request after Swift → Obliterated → Swift → Obliterated always reported 0 cached tokens for a prompt the previous model had cached. No state is ever written to disk (`--slot-save-path` is not used), so nothing can be restored across models.

### How the cache actually behaves (measured, worth knowing)

- **A prompt that shares a long prefix with a cached conversation consumes that entry.** The server restores the cached state to reuse the shared prefix, which moves the entry out of the pool. In the qualification run where the "unrelated" document came from the same source files as the conversation (5,525 shared leading tokens, 35% of the 15,974-token entry; the server's threshold is 25%) the conversation was not restorable afterwards and cost a full 29.7 s re-read. The output was correct, only slower. Evidence: `research/obliterated-q4km-2080ti/isolation/pool-consumption-evidence.log`. A large main conversation sharing a small system prompt (for example 6K of 50K, 12%) is below the threshold, so OpenCode's usual pattern is safe; sub-agent conversations that share a long prefix with a small cached conversation are not.
- The restore path for this hybrid model falls back to a full re-read when no checkpoint matches ("forcing full prompt re-processing"), so stale state is not blindly reused.
- The pool is bounded (8 GiB and a token budget that starts at the context size), so large unrelated prompts can push older entries out; a 34K conversation survived a 42.6K unrelated prompt in testing, but there is no guarantee. The failure mode is always a slower re-read, never a wrong answer.

### Inkling exposure (not changed)

The Inkling profiles launch without `--cache-ram`, so they run the runtime default (8,192 MiB) **with idle-slot caching on**: exactly the default configuration named in the issue. Per instruction they were not touched. They were not part of the isolation matrix (the Inkling fork was not tested). Recommended follow-up: add `--no-cache-idle-slots` to both and test them with the harness. The dashboard now shows each profile's RAM-cache setting, so this is visible.

## 3. Home Assistant

All four profiles are in the existing ENGINEatHome dashboard and package (no separate interface). The profile list, target selector and every table are generated from the gateway's own registry (`profiles.py`), so they cannot advertise something the gateway lacks.

- **Select never switches.** Choosing a target only records it; the Overview shows a comparison of the target and the running model (context, quantization, MTP, RAM prompt cache, qualification, KV, microbatch, VRAM, default thinking) and the explicit **Apply model** confirmation. Apply keeps all existing protections (refused while requests or leases are active, real inference probe, restore on failure).
- **New "Model details" card** for the served model: model name and ID, loaded/not loaded, context capacity, quantization, KV precision, MTP on/off with draft depth and threshold, RAM-cache setting, VRAM used now and expected, recent prefill/decode speed, cache reuse (hit rate over the last 20 requests and the last request's cached tokens), cached follow-up TTFT versus cold TTFT, model-process RAM and host RAM available, and the qualification status and text.
- **New sensors**: `sensor.engineathome_cache` (attributes: enabled, limit, label, recent statistics, process and host RAM) and tiles for hit rate, cached follow-up TTFT, cold TTFT, runtime RAM, host RAM, last cached tokens, recent prefill and decode speed. The model comparison and launch-settings tables gained quantization, MTP, RAM cache and status.
- **Gateway status** (`GET /admin/status`) gained a `cache` block and each profile gained `quantization`, `mtp`, `ram_cache` and `qualification_status`. A profile with status `blocked` cannot be applied (`profile_blocked`); a changed or missing chat-template file is refused before anything is stopped.
- Verified: `check_config`, the offline suite (17 dashboard tests incl. all-null/empty/unavailable data), the isolated Home Assistant instance against the real gateway, and the production Home Assistant polling the upgraded gateway after restart.

## 4. Qualification results (staging gateway, then production)

The qualification ran on a staging gateway (own port, own usage database, production unloaded) and the final steps on production. `serving/qualify_obliterated.py` is the repeatable script.

| Check | Result |
|---|---|
| Native MTP initialisation and acceptance | active on every request; acceptance 72–73% on 4–6K coding prompts (three prompts), 90% (47 of 52 drafts) at 63.6K |
| Short answers | correct at `none` and `low` (math, capital, code) |
| 16K and 63.6K coding prompts | needle (a passphrase in the middle) found at every size; 63,591 actual prompt tokens: 408.5 tok/s prefill, 37.8 tok/s decode |
| Multi-turn prefix restoration | follow-ups 97% cached; 12.6K first turn 22.3 s → 1.2–1.4 s |
| Interleaved unrelated requests | title call plus an unrelated 6.7K prompt from another client: original conversation restored, 14,041 of 14,045 tokens cached, no foreign marker |
| Tool calling (streamed, as OpenCode does) | valid `read_file` call at `none` and `xhigh`; the follow-up with the tool result produced the right final answer, with no stray tool call |
| Real OpenCode 1.18.16 | solved a planted-bug task through `read`, `read`, `edit`, `bash`, and the project's tests pass afterwards. Staging: `none` 29.7 s, `xhigh` 31.5 s, default 29.2 s. **Production endpoint with the user's own key**: `none` 32.4 s, default 30.8 s |
| Thinking and template | `none` produces no reasoning; `low`, `medium`, `xhigh` return reasoning and the right answer |
| GPU memory headroom | 1,680 MiB free at the 63.6K peak (F16); 3,378 MiB (Q8_0) |
| Stability | 12-minute F16 soak (141 requests, 0 errors, VRAM +26 MiB); 8-minute Q8_0 soak (100 requests, 0 errors) |
| Failed-load recovery | `selftest-oom` (a context that cannot fit): `503 backend_exited_1`, the previous profile restored, one process, inference answered, with F16 and Q8_0 |
| Switching | Swift ↔ Obliterated through the production gateway: 4.4–16.6 s; one process each time; cache never crosses models; a request naming the other model gets `409 model_not_active` |

**Speculative decoding output equivalence** was not assumed. At temperature 0 the MTP and non-MTP outputs were byte-identical on the 4K and 16K prompts of the research run, but outputs diverge after the first turn at other draft depths (numerical differences between 1-row and k-row verification), the logits at the divergence point were not inspected, and every qualification prompt's answer was checked semantically (needle, correct tool call, correct answer), not for byte identity.

**Not verified**: soaks longer than 12 minutes; contexts above 65,536; thinking-mode quality of this model; real OpenCode sessions beyond the single bug-fix task; the Inkling profiles' cache isolation; production Home Assistant UI interaction by hand (the isolated instance and entity polling were verified).

## 5. Use it

1. Open the ENGINEatHome dashboard in Home Assistant, choose **Qwen3.8-27B-OBLITERATED Optimized** as Target model, confirm the details, press **Apply model** (about 5–17 s, refused while a request or lease is active).
2. OpenCode's provider already points at the gateway. The model entries `obliterated`, `obliterated-no-thinking`, `obliterated-low`, `obliterated-medium` and `obliterated-xhigh` were added to `~/.config/opencode/opencode.json` (context 65,536). Requests must name `qwen3.8-27b-obliterated`; another profile's ID is refused with 409, never answered by a different model.
3. Swift remains the default: after a gateway restart the applied profile is always Swift, and the first request loads it.

## 6. Deployment, backups and rollback

Backups: `~/llm/serving/backups/obliterated-20261123T054953Z/` (code, `config.json`, previous HA package) and `config.json.before-profiles-20261123T065047`; Home Assistant's own `engineathome/backup-20261123T055652Z/`. The model files, the research runtime and every research artifact are untouched; nothing was downloaded.

Roll back everything:

```bash
sudo systemctl stop engineathome-api.service            # waits for in-flight requests
cd ~/llm/serving && B=backups/obliterated-20261123T054953Z
cp $B/config.json $B/profiles.py $B/lifecycle.py $B/store.py $B/add_profiles.py $B/ha_dashboard.py $B/ha_files.py .
rm -rf templates
sudo systemctl start engineathome-api.service
# Home Assistant: copy engineathome/backup-20261123T055652Z/* over engineathome/, then: sudo docker restart homeassistant
```

Roll back one thing:
- Swift cache only: `python add_profiles.py --swift-cache-ram 0` (restores `--cache-ram 0` and removes `--no-cache-idle-slots`), restart the gateway.
- Remove Obliterated only: delete the `obliterated` key from `config.json`, restart; or set its `qualification_status` to `blocked` to make Apply refuse it while keeping it visible.

## 7. Files

`serving/profiles.py` (profile builder, cache/MTP facts), `serving/lifecycle.py` (preflight, cache and memory status), `serving/store.py` (cache statistics), `serving/add_profiles.py` (verification, template extraction, Swift cache option), `serving/ha_files.py` and `serving/ha_dashboard.py` (sensors and cards), `serving/test_obliterated.py` (10 new tests; 42 pass with the existing suite), and the repeatable tools `validate_cache_isolation.py`, `cache_restore_probe.py`, `qualify_obliterated.py`, `stage_qualification.py`, `validate_opencode_client.py`. Raw evidence: `research/obliterated-q4km-2080ti/{isolation,qualification,opencode}`.
