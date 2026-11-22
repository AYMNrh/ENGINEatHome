# Multi-model inference control center

Deployed 2026-11-22. One RTX 2080 Ti (22 GB, 175 W) serves **one model at a time** through the existing private gateway; Swift stays the default, and Inkling 2 Mini is available in two validated configurations. Models are selected, applied and tuned from the Home Assistant dashboard ([serving-dashboard.md](serving-dashboard.md)). No CUDA kernel, benchmark configuration or Swift inference setting changed.

## Profiles (verified against the homelab, not assumed)

| | Swift Production | Inkling Balanced | Inkling Maximum |
|---|---|---|---|
| Model ID | `swift-1.5-qwen3.8-27b` | `inkling-2-mini-balanced` | `inkling-2-mini-maximum` |
| Model | Qwen3.8-27B Swift 1.5 GSQ-RCO IQ3_S (MTP) | Inkling 2 Mini UD-IQ3_XXS (mainline GGUF) | same file |
| Runtime | stock llama.cpp build 12519 (unchanged) | llama.cpp fork `relpos-gpu`, validated build `bin-rp2` | same binary |
| Context (configured) | 262,144 | 204,800 | 262,144 |
| KV cache | Q4_0 | Q8_0 | Q8_0 |
| Microbatch | 128 (batch 512) | 512 | 256 |
| Other | MTP3, FA, 99 GPU layers, `--cache-ram 0` | FA, `-ngl 99`, one sequence, `--jinja`, prefix cache on | same |
| Loaded VRAM | ~19.0 GiB expected (observed 19,614 MiB) | ~20.1 GiB (observed 20,558 MiB) | ~19.4 GiB (observed 19,840 MiB) |
| Thinking | `none, low, medium, xhigh`; default `xhigh` | `none, minimal, low, medium, high, xhigh`; default `high` | same as Balanced |

- Inkling launch lines are the qualified commands from the long-context report (`-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c <ctx> -ub <ub> -np 1 --jinja --no-warmup`) plus only what the gateway needs: loopback host/port, alias, `--no-webui`, `--log-disable`. Context 204,800 uses ubatch 512 (qualified at 127,671 and 185,099 actual input tokens). Context 262,144 uses ubatch 256 (qualified with **242,134** actual input tokens, 246.7K occupied).
- **262,144 is configured capacity, not a demonstrated maximum input.** Nothing above 246.7K occupancy was tested. ubatch 512 at 262,144 fails to allocate (7.40 GiB compute buffer) and is deliberately not offered; it is used only as the failure-injection profile in staging tests.
- Runtime provenance: `serving/add_profiles.py` pins the research build into `~/llm/runtimes/inkling-7d3e5b1-bin-rp2/` (9 files, SHA256 recorded in its `MANIFEST.json`) and hard-links the GGUF next to the Swift model (SHA256 `fd9ccedae17c…1657`, verified). The binary reports commit `e4a7ea6` in `--version`: it was built (16:43) before 7d3e5b1 was committed (16:47) from the same tree, and the research reports name `bin-rp2` as the 7d3e5b1 baseline. Identity is therefore pinned by hash, not by the embedded string, and validated behaviourally (loads, answers, matches the documented VRAM). It was not rebuilt.
- GPU guards are per profile: free VRAM >= measured peak + 512 MiB, per-process residency floor (weights + KV), 175 W power limit, no other CUDA process, model file unchanged, backend port free.

## Switching

Vocabulary: **selected** (a UI choice, persisted, never touches the runtime), **applied** (what the gateway serves and what an on-demand load starts), **active** (the process on the GPU, none while unloaded), **default** (applied after a clean gateway start; Swift).

Apply sequence (`POST /admin/profile/apply`, all under the existing lifecycle lock, run detached from the HTTP request so a disconnect cannot leave a half-finished switch):
1. Refuse (HTTP 409, nothing touched) if requests are active, a session lease is held, a load is running or the native slot is busy.
2. Static preflight of the target (model file, executable) before anything is unloaded.
3. Stop the owned process, wait until the GPU really has the room the target needs.
4. Load with the per-profile guards, then send a **real tiny inference request** to the new runtime. Success is reported only after that answer.
5. On any failure: clean up, then restore the previous profile, then the configured default (Swift); the outcome (`last_switch`: ok, error, restored) is exposed and shown in the dashboard.

Measured on the homelab (warm page cache; cold-disk loads were not measured): Swift 3.4-3.9 s, Balanced 4.2-4.3 s, Maximum 4.3 s, apply from the dashboard 4.8 s end to end (phases `unloading, loading, verifying_inference` observed in `/admin/status`). One `llama-server` process and 19,614-20,558 MiB VRAM after every switch.

**The prefix cache does not survive a switch** (or Unload/Restart): after Swift -> Balanced -> Swift the first request reported `cached_tokens 0` and the repeat 3,036 of 3,040. The UI says so before Apply.

### API (control scope; inference keys cannot use these)

| Route | Purpose |
|---|---|
| `GET /admin/profiles` | all profiles: model ID, context, KV, microbatch, VRAM, thinking levels, launch settings (no paths/checksums) |
| `GET /admin/profile` | selected / active / applied / default, state, switch phase, last switch |
| `POST /admin/profile/select {profile}` | record the target; **never switches** |
| `POST /admin/profile/apply [{profile}]` | apply the selected (or named) profile as above |
| `POST /admin/load`, `/unload`, `/restart` | unchanged; Load starts the applied profile |
| `PATCH /admin/settings` | existing keys plus `default_profile` and `reasoning_effort` (+ optional `profile`) |
| `GET /admin/status` | existing `model.*` keys kept; adds `profile`, `profiles`, `reasoning`, `model.active_model_id` |

Gateway status distinguishes gateway health, selected/active/applied profile, runtime state (`unloaded, loading, ready, switching, unloading, failed`), requests, leases, errors, capacity and the model configuration.

### Which model answers a request

A request names exactly one model. Unknown ID: 400 `unknown_model`. A known ID that is **not** the served model: 409 `model_not_active` with the served model and how to apply the other one; it is never answered by another model and never triggers a switch. No `model`: the applied profile. `/v1/models` lists the served model (`loaded`, `context`, `profile`). Existing clients, keys, leases, auto-unload, on-demand load (of the *applied* profile, so an idle-unloaded Inkling reloads as Inkling) and Tailscale binding are unchanged.

### Usage attribution

The ledger gained `profile` and `effort` columns (nullable, additive). Historical rows were attributed to `swift` on first start (282 of 282). Every served request is tagged with the model/profile that served it and the effort forwarded; rejected requests keep the model they asked for and no profile. Still metadata only.

## Thinking effort

Read from the models' own chat templates (GGUF metadata) on the homelab:

- **Inkling** (`inkling2_loop`): template variable `reasoning_effort`, a name (`none 0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh 0.99, max 0.99`) or a number 0-0.99; default 0.9 (= high). It renders `Thinking effort level: <n>` as a system message. `max` and `xhigh` are identical, so only `xhigh` is offered (`max` is accepted as an alias).
- **Swift** (`qwen35`): `enable_thinking` (bool) and `reasoning_effort` in `xhigh` (default), `medium`, `low`. `none` is offered as `enable_thinking=false`. Swift's `medium` renders **no instruction text** (a template quirk: it equals the neutral prompt); `low` and `xhigh` add distinct instructions.

Forwarding: the gateway resolves the effort per request (template kwargs, then the top-level OpenAI `reasoning_effort`, then the dashboard default for the served profile, then the model default; same precedence as before), validates it against that model's list (400 `unsupported_reasoning_effort` with the supported list; nothing reaches the model), and sets `chat_template_kwargs.reasoning_effort` (or `enable_thinking=false`). The resolved value is returned as `X-Reasoning-Effort` and recorded. A request's value never changes the stored default or another client's requests. Changing the default (dashboard or `PATCH /admin/settings`) needs no reload. Reasoning content and tool calls pass through untouched; Swift's sampling and `enable_thinking`/`preserve_thinking` defaults are unchanged (Inkling gets none).

Verified on the real runtimes (staging gateway, then production):
- **Rendered template** (`/apply-template` with the forwarded kwargs): every Inkling level renders its exact number (0, 0.1, 0.2, 0.7, 0.9, 0.99; `max` -> 0.99; `0.5` -> 0.5; default 0.9); Swift renders `low` / `xhigh` instructions, nothing for `medium`, and an empty think block for `none`.
- **Through the gateway**: changing the level changes the early prompt line, so the prefix cache misses (high -> low: `cached 0`; repeat: 2,764 of 2,768). Swift `low` and `xhigh` differ in prompt tokens (54 vs 66).
- **Real requests** succeed at every advertised level on both models.
- **Not claimed**: that a level changes how long the model actually thinks. A moderately hard question gave non-monotonic token counts (single samples, default sampling). Requested effort is not a measurement of internal effort. For Inkling `none` is effort 0, not "thinking disabled": a short reasoning may still be produced.

OpenCode 1.18.16 request format, captured from the real client: with a plain model entry it sends no `reasoning_effort` on agent requests (so the dashboard default applies) but sends `low` on its own title-generation requests (supported by both profiles); `options.reasoningEffort` pins a level on every request; `options.chat_template_kwargs` is passed through. The dashboard's snippet follows this.

## Validation summary

| Layer | Result |
|---|---|
| Offline tests (`serving/test_serving.py`, `test_ha_dashboard.py`) | 44 pass: registry validation; selection never switches; apply; invalid profiles; active-request and lease protection; concurrent lifecycle operations; model-ID routing; auto-unload and on-demand reload of the applied profile; failure recovery (previous, then Swift, then nothing); usage attribution and legacy migration; reasoning defaults, overrides, isolation and rejection; status consistency; dashboard entities, cards and charts |
| Live, staging gateway (own port/DB, real GPU) | Swift -> Balanced -> Maximum -> Swift; every reasoning level on both models; apply/unload/restart refused during a running request (request completed untouched); apply refused with a lease; **real OOM failure** (ubatch 512 at 262,144, runtime exits 1): cleaned up, Swift restored, Swift answered, one process, VRAM back |
| Live, production gateway | Swift -> Balanced -> Swift smoke; the same through the Home Assistant UI (select, confirm dialog, switch, thinking default, per-request override, switch back) |
| Home Assistant 2026.10.2 | `check_config`; `validate_ha_history.py` in an isolated instance against the real API with control services stubbed (selection, reasoning options follow the loaded model, one apply path, all cards and filters) |

Tools: `serving/validate_profiles_live.py` (`--full --oom-profile`), `serving/validate_reasoning_live.py`, `serving/validate_ha_history.py`. They run on the homelab with short requests only; no long-context benchmark was run.

## Operation and rollback

Add or refresh profiles: `~/llm/serving-venv/bin/python add_profiles.py` (idempotent, backs up `config.json`, verifies hashes). Regenerate the Home Assistant package: see [serving-dashboard.md](serving-dashboard.md). Gateway change: `sudo systemctl restart engineathome-api.service` (drains).

Rollback: restore `gateway.py`, `lifecycle.py`, `store.py` and `config.json` from `~/llm/serving/backups/multi-model-<stamp>/` (the previous HA files are in `ha-installed/` of the earlier `dashboard-redesign-*` backup and in Home Assistant's `engineathome/backup-*`), restart the gateway and Home Assistant. The ledger change is additive; older code ignores the new columns.

## Limitations

- Switch times are warm-cache; a cold disk read of a 11-12 GB model will take longer.
- 262,144-token Maximum is qualified only to 242K actual input; a cold 242K prefill takes ~9 minutes.
- Only one lifecycle operation and one model at a time; the first request after any switch re-reads its whole prompt.
- Effort levels are forwarded as requested; their effect on behaviour was not shown to be monotonic.
- OpenCode's variant selector can send levels a model lacks (e.g. `high` to Swift); the gateway answers 400 with the supported list instead of guessing.
- The gateway is a single process: `/admin/status` shows live state, but a crash of the gateway itself unloads nothing automatically beyond systemd's restart (the native child is owned and stopped with it).
