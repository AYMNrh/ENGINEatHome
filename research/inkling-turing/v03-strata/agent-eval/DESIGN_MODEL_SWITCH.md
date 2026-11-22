# Design: Inkling-Turing as a second selectable model in ENGINEatHome (not applied to production)

## What exists (read from ~/llm/serving: gateway.py, lifecycle.py, store.py, ha_files.py)
- One `Lifecycle` manager owns one llama-server (config `argv`, `model_id`, backend port 18081). States: unloaded / loading / ready / failed. On-demand load in `enter()`; idle unload (`idle_seconds`, `auto_unload`).
- Load guards: GPU power limit 175 W, minimum free VRAM, no other CUDA process, model-file stat check, backend port free, residency guard (minimum process bytes) and a VRAM reserve. Session leases block unload.
- `/admin/{load,unload,restart}`; `/admin/status` feeds the Home Assistant sensors. The request `model` must equal `config.model_id` (else 400). `/v1/models` lists one model. Analytics rows record the model; HA has `input_select.engineathome_model`.
- Reusable unchanged: the guards, idle logic, leases, drain/stop. Single-model assumptions: `config["model_id"]`, `argv`, `model_stat`, the `minimum_*` thresholds, request defaults (thinking forced on, xhigh, Swift sampling), HA options and `/v1/models`.

## Minimal design
1. **Config.** Add `models: {key: {model_id, argv, model_stat, context, minimum_load_free_bytes, minimum_process_device_bytes, load_timeout, request_defaults}}` and `default_model`. If `models` is absent, synthesize one entry from today's keys, so the current Swift config keeps working untouched. Both models share the backend port because only one runs at a time.
2. **Lifecycle.** Add `current` (model key). `enter(key)` under the existing transition lock:
   - same key and ready: proceed;
   - different key, no active requests, no leases on the other model, `/slots` idle: unload, then load the new key;
   - otherwise `Busy` (HTTP 409 `model_busy`, with `Retry-After`). No preemption of a running request and no queue in v1.
   - 30 s switch cooldown to avoid thrash. After unload, verify the port is closed and free VRAM has recovered before the next start. Guards are looked up per model.
3. **API.** The request `model` selects the entry (unknown: 400 as today; missing: `default_model`). `request_defaults` are per model: Swift keeps thinking forced on at xhigh and its sampling; Inkling gets none (template default). `/v1/models` lists both with `loaded` and `context`. Clients keep one base URL and token.
4. **Admin and Home Assistant.** `POST /admin/load` accepts `{model: key}` (control scope). `/admin/status` keeps `model.*` for the current/default model, so today's HA sensors keep working, and adds `models: {key: {state, load_seconds, context}}` and `current_model`. The HA generator adds a state sensor per model and sets `input_select.engineathome_model` options to `['All', *model_ids]`.
5. **Analytics.** Record the requested model id (column exists); the `model` filter already works.
6. **Idle unload.** Unchanged; applies to whichever model is loaded. Measured switch cost today: Inkling load 3-8 s with a warm page cache (cold disk not measured), Swift 14-40 s. The prefix cache is lost on every switch (`--cache-ram 0`), so the first request after a switch pays a cold prefill.
7. **GPU tooling.** `gpuexec`/`kgpu` still stop the whole gateway; unchanged.

## Validation before touching production
a. Unit tests in `serving/test_serving.py` using the existing fake `manager_factory`: two fake backends; unknown model 400; switch refused while a request is active; a lease pins its model; cooldown; a crash while loading model B leaves state failed with the retry cooldown; analytics model column; `/v1/models`; backward-compatible single-model config.
b. Staging gateway on a different port and database (`config-staging.json`) under `gpuexec`: three rounds Swift to Inkling to Swift with timings, `nvidia-smi` free VRAM after each unload, one streamed request per model, and one switch attempted during a running stream (expect 409).
c. Home Assistant: generate files for the two-model config into a scratch directory and diff against the current package; load only after review.
d. Rollout: back up config and serving dir (`serving/backups` exists), deploy with `default_model` = Swift, keep the Swift-only config as rollback; confirm `/healthz`, a Swift answer and the HA sensors.

## Open risks
- Inkling needs `-c` up to ~139K with `-ub 256` (a 139K context at `-ub 512` failed with a 14 GB compute buffer), and the VRAM guards need per-model numbers.
- llama-server returns HTTP 500 for truncated tool-call output; pass it through, do not mask it. Agent sessions need a larger context than 32K.
- One backend port means a crash-restart race must be rechecked.
- Estimated size: about 150 lines across `lifecycle.py`, `gateway.py` and `ha_files.py`, plus tests; no change to either model runtime.
