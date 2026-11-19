# Homelab production serving

Deployed 2026-11-19. This replaces the retired benchmark scheduler with an on-demand API; it does not replace or optimize the stock inference engine.

| Item | Value |
|---|---|
| OpenAI base URL | `http://100.75.206.4:8080/v1` |
| MagicDNS alternative | `http://homelab.tail5bda2d.ts.net:8080/v1` |
| Model ID | `swift-1.5-qwen3.8-27b` |
| Analytics and controls | `http://homelab.tail5bda2d.ts.net:8080/analytics` |
| Home Assistant dashboard | `http://homelab.tail5bda2d.ts.net:8123/engine-at-home/serving` |
| Gateway service | `engineathome-api.service` |
| Deployment | `/home/marvin/llm/serving` |
| Isolated Python | `/home/marvin/llm/serving-venv` — Python 3.14, aiohttp 3.14.5 |
| Usage database | `/var/lib/engineathome/usage.sqlite3` |

Tailscale must be connected. Prefer the numeric IP on clients without MagicDNS; a hostname lookup failure is distinct from an authentication error. Port 8080 binds only Tailscale IPv4 and localhost; the native server binds only `127.0.0.1:8081`. No router forwarding or public listener was added. HTTP is carried inside Tailscale's encrypted tunnel; this is not a public HTTPS service. Existing HA network exposure was not broadened.

## Runtime preserved

- Original `/home/marvin/llm/llama.cpp/build/bin/llama-server`, build 12519, revision `98d67c8d536dc584897c4adb2510509796517222`.
- Model `/home/marvin/llm/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`.
- SHA256 `9aecf1cd41b2cb2f32a74e0d889e33855ebef43b26f43b43feb5720239e677e5`.
- 262,144 context; Q4_0 target/draft KV; MTP3; Flash Attention; 99 GPU layers; `--fit off`; batch 512 / microbatch 128; one native sequence; 175 W.
- Thinking on, xhigh, `--reasoning-budget -1`, DeepSeek reasoning format. The gateway defaults `enable_thinking=true`, `preserve_thinking=true`, and xhigh template effort. No output-token or reasoning-token cap is injected. Clients select their total output limit, and input plus output must fit context.
- Sampling defaults preserved: temperature 1, top-p .95, top-k 20, min-p 0, presence penalty 0, repeat penalty 1, seed 42. Explicit request settings override these defaults.
- `runtime-manifest.json` preserves original and serving launch arguments and checksums. Changes are executable location pointing to the identical original stock build, loopback port, stable model alias, and disabled native text logs. Model arithmetic and inference settings are unchanged.

Preflight refuses another CUDA workload, insufficient VRAM, a changed model file, the wrong power limit, or an independently occupied backend port. Post-load health and per-process device allocation are checked against the validated stock footprint. No CPU fallback is selected. The short serving check observed 18.97 GiB device usage / 2.51 GiB free, matching the prior allocation. This is not a new filled-256K benchmark.

## Credentials and client configuration

Keys are in mode-0600 `/home/marvin/llm/serving/credentials.json`. A Windows copy with restricted file ACLs is at:

`C:\Users\rhihi\Documents\ChatGPT\MARVIN\.homelab-access\engineathome-serving-credentials.json`

| JSON field | Permission / fixed usage label |
|---|---|
| `opencode`, `hermes`, `python` | Inference, model discovery, own session leases |
| `admin` | Lifecycle controls, all leases, usage analytics; no inference |
| `home-assistant` | Controls and analytics; held only in HA's private secrets file |
| `analytics` | Read-only usage analytics |

Do not paste keys into Git, dashboard sensor states, URLs or screenshots. Usage labels come from authenticated credentials, not an untrusted client header. The gateway stores key hashes in its private configuration. Rotate a compromised key in both private files and restart the gateway after draining requests; update the HA secret too if rotating its key.

OpenCode: merge this provider into the existing configuration; do not replace unrelated providers. Set `ENGINEATHOME_API_KEY` to the **opencode** key in the launching environment.

```json
{
  "provider": {
    "engineathome": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Homelab Swift",
      "options": {
        "baseURL": "http://100.75.206.4:8080/v1",
        "apiKey": "{env:ENGINEATHOME_API_KEY}",
        "timeout": 14400000
      },
      "models": {
        "swift-1.5-qwen3.8-27b": {
          "name": "Stock Swift 256K",
          "reasoning": true,
          "tool_call": true,
          "limit": {"context": 262144, "output": 32768}
        }
      }
    }
  },
  "model": "engineathome/swift-1.5-qwen3.8-27b"
}
```

The example's 32K output allowance is a client total-output setting, not a reasoning budget. Adjust it to the task. Provider structure follows this repository's existing OpenCode runner; launching a real OpenCode/Hermes coding session was not part of the serving smoke test.

Hermes: choose its custom OpenAI-compatible provider, enter the base URL and model ID above, and use the **hermes** key. Enable reasoning and tool support where exposed. Do not reuse the previous 1,024-token benchmark reasoning budget. Actual Hermes configuration syntax depends on the installed version; personal configuration was not modified.

Python, using only the standard library on Windows:

```python
import json
from pathlib import Path
from urllib.request import Request, urlopen

keys = json.loads(Path(r"C:\Users\rhihi\Documents\ChatGPT\MARVIN\.homelab-access\engineathome-serving-credentials.json").read_text())
body = {
    "model": "swift-1.5-qwen3.8-27b",
    "messages": [{"role": "user", "content": "Explain this function."}],
    "max_tokens": 4096
}
request = Request(
    "http://homelab.tail5bda2d.ts.net:8080/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Authorization": "Bearer " + keys["python"], "Content-Type": "application/json"}
)
with urlopen(request, timeout=14400) as response:
    result = json.load(response)
print(result["choices"][0]["message"])
```

Supported inference routes: `/v1/chat/completions`, `/v1/completions`, native `/completion`; discovery `/v1/models`. Streaming SSE, `reasoning_content`, tool-call deltas and usage are forwarded. Responses/embeddings APIs are not advertised. Model discovery does not allocate the GPU; the first inference request autoloads it.

## Lifecycle, leases and operation

The default is **on-demand loading and unload after 60 idle minutes**. Active and queued inference requests protect the model. Leases also prevent administrative/automatic unload and restart; they survive gateway restart until expiry. All lifecycle and lease mutations are serialized. Unload/restart returns HTTP 409 when protected; it does not terminate the workflow. Native slot activity is checked as well, including after a disconnected client.

At most eight requests are admitted; the native server has one sequence. Additional admissions return 409. Each request has a configurable four-hour wall-clock timeout. A failed load has a 15-second retry cooldown. Lifecycle failures use stable error codes without recording payloads. Closing the gateway stops new admissions, drains active inference, then stops only its owned child; a system-level forced kill cannot preserve an active request.

PowerShell controls:

```powershell
$keys = Get-Content 'C:\Users\rhihi\Documents\ChatGPT\MARVIN\.homelab-access\engineathome-serving-credentials.json' -Raw | ConvertFrom-Json
$base = 'http://homelab.tail5bda2d.ts.net:8080'
$admin = @{Authorization = 'Bearer ' + $keys.admin}
Invoke-RestMethod "$base/admin/status" -Headers $admin
Invoke-RestMethod "$base/admin/load" -Headers $admin -Method Post -ContentType application/json -Body '{}'
Invoke-RestMethod "$base/admin/unload" -Headers $admin -Method Post -ContentType application/json -Body '{}'
Invoke-RestMethod "$base/admin/restart" -Headers $admin -Method Post -ContentType application/json -Body '{}'
Invoke-RestMethod "$base/admin/settings" -Headers $admin -Method Patch -ContentType application/json -Body '{"auto_unload":true,"idle_seconds":3600}'
```

Long-running agent session lease, using its inference credential:

```powershell
$infer = @{Authorization = 'Bearer ' + $keys.opencode}
$lease = Invoke-RestMethod "$base/v1/leases" -Headers $infer -Method Post -ContentType application/json -Body '{"ttl_seconds":3600}'
# Renew before expiry, e.g. every 20 minutes during the workflow.
Invoke-RestMethod "$base/v1/leases/$($lease.lease_id)" -Headers $infer -Method Put -ContentType application/json -Body '{"ttl_seconds":3600}'
Invoke-RestMethod "$base/v1/leases/$($lease.lease_id)" -Headers $infer -Method Delete
```

On homelab:

```bash
sudo systemctl status engineathome-api.service
sudo journalctl -u engineathome-api.service -f
sudo systemctl restart engineathome-api.service  # drains active requests
sudo systemctl stop engineathome-api.service     # stops gateway too
sudo systemctl start engineathome-api.service
sudo systemctl disable --now engineathome-api.service  # stop boot activation
```

`/healthz` is unauthenticated gateway liveness only; authenticated `/admin/status` reports model health and telemetry. The gateway remains healthy while the model is unloaded. systemd is enabled at boot and restarts the gateway on unexpected failure; after gateway restart the model is on demand. A native child crash is detected, recorded as failed, and the next inference request reloads it. Tailscale binding retries if the interface is late during boot.

## Home Assistant and analytics

Installed package: `/srv/containers/homeassistant/config/engineathome/package.yaml`; dashboard: `engineathome/dashboard.yaml`; its own private `engineathome/secrets.yaml` contains the HA bearer credential. Existing global secrets were untouched. Configuration backups and protected validation output are in `engineathome/backup-20261119T192127Z/`.

The sidebar **ENGINEatHome** dashboard has Load, Unload, Restart, Auto-unload, GPU/lifecycle/inference sensors, daily token counters, history and an analytics link. Native REST polling is every 15 seconds for status and 60 seconds for usage; GPU samples are five seconds old at most during normal operation, with a sample timestamp exposed. Refresh/load actions are asynchronous from a user's perspective; busy protection is enforced by the API. API URL and model ID are displayed as selectable code fields without secrets.

The analytics page accepts the **analytics** key for read-only history or **admin** for history plus controls. The key stays in tab memory only. Filter by time range, hour/day/week/month, client, model and token view (all/input/output/cached). Copy buttons fall back to selecting text on browsers that restrict clipboard access over HTTP.

Authenticated endpoints:

- `/analytics/summary?start=2026-11-19T00:00:00Z&end=2026-11-20T00:00:00Z&bucket=hour&client=python&model=swift-1.5-qwen3.8-27b`
- `/analytics/requests` with the same filters, newest 500 records.
- `/analytics/totals` for current hour/day/week/month totals.

SQLite records request UUID, start/finish timestamps, authenticated client/model, final status/error, reported token counts, native rates, streaming TTFT and latency. No prompts, outputs, reasoning text or API keys are stored. Native request/text logs and aiohttp access logs are disabled; systemd retains lifecycle and sanitized error events.

Accounting: input includes cached input; fresh input = input minus cached only when both are reported; total = input + output; reasoning is an output subset. Repeated cumulative stream usage overwrites counters rather than adding them. Missing counts stay NULL in per-request and grouped results; period totals sum only known values and include known-count coverage fields. Thus zero reported cached tokens is not proof that all requests had zero cache reuse. Incomplete/disconnected streams may lack final usage and are not labeled complete. Existing running records become interrupted after gateway crash.

Native `prefill_tps` measures fresh processing, and native decode is distinct from client-observed latency. Streaming TTFT includes autoload/queue time; `upstream_ttft_s` excludes gateway load but includes backend queueing. Nonstream first-token time is unobservable and remains NULL. Native reasoning-token counts are currently unreported, even though reasoning text forwarding works. Buckets use request start time in Europe/Paris by default, or UTC; no historical inference content is reconstructed.

## Reproduction, validation and rollback

Source is under `serving/`. Deployment reuses `src/qah/telemetry.py` as `serving/qah/telemetry.py` plus an empty `qah/__init__.py`; no existing inference Python environment is modified. Install serving dependencies only into `/home/marvin/llm/serving-venv` using `serving/requirements.txt`. `configure.py` checks the model SHA and original executable/libraries against the validated stock manifest before generating private configuration. The systemd unit is in `serving/engineathome-api.service`; HA generation/backup/validation is in `ha_files.py` and `install_home_assistant.py`.

Completed validation:

- Three async regression tests passed: concurrent autoload, permissions, leases and idle protection; streaming metadata/privacy/accounting; broken stream and interrupted-record recovery.
- Live short requests passed autoload, cached-prefix counters, streamed reasoning/usage, active unload rejection, client disconnect classification, forced tool call plus continuation, lease renewal/release, restart, settings and filtered analytics.
- Deliberately killed the owned native process and gateway separately: autoload recovery, systemd restart, child cleanup and durable usage totals passed.
- Windows reached both numeric and MagicDNS endpoints. HA config validation passed, the container restarted successfully, all 25 entities registered and authenticated polling was observed. The Lovelace buttons were not manually clicked; their underlying operations were tested through the API. Analytics HTML was served and its JavaScript syntax checked, but browser interactions were not automated.
- No long-context benchmark, kernel modification or model-quality sweep ran. An actual machine reboot was not performed; boot enablement and crash restart were verified.

Re-run lightweight tests explicitly (live tests intentionally start/restart/unload the owned model):

```bash
cd ~/llm/serving
~/llm/serving-venv/bin/python -m unittest test_serving -v
~/llm/serving-venv/bin/python validate_live.py
# Only during a maintenance window: intentionally kills the owned backend/gateway.
~/llm/serving-venv/bin/python validate_live.py --recovery
```

Sanitized evidence: `validation-live.json`, `validation-recovery.json`, `runtime-manifest.json`; logs: `~/llm/serving-tests.log`, `serving-validation.log`, `serving-recovery.log`. The live validation requests are labeled `python` in initial usage totals.

The old `swift-bench.service`, `swift-bench-progress.timer` and Windows `ENGINEatHome-Swift-Hourly` task are disabled; their files, queue, builds, models and results remain. Lingering remains enabled. No Git publication was performed.

Rollback: stop/disable only `engineathome-api.service`, restore HA's `configuration.yaml` from the backup, validate HA config, then restart HA. The original stock executable/model and old configurations are untouched. Do not restart benchmark scheduling merely to restore serving. Preserve `/var/lib/engineathome` and protected credentials if retaining usage history. Do not start a standalone server on 8080 while this gateway owns it.

Sources used for the integration: [aiohttp server lifecycle](https://docs.aiohttp.org/en/stable/web_advanced.html), [HA REST sensors](https://www.home-assistant.io/integrations/rest/), [REST commands](https://www.home-assistant.io/integrations/rest_command/), [template entities](https://www.home-assistant.io/integrations/template/), [HA secrets](https://www.home-assistant.io/docs/configuration/secrets/), and the installed HA 2026.10.2 schemas.
