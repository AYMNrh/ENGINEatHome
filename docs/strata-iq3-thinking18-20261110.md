# Strata IQ3_S thinking-on: 18-attempt validation

**13/18 strict agent passes (72.2%); 16/18 held-out patch passes (88.9%).** All18 attempts graded, six frozen SWE-bench Verified tasks times three. Pilot6/6 is separate and is not added to this score.

Compared with thinking-off18: strict passes improved7->13 and patch passes10->16. Successful tasks per episode-hour improved8.35->11.36. Median episode latency195.8seconds. Thinking remains the best measured Strata agent configuration, but fails the documented15/18 strict promotion gate and the8GiB available-physical-RAM reserve. No128K or automatic next experiment.

Five strict failures: two timeouts with correct patches, one40-turn cap with a correct patch, one40-turn cap with a failing patch, and one normal completion with a failing patch. Thirteen strict passes finished normally with correct patches. All failures retained; no retries. No grading/transport setup failures among the18 scored attempts. Earlier zero-task infrastructure failures remain separate.

| Actual prompt bucket | Accepted decode tok/s | Fresh prefill tok/s |
|---|---:|---:|
| Under8K |92.20|1709.60|
|8-32K|98.20|888.07|
|32-48K|96.99|726.68|
|48-64K (one request)|92.57|157.02|

Capacity65536; maximum observed prompt49964, no forced filler. Decode includes reasoning and startup probes in these buckets.388tool calls/39error results (including failing commands/tests),0confirmed completed compactions; counters missing for two timeout episodes, with compaction attempts observed in logs. This is not a successful compaction/resume stability validation.

Measured device-wide GPU-only energy140.51Wh, mean122.02W/peak261.45W, peak59.64C, average utilization61.55%. Agent telemetry covers100% of its4145.48-second window; campaign coverage and native cache reuse are in theJSON. No CPU/wall energy; overlapping runtime telemetry excluded. Peak sampled VRAM14.52GiB, minimum free1126MiB; native process RSS48.79GiB peak (can include shared/mapped weights). Minimum Windows available physical RAM14.8MiB, below original reserve. Legacy raw comparison field memory_linux_scope contains Windows native measurements for this run.

Unchanged inference settings: IQ3_S, Strata0.1.59, INT8KV, MTP4/min-p0.5, thinkingon/requested reasoning1024, output4096, max40turns,1500MiB VRAMreserve, approximate expert-tail skip7, fullPLE, automatic full expert arena. Windows-native Hermes, pinned revision and frozenmanifest retained. The operational fix loads Strata before Docker, then starts the same Docker engine/images/oracle. Two stale socket-only Docker runtime directories were retained under dated backups and recreated; no images, volumes, settings or credentials were reset. This startup-order change and sequential testing limit isolated causal attribution.

Official runtime-stop returned ok:true/stopped; owned Strata process gone. No next GPU job launched. Best-reference settings are preserved; next work requires reviewing timeout/turn-cap/compaction behavior and memory headroom at64K. Raw evidence: bench/results/strata-iq3-thinking18-live-20261110. Historical comparisons and failures unchanged.

[Sanitized settings and measurements](results/strata-iq3-thinking18-20261110.json)
