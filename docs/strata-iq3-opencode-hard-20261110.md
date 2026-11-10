# OpenCode 64K: hardest saved task, thinking-budget comparison

**Preferred candidate: OpenCode + 1K thinking.** Rank 1 within this three-setup comparison: tied best correctness and fastest observed task completion. Rank 2 is 4K thinking; rank 3 is no explicit thinking cap. Full 18-attempt validation is the next experiment, not a completed result.

**5/6 strict passes and5/6 patch passes.** Frozen Sphinx10323, two fresh attempts per setup; not six distinct issues and not a general capability ranking.

| Requested thinking budget | Strict / patch passes | Attempt times | Native accepted decode |
|---|---|---|---:|
|1024 tokens|2/2 /2/2|427.9s,387.6s|102.08tok/s|
|4096 tokens|2/2 /2/2|432.6s,546.2s|101.96tok/s|
|0 (no explicit cap)|1/2 /1/2|479.9s,748.7s|103.48tok/s|

The second uncapped attempt reached100 API steps and failed one held-out test (20passed/1failed): combined dedent/append/prepend stripped a comment marker. Its final forced text response used stop;100steps is still a strict completion failure by the recorded criterion. No attempt timed out. Increasing thinking did not improve measured correctness on this issue. The1K profile has the best observed latency among tied2/2 profiles; two trials are insufficient to establish a reliable quality ranking or that larger budgets are generally worse.

OpenCode1.18.16, native Windows isolated configuration/data, local Strata API only, documented custom bash tool running the task's offline Docker environment plus native file tools sharing the same workspace. Normal tool/schema errors are retained; the exposed tool-error counter counts OpenCode error states and can omit nonzero shell exits serialized inside completed tool output. Compaction counts are unavailable, not zero. Actual maximum prompts:41941(1K),55346(4K),57324(uncapped),65536capacity; no injected filler.

Fixed settings: IQ3_S/Strata0.1.59/INT8KV/MTP4/min-p0.5/1500MiBreserve/fullPLE/automaticfullarena/default approximate expert-tail skip7. Thinking enabled/high effort,8192 outputtokens,100steps,1200second agent limit,temp0/seed42. Only requested reasoning budget differs within this experiment. Budget0 still has the8192 total output cap and server answer-room behavior; it is not unlimited generation. Accepted decode includes reasoning. No full-expert parity or maximum-intelligence claim.

Device-wide GPU-only energy95.09Wh;113.23Wmean/266.97Wpeak,62.75Cpeak.100% coverage of3023.31-second agent telemetry window; startup excluded and overlapping runtime telemetry not added. CPU/wall energy unavailable. Peak sampledVRAM14.52GiB; nativeRSS48.41GiB (can contain shared/mapped memory); minimum available WindowsphysicalRAM117.68MiB. Original8GiB reserve criterion still fails.

The earlier Hermes Sphinx10323 result was1/3 strict and2/3 patches. This is not an isolated agent comparison: OpenCode also had100steps/1200seconds/8192output instead of40steps/~468agentseconds/4096output, different tools and compaction. Raw outcomes and failures remain separate. Known inference approximation tail7 is unchanged; evaluating full experts is a separate future variable, not a completed optimization.

Official runtime-stop succeeded; owned Strata process gone. No next GPU experiment started. Hourly monitoring paused after completion. Raw evidence: bench/results/strata-iq3-opencode-hard-20261110; configs, exact definitions, grading logs/patches, OpenCode events and native timing/telemetry retained. Personal OpenCode/Hermes/Strata installs/settings unchanged. Raw traces and private configuration remain local; this report contains reviewed findings only.

[Ranked settings and sanitized measurements](results/strata-iq3-opencode-hard-20261110.json) · [Benchmark card](assets/strata-opencode-64k-card.png)
