# SenseNova-Loop-9B at 256K

## Phase 2 · Multi-hop reasoning

**Strict standalone JSON score: 0/4. Content score: 1/4 after removing a single enclosing Markdown JSON fence.** Three cases contained substantive reasoning errors. The synthesis case returned every expected field.

Four deterministic requests used SenseNova-Loop-9B Q4_K_M, Q8 K/V, microbatch 1,024, temperature 0 and seed 42. The chat template disabled thinking. The installed CUDA library matched the preserved baseline SHA-256. Each prompt contained four relevant evidence sections, separated by about 24% of the context per case. The cases included signed and superseded records, conflicting scopes, and plausible distractors. Fixtures and expected answers were saved before the first inference. The first request was cold. The other three reused the common archive where the server could restore a checkpoint.

| Case | Input tokens | Evidence depths | Mode | Reused tokens | TTFT (s) | Fresh prefill (tok/s) | Decode (tok/s) | Peak device VRAM (GiB) | Strict / content |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| temporal | 261,622 | 4%, 28%, 52%, 76% | cold | 0 | 231.475 | 1,130.5 | 29.0 | 15.43 | 0 / 0 |
| dependency | 261,616 | 10%, 34%, 58%, 82% | cached | 260,594 | 1.757 | 599.6 | 29.1 | 15.36 | 0 / 0 |
| overrides | 261,632 | 16%, 40%, 64%, 88% | cached | 260,594 | 1.823 | 586.4 | 29.1 | 15.35 | 0 / 0 |
| synthesis | 261,618 | 22%, 46%, 70%, 94% | cached | 260,604 | 1.756 | 595.7 | 29.1 | 15.35 | 0 / 1 |

Actual inputs covered 99.80% of the 262,144-token allocation rounded to two decimals. Cached rows report throughput for fresh tokens only. VRAM is the sampled peak for the whole GPU during each request, including other GPU users. Cold and cached rows use different questions, so their timing difference combines cache state and workload.

## Grading and failure analysis

- **Temporal.** Selected N24, which was signed a day after activation, and copied activation time without adding the 30-hour retention interval. Lane Sable was correct.
  Expected `{"policy":"N21","release_utc":"2031-05-13T15:00Z","lane":"Sable"}`. Observed fields `{"policy":"N24","release_utc":"2031-05-12T09:00Z","lane":"Sable"}`.
- **Dependency.** Selected Relay 3.8 / Bridge 2.4 / Pollen. That chain fits staging ABI 8, while production is ABI 7, which selects Relay 3.5 / Bridge 2.3 / Lumen.
  Expected `{"relay":"3.5","bridge":"2.3","codec":"Lumen"}`. Observed fields `{"relay":"3.8","bridge":"2.4","codec":"Pollen"}`.
- **Overrides.** Kept west's onyx endpoint instead of Kite's higher-precedence sable. The west worker count and emergency timeout were correct.
  Expected `{"workers":8,"timeout_seconds":20,"endpoint":"sable"}`. Observed fields `{"workers":8,"timeout_seconds":20,"endpoint":"onyx"}`.
- **Synthesis.** Returned all four expected fields, including available sets and carrier, inside a Markdown code fence. The requested standalone JSON format failed.
  Expected `{"depot":"Alder","authorized_grade":"A","available_sets":13,"carrier":"Tern"}`. Observed fields `{"depot":"Alder","authorized_grade":"A","available_sets":13,"carrier":"Tern"}`.

One synthetic shared archive and one request per category make this a screening result. The content failures limit confidence in using this setup as a full-context coding agent. The next quality decision should distinguish cross-document reasoning from repository comprehension.

[JSON results and grading](results.json) · [Case fixtures, evidence, and prompt suffixes](fixtures.json) · [Full shared prompt text](shared-prefix.txt) · [Shared token IDs](shared-prefix-tokens.json) · [Runner](phase2.py) · [Engine log](engine.log.gz)
