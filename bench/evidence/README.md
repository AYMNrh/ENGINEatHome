# Published benchmark evidence

Explicit allowlist: Huihui 64K (26 completed requests), five DSpine/MTP/no-speculation profiles at 8K (65 completed requests), their summaries, and one cancelled DSpine setup outcome. The cancellation is excluded from successful-run summaries.

Absolute local paths and machine names are replaced with placeholders. Duplicate generated content is retained once per measurement. No raw logs, original source snapshots, generated execution directories, weights or binaries are included. Only synthetic fixtures and generated responses are retained.

`publication.json` records SHA-256 of each original local artifact and each sanitized published artifact. Historical model, binary, source and output hashes remain unchanged; `publication.published_output_sha256` separately identifies the published output string. A redacted artifact is not byte-identical to its original. Missing local log/snapshot references are intentional.

Read the [64K report](../../docs/huihui-64k.md) and [DSpine report](../../docs/dspine-results.md) for settings, correctness failures, memory limits and sample counts. The DSpine measurements cover 8K, with reduced draft limits; they do not establish 64K+ DSpine performance or general sampling equivalence.
