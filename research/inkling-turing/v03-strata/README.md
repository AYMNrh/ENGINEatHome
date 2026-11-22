# Research bundle (2026-11-22): agent evaluation harness, Strata A/B, Strata lessons

- strata/STRATA_LESSONS.md: staged tree verification trace, 256K memory audit and plan, DSpark on/off numerical-correctness analysis (start here).
- strata/STRATA_AB.md: Strata v0.2.19 vs Inkling-Turing e0be55a A/B on a Turing RTX 2080 Ti 22 GB.
- strata/results/: raw JSON/logs. Large binaries (teacher-forced logit dumps, 4 GB) are not included; their sizes and sha256 are in results/LOCAL_ONLY_BINARIES_SHA256.txt. archive_SHA256SUMS belongs to a local tarball of the A/B results.
- agent-eval/: sandboxed coding-agent harness (docker, non-root, no network), reports (REPORT_*.md), DESIGN_MODEL_SWITCH.md, raw results.
Machine-specific details (home directory, host/tailnet names, IPs) were scrubbed; harness scripts read credentials from $CREDENTIALS_FILE, never stored here.
