# THEORY: Inkling 2 Mini decode at long context (2080 Ti, 175 W)
Builds on `research/inkling-turing/THEORY.md` (measured 546.8 GB/s, 3.22 GB weights/token, KV q8_0 2176 B/token/visit, 8 full-attention visits/token).
Squid5090's RTX 5090 numbers are not used as ceilings.

| depth | bytes/token (weights + 8 x ctx x 2176) | ceiling @547 GB/s | measured (D1-BASE) | % |
|---|---|---|---|---|
| 16K | ~3.5 GB (+0.28) | ~150 | 55.9 | 37% |
| 64K | 3.22+1.14 = 4.36 GB | ~125 | 45.8 | 37% |
| 128K | 3.22+2.28 = 5.50 GB | ~99 | 36.8 | 37% |
| 242K | 3.22+4.29 = 7.5 GB | ~73 | 26.8 (historical) | 37% |

The achieved fraction is flat at ~37%, i.e. the whole decode is latency-bound, not only attention.
Attention-specific (D2): 1.137 ms/launch for ~285 MB at 128K = 251 GB/s (46%); marginal KV cost 64K->128K = 5.4 ms/token for 1.14 GB = 211 GB/s.
Even at 90% of ceiling attention would be 0.58 ms/launch: potential -4 ms/token at 128K (up to ~+15%). Not reached: the kernel steps through
128-position KV tiles serially (shuffle reduction per row, two syncs per tile, 244 regs, 4 warps/block, 1 block/SM by launch bounds).
