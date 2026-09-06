# F-1 — crypto feasibility (measured costs)

[SPEC] Universe tag `prio50_2y`, 55 symbols (survivorship: delisted names retained). 1h base panel.

Half-spread used: 0.854 bps (realised, from aggTrades trade data across 22 symbols spanning the liquidity range; p60). Mean |funding|/8h: 1.17 bps. Round-trip turnover assumed 1.6x per rebalance. Cost floor is dominated by 2x taker fee (10 bps).

| horizon | cost floor (bps) | dispersion D (bps) | typ move R (bps) | required IC | required hit-rate | FEASIBLE |
|---|---|---|---|---|---|---|
| 5m | 12.21 | 17.4 | 12.2 | 1.1205 | 0.9997 | NO (closed) |
| 15m | 12.23 | 30.2 | 21.2 | 0.6476 | 0.7888 | NO (closed) |
| 1h | 12.28 | 60.4 | 42.3 | 0.3252 | 0.645 | NO (closed) |
| 4h | 12.5 | 122.6 | 83.0 | 0.1631 | 0.5753 | NO (closed) |
| 1d | 13.96 | 313.4 | 223.0 | 0.0713 | 0.5313 | NO (closed) |
| 3d | 17.47 | 566.8 | 410.1 | 0.0493 | 0.5213 | yes |

**Feasible horizons:** 3d

**Closed (infeasible) horizons:** 5m, 15m, 1h, 4h, 1d

IC ceiling 0.05 (generous upper bound on admissible crypto cross-sectional IC); hit-rate ceiling 0.56.

## Partition hashes (ledger `datasets`)

- `test:1d` sha256 `971cd3b4e3b326f0a9ba07e9…` — 92 bars × 55 symbols, 2025-06-01 00:00:00+00:00 … 2025-08-31 00:00:00+00:00
- `test:1h` sha256 `d554f89a61a8a552384b6447…` — 2208 bars × 55 symbols, 2025-06-01 00:00:00+00:00 … 2025-08-31 23:00:00+00:00
- `test:4h` sha256 `aaf182081dd20d897f93e5c3…` — 552 bars × 55 symbols, 2025-06-01 00:00:00+00:00 … 2025-08-31 20:00:00+00:00
- `train:1d` sha256 `531b77ae766b4d9e7fa4f468…` — 547 bars × 55 symbols, 2023-09-01 00:00:00+00:00 … 2025-02-28 00:00:00+00:00
- `train:1h` sha256 `bd19f304da0c109fa09c6ae1…` — 13128 bars × 55 symbols, 2023-09-01 00:00:00+00:00 … 2025-02-28 23:00:00+00:00
- `train:4h` sha256 `4951e7ded0a40f3e58ad0dfc…` — 3282 bars × 55 symbols, 2023-09-01 00:00:00+00:00 … 2025-02-28 20:00:00+00:00
- `valid:1d` sha256 `7041a23d22a23c5206f9fc40…` — 92 bars × 55 symbols, 2025-03-01 00:00:00+00:00 … 2025-05-31 00:00:00+00:00
- `valid:1h` sha256 `2b300b1436a0ff6ec70788a8…` — 2208 bars × 55 symbols, 2025-03-01 00:00:00+00:00 … 2025-05-31 23:00:00+00:00
- `valid:4h` sha256 `7dd8af9452c1c1ef71263325…` — 552 bars × 55 symbols, 2025-03-01 00:00:00+00:00 … 2025-05-31 20:00:00+00:00