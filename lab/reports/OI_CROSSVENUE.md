# oi_leverage_state — cross-venue / cross-regime re-score (spec J4)

_Binance metrics OI, 2021-01-01 .. 2023-09-28, 8 coins. Same signal, same params. Not a trial._

| | Sharpe(ann) | P&L | n_bar |
|---|---|---|---|
| **full window** | maker **0.12** / taker 0.1 / gross 0.13 | | 24001 |
| LUNA 2022 H1 | 1.17 | 0.1343 | 4344 |
| FTX 2022 H2 | -1.09 | -0.1264 | 4416 |
| recovery 2023 | 0.17 | 0.0342 | 6481 |

- walk-forward (4 windows): [np.float64(0.0), np.float64(-0.12), np.float64(1.05), np.float64(-0.51)]
- turnover/bar: 0.0008   max DD: -0.3255
- placebo p: 0.465   surrogate p: 0.465

## → FAILS on Binance 2021-2023 — oi_leverage_state is Bybit/regime-specific (a finding)

