# Research Round 2 — report

Executed P0 → P3 in order. Global rules held: `data/test` never opened, no
failure deleted, no gate loosened. Every methodology change was written to
`lab/reports/CHANGELOG_METHOD.md` before it ran.

---

## P0 — cost-model correction + re-score (measurement fix, no new trials)

### P0.1–P0.4  — the model

`lab/exec/cost_model.py :: cost_bps(coin, side, notional, ts, mode)` returns a
per-trade breakdown `{fee, spread, impact, slippage, adverse_sel, p_fill,
total_bps}`. Frozen calibration (see `REPRICING_RULE.md`):

- **fee** — Binance USDT-M VIP-0: taker 4.5 bps, maker 1.8 bps
- **half-spread** — per coin, measured (25 coins) → liquidity-tier fallback
- **impact** — square-root law `1e4·Y·σ_daily·√(Q/ADV)`, one global `Y=0.5`,
  σ_daily and ADV from 1m klines for all 55 coins
- **p_fill** — queue-depletion model (near-touch depth + arrival rate),
  shrunk by realised vol
- **adverse selection** — per coin (5 calibrated) → `max(0.75, 0.35·half_spread)`
- **residual slippage** — walk-the-book beyond L1

**Unit tests (`tests/test_cost_model.py`) — 8/8 pass**, including the mandatory
reconciliation: BTC hybrid round-trip **5.703 bps vs the old calibration's 5.662
(0.7 % error, band ±15 %)**.

What the old flat number got wrong:

| | old flat (ROUND 1) | new per-coin (median liquid) | new per-coin (thin) |
|---|---|---|---|
| hybrid one-way | 4.27 bps | ~4.0 bps (BTC 2.8) | 8–20 bps |
| **taker one-way** | **5.85 bps** | **~10 bps** | **20–60 bps** |

The old taker cost (fee + 0.85 bps slippage) **omitted the half-spread and
impact entirely** — it understated real taker cost by ~2×.

### P0.5  — re-score of the cross-sectional campaign

`lab/research/rescore.py` re-runs every pre-registered config through
`run_study` twice (v1 flat, v2 per-coin) with **no trial logged** (a
`RescoreLedger` proxies the real trial counts and swallows all writes). Signals,
params, universe, dates unchanged.

**Result (193 configs, 12 families, 1d + 3d):**

| | |
|---|---|
| gate-status changes | **0** |
| net Sharpe worse under v2 | ~91 % |
| net Sharpe better under v2 | 0 |
| mean Δ net Sharpe (v2 − v1) | **−0.13** |

The 55-coin cross-sectional universe is thin-coin-heavy, so per-coin cost is
*higher* on average and every already-failing verdict fails by a little more.
**The NO_SURVIVOR bound is robust to the cost correction — strengthened, not
weakened.** Full table: `lab/reports/RESCORE_305.md`.

### P0.5  — the one candidate, re-scored

`oi_leverage_state` (liquid-28) re-scored via `RescoreLedger`:

| | v1 flat | **v2 per-coin** |
|---|---|---|
| held-out maker Sharpe | +1.25 | **+1.22** |
| held-out taker Sharpe | +1.19 | **+0.90** |
| full-sample maker / taker | +2.27 / +2.21 | +2.26 / +1.96 |
| turnover / bar | 0.0077 | 0.0077 |
| walk-forward | [3.07,2.30,2.05,1.54] | [3.06,2.30,2.02,1.51] |
| placebo p / surrogate p | 0.003 / 0.020 | 0.003 / 0.020 |
| \|IC\| / DSR(F_OI) | 0.004 / 0.72 | 0.004 / 0.68 |

**The candidate survives the corrected cost.** Turnover is so low that the maker
Sharpe barely moves; the higher taker cost pulls held-out taker Sharpe from
+1.19 to +0.90 — still positive. It still fails the same two gates (|IC| < 0.010,
DSR < 0.95). Verdict unchanged: **sub-threshold near-miss, paper-test candidate.**

---

## P1 — paper-test harness (built, not armed)

`lab/live/paper_trader.py` computes the `oi_leverage_state` hourly book, prices
each order with `cost_model`, and logs `intended_price, actual_fill,
predicted_cost_bps, realized_cost_bps, p_fill_predicted, filled, latency_ms`.

Dry-run works today (28-name market-neutral book, gross 1.0). Arming
(`--arm`, real keys) is **gated behind R8** — the sign-off checklist is
`lab/reports/PAPER_TEST_PLAN.md`. Two goals: a 3rd out-of-sample regime (Bybit
OI is only 2 y) and live validation of the cost model. Review every 2 weeks.

---

## P2 — imbvol_fade diagnostics (measurement, not trials)

### P2.1  IC-decay curve

| horizon | rank-IC | t (overlap-adj) |
|---|---|---|
| 5 m | +0.0091 | 7.2 |
| 15 m | +0.0157 | 7.2 |
| **30 m** | **+0.0185** | **6.0** |
| 1 h | +0.0114 | 2.6 |
| 4 h | +0.0096 | 1.1 |
| 1 d | +0.0044 | 0.2 |

The signal **peaks at 30 min** and **persists to 1 h** (t ≈ 2.6), gone by 4 h.
Per the pre-registered P2.1 rule it is **registered as a candidate family
(`F_BOOK_SLOW`) in `MECHANISM_REGISTRY.md` — but NOT budgeted**: at 1 h its IC
(0.011) is ~15× below the 1 h cost floor (~0.18–0.24). It is the same F-1
feasibility bound, now measured precisely.

### P2.2  execution-timing overlay

Using `imbvol_fade` only to pick the entry bar inside an already-fixed rebalance
window (no direction, no size change): execution edge **+0.29 ± 0.15 bps** vs the
arbitrary-bar baseline — **not material** (< 2σ). Not folded in.

---

## P3 — F_PAIRS : cointegration / statistical arbitrage (new family)

Pre-registered: 40-trial budget, own ledger (`ledger/pairs.sqlite`), frozen gate
P3.7. Three mechanisms — Johansen rank-1 + OU (static β), Kalman dynamic β + OU,
Avellaneda-Lee PCA-residual s-score. GJR-GARCH(1,1) for sizing only.

**Outcome — BOUNDED, 3 / 40 trials used.**

| mechanism | gross Sharpe | held-out net | turnover/bar | surrogate p |
|---|---|---|---|---|
| pairs_johansen_ou | −1.80 | −1.42 | 0.34 | 0.56 |
| pairs_kalman_ou | −23.5 | −36.5 | 2.60 | 0.46 |
| pairs_avellaneda | −0.83 | −1.08 | 0.12 | 0.52 |

All three fail the initial screen (gross Sharpe > 0.5), so the grid was not run.

**Decisive OOS diagnostic** (208,864 pair-bars): the correlation between the
entry s-score and the forward 24 h spread change is **+0.0045** — near zero and
the *wrong* sign. When s > 1.25 (spread stretched high) the spread's mean
forward move is **+0.0018 — it widens, it does not revert.**

**In-sample Johansen cointegration does not persist one week out of sample.**
This is not a turnover artifact or a Kalman-tuning problem: the raw spread-
reversion bet has negative gross Sharpe and is indistinguishable from block-
bootstrapped noise. Relative-value spread reversion between liquid crypto perps
is empirically null for 2023-09 → 2025-05.

---

## Where Round 2 leaves the project

- **21 families now bounded** (20 from Round 1 + F_PAIRS). The cost correction
  did not rescue any of them; it strengthened the bound.
- **One open decision unchanged:** `oi_leverage_state` — held-out maker +1.22 /
  taker +0.90 net of *corrected* per-coin cost, survives every robustness test
  except |IC| ≥ 0.010 and DSR ≥ 0.95. The paper-test harness is built and
  waiting on R8 sign-off.
- New reusable infrastructure: per-trade `cost_model.py`, `pairs_lib.py`
  (Johansen / OU / Kalman / PCA-residual / GJR-GARCH), `paper_trader.py`.
