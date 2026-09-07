# ROUND3_GATE — pre-registered design for the next trial budget (spec K)

**Status: NOT opened.** Spec K opens "kalau J berhasil"; J4 showed
`oi_leverage_state` does not generalise, so this is held pending an explicit
decision to open it. Everything below is frozen and will not be changed after
results are seen.

## K1 — pre-screen (reject without a backtest)

A mechanism is dropped before any backtest unless **all** hold:

1. `edge_bps > 3 × cost_bps` at the mechanism's own horizon and universe,
   where `edge_bps ≈ 1e4 · IC · σ_ret · √breadth` (daily)
2. turnover ≤ 0.05 / bar
3. horizon ≥ 4 hours
4. mechanism is **not** directional-timing (18 families already bounded there)

## K2 — budget

- **hard cap 30 trials**, pre-registered here
- separate ledger: `ledger/round3.sqlite`, family tag per mechanism
- 1 economic mechanism = 1 trial at the author's default parameters, **no grid**
- duplicate mechanisms collapse to one entry

## K3 — candidate mechanisms (pre-screened, never tested here)

| # | mechanism | horizon | data (now available) | predicted sign |
|---|---|---|---|---|
| 1 | **perp–quarterly basis / term-structure** — annualised basis of the quarterly future vs the perp; trade the roll | days | Binance quarterly klines + perp | rich basis → short the quarterly / long perp; converges into expiry |
| 2 | **funding momentum (persistence, not level)** — sign & magnitude of the *change* in the trailing funding trend | 8 h – 3 d | multi-venue funding (Binance/Bybit, 2021+) | funding regimes persist for days; fade only at extremes |
| 3 | **cross-venue OI divergence** — one venue's OI rises while another's falls (positioning rotating between venues) | 4 – 24 h | multi_venue OI USD (Binance 2020-09+, Bybit 2023+) | divergence precedes a squeeze on the crowded venue |
| 4 | **liquidation-cascade aftermath (24–72 h)** — after a large forced-liquidation cluster, does price over- or under-shoot over the next 1–3 days | 24 – 72 h | coin-M `liquidationSnapshot` (BTC/ETH) + funding | overshoot → mean-revert over 24–72 h |

**Auto-rejected at intake:** microstructure < 4 h, statistical arbitrage,
cross-sectional directional.

## Gate to pass (frozen)

Same ladder as ROUND 1/2 with the per-coin cost model:
purged CV ≥ 4/6 folds positive · placebo p < 0.05 · surrogate p < 0.05 ·
PBO < 0.5 · net Sharpe (cost_model) ≥ 1.0 held-out · turnover ≤ 0.05/bar ·
walk-forward not monotone declining · **DSR ≥ 0.95 vs the 30-trial budget**
· cross-venue / cross-regime re-check before any "PASS" is final.
