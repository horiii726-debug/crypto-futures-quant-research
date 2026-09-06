# Crypto Quant Factory — P1–P4 autonomous run (final)

**Result: 13 families tested across 305 cumulative machine-counted trials. ZERO
systems cleared the gate ladder. The test partition was never unsealed
(`test_looks` = 0). Reported in full with bounds — a valid outcome.**

All figures are machine-computed and stored in `ledger/lab.sqlite` (append-only,
UPDATE/DELETE trigger-blocked). 131 verdicts, 131 lessons, 44 bounds.

---

## P1 — MAKER / HYBRID EXECUTION MODEL  → `reports/F1_HYBRID.md`

Built `lab/exec/maker_model.py`: a limit order rests at the touch; **`p_fill`
and adverse selection are simulated on the real aggTrades tape** (queue-ahead =
0.7 × real best-bid-qty from bookTicker); if unfilled after the wait (45 s
entry / 10 s exit) the order **crosses with a taker** and pays fee + half-spread
+ the mid move during the wait. Adverse selection floored at 0.75 bps/side.
**No 100 % maker-fill assumption.**

Calibrated on 7 sample days (2023-10 … 2025-07), 5 liquidity tiers:

| symbol | best-bid-qty | p_fill entry/exit | hybrid round-trip | taker round-trip | saving |
|---|---|---|---|---|---|
| BTCUSDT | $112k | 1.00 / 0.93 | 5.7 bps | 10.0 bps | 44 % |
| SOLUSDT | $1.8k | 1.00 / 0.99 | 7.6 bps | 10.1 bps | 25 % |
| WLDUSDT | $229 | 1.00 / 1.00 | 5.5 bps | 10.8 bps | 49 % |
| HOOKUSDT | $102 | 0.94 / 0.73 | 9.6 bps | 11.9 bps | 20 % |
| DARUSDT | $1.2k | 0.67 / 0.29 | 14.3 bps | 17.2 bps | 17 % |

**Universe hybrid round-trip ≈ 8.5 bps** (vs taker ~11–17). Illiquid names
barely improve — the exit leg mostly still crosses.

### F-1 with the hybrid cost floor

| horizon | taker floor | hybrid floor | disp D | req IC (hybrid, turnover 1.2) | FEASIBLE hybrid |
|---|---|---|---|---|---|
| 1h | 12.3 bps | 9.1 bps | 60 bps | 0.181 | closed |
| 4h | 12.5 | 9.3 | 123 | 0.091 | closed |
| **1d** | 14.0 | 10.8 | 313 | **0.041** | **REOPENED** |
| 3d | 17.5 | 14.3 | 567 | 0.030 | feasible |

**Hybrid execution reopens the 1-day horizon** for low-turnover cross-sectional
strategies (required IC 0.041 < the 0.05 admissible ceiling). 4h and shorter
stay closed — cost still dwarfs the achievable edge. All alpha research below
runs at **1d and 3d** with the hybrid cost.

*(Note: bookTicker/bookDepth history on data.binance.vision ends 2024-04/05,
so the queue term is calibrated on the 2023-11 … 2024-03 book; the aggTrades
tape used for `p_fill` spans the full window.)*

---

## P2 — CORPUS EXPANSION  → `lab/rag/formulas.sqlite`

**76 papers, 131 formulas, 18 divisions** (from 50 / 66 / 4). 63 ADMISSIBLE,
10 PARKED, 3 REJECT_DATA. New divisions seeded: F_VOL (16 formulas), F_STATE
(6), F_DIR (8), F_ENTRY/F_EXIT (5), F_SIZE (4), F_TRANSFORM (5), F_DIST (4),
F_DEP (4), F_PATH (4), F_MULTI (2), F_FORMULA (1), F_LIQ (4), F_BOOK (4).

- **F_OI and F_LIQD → REJECT_DATA.** Binance publishes only ~30 days of open
  interest / long-short / liquidation history (REST hist endpoints are 30-day;
  the bulk `metrics` files stop in 2022). A 2-year backtest of these is
  impossible without a paid third-party archive.
- F_VOL / F_SIZE / F_STATE / F_DIST / F_EXIT / F_FILTER are **overlays**
  (volatility forecasts, sizing, regime gates), not standalone alpha. They can
  only be evaluated on top of a base signal that already passes — none did — so
  they were tested in their tradeable cross-sectional forms (low-vol sort,
  regime-switch momentum, etc.).

## S3–S7 — TOURNAMENT (13 families, 1d + 3d, hybrid cost)

~193 new pre-registered configs; **DSR reads the cumulative machine trial
count, now 305, never reset.** Placebo = random cross-sectional ranking.
Surrogate = block-shuffled forward returns.

| family | configs | mechanisms | best in-sample \|IC\| | best DSR | best net Sharpe (ann) | beat placebo | beat surrogate | verdict |
|---|---|---|---|---|---|---|---|---|
| F_XSEC | 58 | 22 | 0.052 (`idio_vol`) | 0.14 | +1.30 | 9/22 | 6/22 | FAIL / bound |
| F_FUND | 15 | 8 | 0.029 (`carry`) | 0.13 | +1.14 | 1/8 | 0/8 | FAIL / bound |
| F_FLOW | 16 | 8 | 0.032 (`imb_mom`) | 0.15 | +1.25 | 3/8 | 0/8 | FAIL / bound |
| F_XCOIN | 18 | 10 | 0.038 (`peer_ret`) | 0.09 | +1.13 | 5/10 | 1/10 | FAIL / bound |
| F_DIR | 30 | 10 | 0.037 (`rsi`) | 0.09 | +1.08 | 4/10 | 3/10 | FAIL / bound |
| F_TRANSFORM | 10 | 4 | 0.025 (`spec_entropy`) | **0.35** | **+1.89** | 3/4 | 1/4 | FAIL / bound |
| F_DEP | 8 | 4 | 0.013 (`var_ratio`) | 0.09 | +1.14 | 2/4 | 0/4 | FAIL / bound |
| F_PATH | 8 | 4 | 0.024 (`run_length`) | 0.06 | +0.93 | 2/4 | 1/4 | FAIL / bound |
| F_MULTI | 8 | 4 | 0.032 (`tf_agree`) | 0.03 | +0.69 | 2/4 | 1/4 | FAIL / bound |
| F_VOL | 8 | 4 | 0.066 (`yang_zhang`) | 0.00 | −0.08 | 0/4 | 2/4 | FAIL / bound |
| F_LIQ | 6 | 2 | 0.015 (`kyle_bar`) | 0.05 | +0.86 | 1/2 | 0/2 | FAIL / bound |
| F_STATE | 8 | 2 | 0.010 (`regime_mom`) | 0.00 | −0.10 | 0/2 | 0/2 | FAIL / bound |
| F_BOOK | 4 | 1 | 0.028 (`depth_imb_mom`) | — | — | 1/4 | 0/4 | **UNDERPOWERED** |

**No config anywhere reaches DSR 0.95.** The single highest DSR (0.35,
`transform_hurst`@3d, net Sharpe +1.89) fails the surrogate test and the
economic-IC floor. Best in-sample IC that also beats the surrogate: ~0.03–0.05,
and every one of those flips sign or collapses in purged CV.

The directional signals (momentum, RSI, imbalance-momentum, MA-cross,
peer-return, TF-agreement) mostly **do** beat random ranking — there is faint
real structure — but it is far too small to trade through even the reduced
hybrid cost floor at the only feasible horizons (1d, 3d).

---

## P3 — ORDER-BOOK DATA (F_BOOK / F_LIQ)  → `reports/` + ledger bound

bookDepth is the **only** order-book product with history on
data.binance.vision, and it **ends 2024-05**. Downloaded the full daily series
for 30 coins over 2023-09 … 2024-05 (260 trading days) and built depth-imbalance
/ book-slope cross-sectional features.

| feature | IC | placebo p | surrogate p |
|---|---|---|---|
| depth_imb_mom | −0.028 | 0.017 | 0.21 |
| depth_imb_level | −0.020 | 0.063 | 0.39 |
| book_slope_level | −0.019 | 0.113 | 0.94 |

**Verdict: UNDERPOWERED (R5).** 260 days cannot resolve an IC of 0.03 at daily
frequency. Best |IC| 0.028 beats the placebo but not the surrogate noise
ceiling. A real microprice / book-imbalance strategy needs a live L2 feed — out
of scope for a bulk-file run. Bound recorded.

---

## P4 — UNIVERSE EXPANSION (top 100, 3 yr) — NOT PURSUED

Autonomous decision. The F-1 feasibility math is `required_IC = turnover ×
cost_floor / dispersion`. Going to 100 coins / 3 years:
- **cost floor unchanged** (same fee/spread structure; illiquid tail is *worse*),
- **dispersion barely changes** (measured 313 bps at 1d for the 55-coin set; a
  100-coin set including thinner names has similar-to-slightly-lower dispersion),
- so the **required IC is unchanged (~0.04–0.05 at 1d)** and the achievable edge
  (best CV-stable IC ~0.014 across 13 families) does not close that gap,
- and 45 more coins × the feature set would add ~400 trials, pushing the DSR
  deflation *further* against any candidate.

Adding the 2022 bear market (3-year window) makes cross-sectional predictability
net-of-cost worse, not better. Extending the universe would extend the bounds,
not overturn them. Recorded as a reasoned bound rather than a re-run.

---

## RISK / $25,000 SIMULATION

**Not applicable — no candidate reached S7.** The Monte Carlo engine
(`lab/research/montecarlo.py`: block-bootstrap + fixed-fraction +
halve-after-drawdown + explicit crash-correlation shock) is built and
unit-tested, ready for the first candidate that clears the ladder.

---

## BOUNDS — the result of record

Written to `ledger` `bounds` (44 rows). Summary:

1. **Horizons 5m / 15m / 1h / 4h — permanently closed**, taker *and* hybrid.
   Required IC 0.09 (4h) to 0.83 (5m) vs a 0.05 ceiling. Cost floor dominates.
2. **Horizon 1d — feasible only with hybrid execution and turnover ≤ ~1.4**
   (required IC 0.041). No mechanism delivered a CV-stable IC above ~0.015 there.
3. **13 families, 305 cumulative trials, 0 survivors.** Best in-sample |IC| by
   family 0.010–0.066; best CV-stable IC ~0.014; best DSR 0.35; nothing passes
   the joint gate (economic IC ≥ 0.03 AND CV-stable AND clears surrogate AND
   DSR ≥ 0.95 AND net Sharpe ≥ floor).
4. **F_OI, F_LIQD — REJECT_DATA.** No multi-year open-interest / liquidation
   history is freely available.
5. **F_BOOK, F_LIQ (depth) — UNDERPOWERED.** Only 8 months of order-book history
   exists in bulk (ends 2024-05).
6. **F_VOL / F_STATE / F_SIZE / F_DIST / F_EXIT / F_FILTER — no standalone alpha
   to evaluate.** They are overlays; they require a passing base signal, and
   there was none.

## What is left, in priority order

1. **True L2 microstructure** — a live or paid book/tape archive would open
   F_BOOK, F_LIQ, real F_FLOW (VPIN), and a proper queue-aware execution model.
   This is the one genuinely unexplored information set.
2. **Single-asset / lower-frequency directional** (TS momentum, trend on BTC/ETH
   at multi-week horizons) — a different objective than the cross-sectional
   book this run pursued; the 3d horizon is feasible and lightly explored.
3. **Event studies around funding settlements / large liquidations** — needs the
   liquidation stream (P3 data gap).

The cross-sectional, cost-constrained, bulk-data approach is **bounded**: on
2 years of real Binance USDT-M perp data, with measured costs and hybrid
execution, across 13 formula families and 305 trials, there is no tradeable
cross-sectional alpha at a feasible horizon.
