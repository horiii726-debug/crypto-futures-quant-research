# CHANGELOG_METHOD — methodology changes (append-only, newest last)

Every entry is written **before** the change is executed. Format:
`YYYY-MM-DD | area | what changed | why | what it CANNOT do`

---

## 2026-09-07 | cost model | flat pooled cost → per-coin dynamic `cost_bps()` | RESEARCH ROUND 2 · P0

**What changed.** The 305 cross-sectional trials and the F_OI / F_BOOK / F_LIQD /
F_MICRO studies priced turnover with a single pooled scalar:

    MAKER_1W  = exec_summary.hybrid_roundtrip_bps_mean / 2 · 1e-4   ≈ 4.27 bps one-way
    TAKER_1W  = 0.0005 + 0.85e-4                                    = 5.85 bps one-way

That is a **measurement error**: real cost is per-coin (BTC ≈ 5.7 bps round-trip,
DAR ≈ 14.3 bps) and depends on order size (square-root impact) and fill mode.
Replaced with `lab/exec/cost_model.py :: cost_bps(coin, side, notional, ts, mode)`
returning `{fee, spread, impact, slippage, adverse_sel, p_fill, total_bps}`, all
per-trade, calibrated from `exec_calibration.json`, `spread_measured.json` +
`spread_extra.json` + `aggtrades_micro.json` (half-spread), `bookdepth_features.json`
(near-touch depth), and 1h klines (σ_daily, ADV).

**Why now.** Two "positive" leads (F_XVENUE, F_OI·oi_leverage_state) and the whole
"NO_SURVIVOR" bound rest on the cost number. If the pooled scalar is wrong the
bound is wrong. This corrects the measurement; results are taken as they come.

**What this change CANNOT do.**
- It does **not** add trials. Trial count and every family-scoped DSR denominator
  are unchanged. Re-scoring an existing trial with a corrected cost is a
  measurement fix (R3 is about *count*, not *value*).
- It does **not** touch signals, parameters, universe, or dates.
- It does **not** get to be "tuned" after results are seen. The calibration
  inputs are frozen by `REPRICING_RULE.md` before the re-score runs.
- A coin with **no** calibration data falls to an explicit liquidity-tier
  fallback (documented in the rule); it is never silently given BTC's cost.

## 2026-09-07 | new family | F_PAIRS (cointegration / statistical arbitrage) | RESEARCH ROUND 2 · P3

**What changed.** New family with its own ledger namespace and a **pre-registered
40-trial budget**. Mechanism class is *relative-value* (spread mean-reversion
between cointegrated coins), economically distinct from every prior family, all
of which were directional. Methods: Johansen trace test → VECM rank-1 pair
selection (rolling 90d, weekly re-estimate); OU fit on the spread with a hard
half-life filter of 4h–5d; Kalman dynamic hedge ratio (Q/R fixed at 1e-5, ≤3
sensitivity values); Avellaneda-Lee PCA-residual s-score variant (params taken
from the paper, not tuned); GJR-GARCH(1,1) **for position sizing only, never
direction**.

**Why.** The RESEARCH ROUND 1 bound is specifically about *directional*
predictability under cost. Relative-value has a different failure mode and was
never tested.

**What this change CANNOT do.**
- No parameter is grid-searched before a mechanism clears the initial screen;
  grid ≤ 8 configs, every config counts as a trial.
- Microstructure horizons < 15 min are auto-rejected at registry intake.
- The gate (`P3.7`) is frozen here and is not moved regardless of outcome.
- GARCH is a sizing input; it may not become a directional signal.

## 2026-09-07 | diagnostic | imbvol_fade IC-decay curve + execution-timing overlay | RESEARCH ROUND 2 · P2

**What changed.** Measure `imbvol_fade` rank-IC vs forward return at
{5m,15m,30m,1h,4h,1d} (pure measurement, **not** a trial). Separately, allow
`imbvol_fade` to choose the *entry bar within an already-fixed rebalance window*
for `oi_leverage_state` — no new direction, no new sizing, window cannot stretch.

**What this change CANNOT do.**
- The IC-decay curve does not enter the trial ledger. Only if IC survives ≥1h is
  a new family registered (with its own budget).
- The execution overlay introduces no directional hypothesis, so it is not a
  trial. Its only metric is Δ realised cost bps vs the arbitrary-bar baseline.
- The overlay may not change position sign or size, and forced execution at
  window end is mandatory (no lookahead).
