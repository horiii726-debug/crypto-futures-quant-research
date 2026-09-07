# MECHANISM_REGISTRY — RESEARCH ROUND 2

Pre-registered **before** results are seen. One economic mechanism = one trial at
the author's default parameters. A grid (≤ 8 configs) runs only after the default
clears the initial screen; every grid config counts as a trial.

Intake filter: **horizon ≥ 1 hour.** Microstructure mechanisms < 15 min are
auto-rejected (18 families already bounded there). Duplicate mechanisms collapse
to one entry.

| paper_id | mechanism | data | horizon | predicted_sign | params_default | status |
|---|---|---|---|---|---|---|
| Engle-Granger 1987; Johansen 1991 | `pairs_johansen_ou` — Johansen rank-1 cointegrated pair, static hedge ratio per 90d window, trade OU s-score of the spread | 28 liquid perps, 1h close, train+valid 2023-09..2025-05 | mean-reversion: short spread when s>1.25, long when s<−1.25, exit \|s\|<0.5 | corr_min 0.70, window 2160h, refit 168h, HL∈[4h,120h], entry_s 1.25, exit_s 0.5, stop_s 4.0, target_vol 10%/yr, max_gross_lev 3, max_pairs 20 | **FAIL** (default fails screen; gross Sharpe < 0, surrogate p ≈ 0.5) |
| Elliott-van der Hoek-Malcolm 2005; Chan 2013 | `pairs_kalman_ou` — same pair selection, hedge ratio from a Kalman filter (state = [β, α]) | as above | mean-reversion, dynamic β | Q/R = 1e-5 (sensitivity: 1e-6, 1e-4 only) | **FAIL** (default fails screen; gross Sharpe < 0, surrogate p ≈ 0.5) |
| Avellaneda & Lee 2010 | `pairs_avellaneda` — remove 3 PCs from the return cross-section, OU on the cumulative residual, s-score per coin | as above | residual mean-reversion, entry \|s\|>1.25, exit \|s\|<0.5 | n_pcs 3, pca_win 504h, OU win 1080h, HL∈[4h,120h] | **FAIL** (default fails screen; gross Sharpe < 0, surrogate p ≈ 0.5) |
| sizing overlay (not a mechanism) | GJR-GARCH(1,1) conditional vol → position size only, never direction | pnl series | n/a — sizing | refit 168h, target_vol 10%/yr, lev cap 3 | applied to all F_PAIRS |

## Frozen gate — P3.7 (not moved regardless of outcome)

A mechanism PASSES only if the best of its ≤ 40-budget configs satisfies **all**:

1. net Sharpe (per-coin `cost_model`) **≥ 1.0 on the held-out last 30 %**
2. turnover **≤ 0.05 per bar**
3. purged 6-fold CV: **≥ 4 folds positive**
4. walk-forward Sharpe **not monotonically declining** across 4 quarters
5. placebo p **< 0.05** (block sign-shuffle)
6. surrogate p **< 0.05** (block bootstrap of pnl)
7. PBO **< 0.5**
8. DSR **≥ 0.95** against the F_PAIRS 40-trial budget (`ledger/pairs.sqlite`, not the 745)

Ledger: `ledger/pairs.sqlite`, family `F_PAIRS`, budget **40** trials.

### F_PAIRS outcome (2026-09-07) — BOUNDED, 3/40 trials used

All 3 mechanisms fail the initial screen at their pre-registered defaults, so the
grid was not run. Decisive OOS diagnostic (208,864 pair-bars): correlation
between the entry s-score and the forward 24h spread change is **+0.0045** —
near zero and the *wrong* sign; when s > 1.25 the spread widens (+0.0018), it
does not revert. In-sample Johansen cointegration does not persist one week out
of sample. Best gross Sharpe −0.83, best held-out net Sharpe −1.08. Bound
written to `ledger/pairs.sqlite`.

---

## Candidate family from P2.1 (diagnostic, not yet budgeted)

| paper_id | mechanism | data | horizon | predicted_sign | params_default | status |
|---|---|---|---|---|---|---|
| P2.1 IC-decay measurement | `imbvol_fade` @ ≥1h — DOM imbalance-volume × recent-move sign, cross-sectional fade | 12 coins, bookDepth 5m→1h, 2023-09..2024-05 | rank-IC persists to 1h (measured +0.011, t≈2.6); **decays by 4h** | k=3, hold 1h | **REGISTERED — NOT BUDGETED.** IC at the shortest tradeable horizon (1h) is ~15× below the 1h cost floor (~0.18–0.24). Needs an explicit budget decision; not run in ROUND 2. |
