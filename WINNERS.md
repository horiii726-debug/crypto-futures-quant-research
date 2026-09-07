# WINNERS — formulas that passed, and what happened to them

160 formulas · 8 divisions · triple-barrier event backtest · 5–7 day holds ·
volatility-scaled SL/TP · per-coin cost.

## Passed the tournament (expectancy > 0, t > 2, n ≥ 50)

| # | formula | division | paper | expectancy | t | win rate | trades | Sharpe |
|---|---|---|---|---|---|---|---|---|
| 1 | **V10_bipower_jump** | D2_VOL | Barndorff-Nielsen & Shephard 2004, *J. Financial Econometrics* | **+160 bps** | **3.27** | 54.1 % | 677 | 2.50 |
| 2 | **V02_yang_zhang** | D2_VOL | Yang & Zhang 2000, *J. Business* | **+237 bps** | **2.82** | 57.2 % | 297 | 2.14 |
| 3 | **R14_dispersion** | D8_REGIME | Stivers & Sun 2010, *JFQA* | **+117 bps** | **2.45** | 52.5 % | 835 | 1.89 |
| 4 | **V04_garman_klass** | D2_VOL | Garman & Klass 1980, *J. Business* | **+188 bps** | **2.16** | 57.1 % | 287 | 1.64 |
| 5 | **C17_premium_vol** | D6_FUND | Bakshi, Gao & Rossi 2019, *Management Science* | **+197 bps** | **2.09** | 57.1 % | 198 | 1.59 |

**5 winners out of 160.** Four of the five are the same economic effect measured
different ways: **short high-volatility / high-jump coins, long low-volatility
ones** — the low-volatility anomaly (Ang-Hodrick-Xing-Zhang 2006;
Frazzini-Pedersen 2014).

### What the winners actually say

| formula | the trade |
|---|---|
| `V10_bipower_jump` | short coins whose recent variance is mostly **jumps** (RV − BV), long the smooth ones |
| `V02_yang_zhang` | short high **Yang-Zhang range volatility**, long low |
| `V04_garman_klass` | same, via the **Garman-Klass** estimator |
| `C17_premium_vol` | short coins with volatile **funding premium**, long stable ones |
| `R14_dispersion` | trade **momentum when cross-sectional dispersion is high**, reversal when low |

---

## What happened to them: all five failed validation

**Correction 1 — sample uniqueness** (López de Prado 2018, ch.4).
677 overlapping trades ≠ 677 independent observations. `n_effective = T / h`
(sample length ÷ holding period), verified exactly against the data.

| | raw | adjusted |
|---|---|---|
| V10_bipower_jump | n 677, t **3.27** | n_eff 117, t **1.36** |
| V02_yang_zhang | n 297, t **2.82** | n_eff 97, t **1.61** |

Deflated threshold for 160 formulas (Bailey & López de Prado 2014): **t = 2.69**.
Nothing clears.

**Correction 2 — the 5.49-year out-of-sample test.**
Built a 62-coin, 48 121-bar panel (2020-01 → 2025-06, delisted coins retained),
3.1× the discovery sample. The effect collapsed instead of strengthening:

| formula | 1.75 y (discovery) | **5.49 y (full)** |
|---|---|---|
| V02_yang_zhang | +237 bps | **+28 bps** |
| V10_bipower_jump | +160 bps | **+34 bps** |
| V04_garman_klass | +188 bps | **+14 bps** |

**Why — it is regime-conditional:**

| period | V02_yang_zhang | V10_bipower_jump |
|---|---|---|
| 2020 covid + DeFi | −31 bps | +6 bps |
| **2021 mania** | **−82** | **−95** |
| 2022 LUNA + FTX | **+181** | +36 |
| 2023 recovery | +72 | +94 |
| 2024–25 | −15 | +87 |

The low-volatility / jump-risk premium is a flight-to-quality effect: it pays in
stress and **loses in a mania**, when junk outperforms.

---

## Final measured bound

```
t = SR_per_trade × sqrt(T / h)

T = 5.49 years (all the free data that exists)
h = 7 days      ->  n_effective = 286
deflated threshold (N = 160)   =  t 2.69
required SR per trade          =  0.159
best observed over 5.5 years   =  0.027
SHORTFALL                      =  5.9x

closing it needs 190 years of data, or a 5.9x stronger edge.
```

**0 tradeable survivors.** Full detail: [`lab/reports/BARRIER_SYSTEM.md`](lab/reports/BARRIER_SYSTEM.md).
