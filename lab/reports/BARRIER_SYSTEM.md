# Triple-Barrier Trading System — build + result

Built to the spec: **no scalping** (min hold 1 day, actual mean 5–7 days),
**volatility-scaled SL/TP** from academic estimators (not ATR/RMA), **let the
trade run to a barrier**, **wick-protected stops**, 8 divisions × 20 formulas
from SSRN/arXiv/journal sources.

---

## 1. The architecture change — and it worked

Rounds 1–3 tested every signal as a **continuous cross-sectional rebalance**:
turnover 0.3–2.0 per bar × 5–15 bps cost = the cost drag exceeded the edge at
every horizon (`COST_BOUND`, the dominant root cause in 814 trials).

This round switched to **event-driven triple-barrier** (López de Prado 2018 ch.3):
one entry event → hold → exit at a volatility-scaled TP, SL, or time barrier.

| | continuous rebalance | **triple barrier** |
|---|---|---|
| cost paid | every bar | **once per trade** |
| 7-day σ (crypto) | — | **14 %** |
| TP at 1.0 σ | — | **14 % target move** |
| cost as % of target | **> 100 %** → dead | **≈ 0.1 %** |
| breakeven requirement | rank-IC ≈ 0.04 | **win rate 50.5 %** |
| realised expectancy | net negative everywhere | **+160 … +237 bps / trade** |
| realised win rate | — | **54 – 57 %** |
| mean holding period | 1 bar | **128 h = 5.3 days** |

**The cost problem is solved.** That was the correct diagnosis and the correct fix.

## 2. Mathematics used (no retail indicators)

| component | method | source |
|---|---|---|
| barrier width | Yang-Zhang σ, Rogers-Satchell σ, Parkinson σ (max of the three) | Yang-Zhang 2000 JB · Rogers-Satchell 1991 AAP · Parkinson 1980 JB |
| wick protection | SL floored at the Rogers-Satchell **range** σ, not close-to-close | RS is drift-independent and uses the full H/L excursion |
| optimal TP/SL | first-passage of ABM between absorbing barriers, solved not guessed: `P(+a before −b) = (1−e^{−2μb/σ²})/(1−e^{−2μ(a+b)/σ²})` | Karatzas-Shreve 1991 §3.5 |
| variance forecast | HAR-RV cascade, GJR-GARCH | Corsi 2009 JFEC · Glosten-Jagannathan-Runkle 1993 |
| label overlap | concurrency-weighted sample uniqueness | López de Prado 2018 ch.4 |
| multiple testing | deflated threshold `E[max t]` over N trials | Bailey-López de Prado 2014 |
| cost per fill | per-coin fee + half-spread + √-impact + queue p_fill + adverse selection + walk-the-book | Almgren-Chriss · Kyle 1985 · project calibration |

Verified: the first-passage formula reproduces the analytic gambler's-ruin value
(P = 1.5/3.5 = 0.4286 at μ=0) exactly.

## 3. Tournament — 160 formulas, 8 divisions

| division | formulas | best | t (raw) |
|---|---|---|---|
| D1_TREND | 20 | T08_vortex +34 bps | 0.81 |
| **D2_VOL** | 20 | **V02_yang_zhang +237 bps** | **2.82** |
| | | **V10_bipower_jump +160 bps** | **3.27** |
| D3_DOM | 20 | B07_slope_amplify +124 bps | 1.79 |
| D4_TAPE | 20 | O01_ofi +36 bps | 1.01 |
| D5_OI | 20 | I17_long_liq_bounce +47 bps | 1.14 |
| **D6_FUND** | 20 | **C17_premium_vol +197 bps** | **2.09** |
| D7_VOLUME | 20 | U10_vol_vol +108 bps | 0.94 |
| D8_REGIME | 20 | R01_trend_regime +35 bps | 0.35 |

4 champions cleared `expectancy > 0, t > 2, n ≥ 50` — **and all four were the
same economic effect**: short high-volatility / high-jump / high-premium-vol
coins. That is the low-volatility anomaly (Ang-Hodrick-Xing-Zhang 2006;
Frazzini-Pedersen 2014), one of the most replicated effects in finance,
appearing in crypto.

## 4. Where it broke — two corrections, both fatal

### 4.1 Sample uniqueness

677 raw trades is **not** 677 independent observations. Eight positions open
simultaneously across coins with mean pairwise correlation 0.45 (measured;
effective independent assets = 13.7 of 55, PC1 explains 47.8 %) over 5–7 day
holds collapse to far fewer independent bets.

    n_effective = T / h        (sample length ÷ holding period)

Verified exactly: T = 15 336 bars ÷ h = 128 bars = 120 = the measured n_eff.
**Adding coins does not help** — they are all held over the same period.

| | raw | uniqueness-adjusted |
|---|---|---|
| V10_bipower_jump | n = 677, t = 3.27 | **n_eff = 117, t = 1.36** |
| V02_yang_zhang | n = 297, t = 2.82 | **n_eff = 97, t = 1.61** |

Deflated threshold for N = 160 formulas (Bailey-LdP): **t = 2.69**. Nothing clears.

### 4.2 The 5.5-year out-of-sample test

`t` scales as √T, so the fix was more data. Built a 5.49-year panel (62 coins,
48 121 hourly bars, 2020-01 → 2025-06, delisted coins retained). 3.1× the
discovery sample — enough, in principle, to lift t = 1.61 to ≈ 2.8.

**It went the other way:**

| formula | 1.75 y (discovery) | **5.49 y (full)** |
|---|---|---|
| V02_yang_zhang | +237 bps, t 1.61 | **+28 bps, t 0.36** |
| V10_bipower_jump | +160 bps, t 1.36 | **+34 bps, t 0.52** |
| V04_garman_klass | +188 bps | **+14 bps, t 0.17** |

`n_eff` rose as predicted (117 → 373) but the per-trade Sharpe collapsed
(0.126 → 0.027). **The effect did not exist before 2023.**

### 4.3 Why — it is regime-conditional

| period | V02_yang_zhang | V10_bipower_jump |
|---|---|---|
| 2020 covid + DeFi | −31 bps | +6 bps |
| **2021 mania** | **−82** | **−95** |
| 2022 LUNA + FTX | **+181** | +36 |
| 2023 recovery | +72 | +94 |
| 2024–25 | −15 | +87 |

Economically coherent: the low-volatility / jump-risk premium is a
flight-to-quality effect. It pays in stress and **loses in a mania**, when junk
outperforms. `bipower_jump` is positive in 4 of 5 periods — the single 2021
loss cancels the rest.

## 5. The measured bound

```
t = SR_per_trade × sqrt(T / h)

T = 5.49 years (all the free data that exists)   h = 7 days
n_effective                    = 286
deflated threshold (N = 160)   = t 2.69
=> required SR per trade       = 0.159
   best observed over 5.5 y    = 0.027
   SHORTFALL                   = 5.9x

closing it needs either 35x more data (190 years — does not exist)
or a genuinely 5.9x stronger per-trade edge.
```

## 6. What is now built and reusable

- `lab/engine/barriers.py` — triple-barrier labelling, academic σ estimators,
  first-passage barrier solver, sample-uniqueness weights
- `lab/engine/event_backtest.py` — event-driven backtest with cooldown,
  concurrency caps, min-hold constraint, per-coin cost
- `lab/divisions/` — 160 formulas, 8 divisions, each with its paper reference
  and the author's default parameters
- `lab/research/tournament.py` — division tournament + ranking
- `lab/research/champion_validate.py` — deflated-t, uniqueness, placebo,
  surrogate, held-out
- `lab/data/long_panel.py` — 5.49-year, 62-coin survivorship-correct panel

If a genuine signal ever arrives (better data — live L2, longer multi-venue
history, options surface), the machine to trade it correctly is finished.
