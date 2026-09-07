# Crypto Quant Factory — autonomous run, terminal report

**STOP CONDITION B: every accessible objective is bounded. NO_SURVIVOR.**
**Test partition never opened (`test_looks = 0`) across 401 machine-counted trials.**

The engine (S0–S7) is verified and works — it produced everything below. It is
not "live-ready" because no strategy survived the ladder and because S8–S10
(paper trading, execution testing, sign-off) require a live account and
calendar time by design (R8). The engine is proven and extensible.

All figures are machine-computed and stored in `ledger/lab.sqlite`
(append-only). 157 hypotheses, 136 verdicts, 132 lessons, **52 bounds**.

---

## 1. What was fixed first (SYSTEM_SPEC [FIX 1])

The original spec's "thousands of formulas + one global cumulative DSR" is
mathematically self-defeating. Implemented **two-tier counting**:
- **cheap screen** (rough IC, turnover, cost feasibility) — 432 evaluations,
  `counts_as_trial = 0`, never a DSR input;
- **full trial** (purged CV + placebo + surrogate + DSR + cost stress) — 401,
  DSR **family-scoped** at G5, a portfolio-DSR reserved for G10.

This is why new families (F_CARRY, F_MICRO_*, F_XVENUE) got a fair test instead
of being buried under the earlier 305 cross-sectional trials.

---

## 2. Families tested and BOUNDED

### 2.1 Cross-sectional OHLCV + funding (13 families, 305 trials) — prior runs
F_XSEC, F_XCOIN, F_FUND, F_FLOW, F_DIR, F_TRANSFORM, F_DEP, F_PATH, F_MULTI,
F_VOL, F_LIQ, F_STATE, F_BOOK(daily). Best CV-stable IC ≈ 0.014 vs ≈ 0.04
required at the feasible horizons. Bounded. (See `FINAL_REPORT.md`.)

### 2.2 F_CARRY — delta-neutral funding carry
55 coins, 2 years, real premium-index basis, hysteresis to cut turnover,
2-leg cost + hedge tracking-error charge, capital-efficiency 1.25×.

| metric | best config |
|---|---|
| gross funding harvest | **+10.2 %/yr** (real) |
| net Sharpe (maker legs) | +2.5 |
| net Sharpe (taker legs) | +0.6 |
| **surrogate p-value** | **1.00** |
| walk-forward | +4.7 → +3.6 → +0.5 → +1.0 (decaying) |
| max drawdown | −1.9 % |

**Verdict: BOUNDED.** The ~10 %/yr mean funding is a genuine **short-perp risk
premium**, not alpha — `surrogate p = 1.00` means a *static* short-perp basket
does exactly as well; coin selection and timing add nothing. After a
conservative 2-leg cost + hedge tracking error the net Sharpe is ~+2.5 (maker) /
~+0.6 (taker) and **decaying** over the sample (consistent with the published
collapse of crypto carry in 2024–25). A human may harvest it as a discretionary
beta allocation with active hedging; it is not a system.

### 2.3 F_MICRO_OFI / F_MICRO_BOOK — tape microstructure at M3 / M5 / M15
Built a real event-level tape engine (`lab/data/tape.py`, numba, streamed):
OFI, cumulative delta, aggressor imbalance, VPIN-style, Kyle's λ from signed
volume, microprice-from-VWAP, big-trade imbalance, trade-sign autocorrelation —
from **10 liquid coins × 12 months of real aggTrades** (31 GB), aggregated
causally to M3/M5/M15. Maker execution model (`lab/exec/maker_model.py`,
tape-calibrated, ~8.5 bps round-trip).

**432-config cheap screen. 0 passed.** Every single config:
- rough IC: mean −0.003, **max +0.016**
- gross per-bar edge: mean −0.03 bps, **max +0.27 bps**
- net per-bar edge (after maker cost): **mean −0.65 bps, max −0.008 bps** — all negative

**Verdict: BOUNDED.** Directional bar signals built from aggregate trades at
M3–M15 have **essentially zero gross edge** (best IC 0.016, best gross edge
0.27 bps/bar) and are net-negative after even maker execution. The signal that
market-makers exploit lives at the second-to-10-second scale with L2 book
events, not aggregate-trade bars.

### 2.4 F_MICRO_OFI_EVENT — event-level OFI, 10 s – 5 min
10-second grid, trailing OFI over W seconds → forward H-second return,
held for the signal half-life. BTC / SOL / DOGE, 30 sampled days.

**13 configs. Best |IC| ≤ 0.023 (mostly negative). Gross Sharpe negative at
every window.** BOUNDED — aggregate-trade OFI carries no directional edge at
any accessible intraday timescale. Real OFI needs a live L2 feed.

### 2.5 F_XVENUE — Binance–Bybit funding-spread arbitrage
Downloaded Bybit funding history (55 coins, 2 yr). Spread =
`funding_binance − funding_bybit`, market-neutral & delta-neutral pair.

This one is **different**: the spread is a **real, mean-reverting inefficiency**
— `surrogate p = 0.01`, autocorrelation 0.53, gross harvest ~14 %/yr. But:

| check | result |
|---|---|
| net Sharpe, maker 4-leg | +3.0 (one config) |
| net Sharpe, **taker cost-stress** | **+0.22** |
| OOS (unfiltered config) | Sharpe **−0.42** |
| OOS (top-20 filter config) | Sharpe +2.7 **but flat 70 % of bars, ~1 coin on** |
| P&L concentration | **IDEXUSDT ≈ 40 % of gross**, then AMB / BOND / COMBO / DAR / LINA / MATIC / OM / LOOM — **all illiquid or delisted** |

**Verdict: RED-TEAM VETO → BOUNDED.** The spread is only wide and persistent
for coins with distressed liquidity or a venue winding the contract down —
exactly where the maker-execution assumption is false and you cannot actually
trade. For healthy liquid coins the spread is <1 bp/8h and net-negative after
any realistic 4-leg cost. Taker cost-stress Sharpe +0.22. This is a
liquidity/data artifact, not a harvestable strategy. Capacity would be low tens
of thousands of USD even if executable.

---

## 3. Objectives that are DATA_UNAVAILABLE / PARKED

Not bounded — untestable with free historical data. Each needs a **live
recorder** started now (per SYSTEM_SPEC §4.2):

| objective | why parked | recorder |
|---|---|---|
| F_OI (open-interest divergence) | Binance OI history ≈ 30 days; bulk `metrics` stop 2022 | WS `openInterest` |
| F_LIQD (liquidation cascades) | liquidation stream is real-time only | WS `forceOrder` |
| true L2 F_BOOK / F_MICRO_QUEUE / real VPIN | bookTicker/bookDepth bulk end 2024-04/05 | WS `depth` + `bookTicker` |
| cross-venue beyond Bybit (OKX, Hyperliquid) | need each venue's feed | per-venue recorders |

---

## 4. The bound, stated plainly

On **2 years of real Binance USDT-M perp data** (+ Bybit funding, + 31 GB of
real tape), with a **measured per-trade cost model** and a **tape-calibrated
maker execution model**, across **18 formula families** and **401 machine-
counted trials**, with placebo and block-bootstrap surrogate gates on every
candidate:

**There is no cross-sectional, time-series, carry, or aggregate-tape
microstructure signal that is tradeable net of cost at a feasible horizon.**

- The cross-sectional price/volume/funding space is bounded below the cost
  floor (best CV-stable IC ~0.014, need ~0.04).
- Funding carry is a decaying risk premium, not alpha.
- Aggregate-trade microstructure has ~zero gross directional edge at every
  timescale from 10 s to 1 day.
- The one genuine inefficiency found (cross-venue funding spread) is only
  present in un-tradeable illiquid/delisted names.

The remaining unexplored edge is **L2 order-book microstructure and the
liquidation/OI stream** — a different information regime that free historical
data does not contain. The next step is not more search on this data; it is a
**live recorder** feeding a 2–3 month dataset, then a focused L2 study.

---

## 5. Engine state (ready for backtest, extensible)

S0 green (10/10, kernels IDENTICAL). Two-tier trial counting live. Modules
added this run: `lab/data/{tape,microstructure,f1_micro}.py`,
`lab/exec/maker_model.py`, `lab/research/{micro,micro_study,carry_study,
carry_hardened,xvenue_study}.py`. `lab/research/montecarlo.py` ($25k risk sim)
built, unused — no candidate to simulate.

`HANDOFF.md` and `SYSTEM_SPEC.md` updated for the next session. Do **not**
re-run the 401 trials — climb to L2 (live L2 recorder) or L5 (a different
target: pure market-making / liquidity provision, which *earns* the spread
instead of paying it — the one game where cost is a tailwind).
