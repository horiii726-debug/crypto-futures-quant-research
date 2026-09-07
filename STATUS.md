# STATUS — where the project is, and what to do next

_Last updated: 2026-09-07_

---

## Where it stands

**Zero tradeable survivors** after four research rounds. That is the result, and
it is documented with the evidence attached, not hidden.

| | |
|---|---|
| distinct mechanisms tested | **58** (Rounds 1–3) + **160** formulas (Round 4 tournament) |
| machine-counted trials | **844** across 3 ledgers |
| signal families bounded | **26** |
| **surviving strategies** | **0** |
| test-partition looks | **0** (never opened) |
| self-tests | 10/10 · cost-model 11/11 · leak calibration PASS |

## What was done, round by round

| round | what | outcome |
|---|---|---|
| **1** | S0–S7 engine, 305 cross-sectional trials, 13 families | NO_SURVIVOR — `COST_BOUND` everywhere |
| **2** | Per-coin cost model (`cost_bps`), re-score of all 193 configs, F_PAIRS (Johansen/OU/Kalman/Avellaneda) | 0 gate flips; cointegration does not persist OOS (s-score → forward spread corr **+0.004**, wrong sign) |
| **3** | Universe filter, regime attribution, vol/Kalman/execution/timing modules, multi-venue OI back to 2020 | **`oi_leverage_state` killed**: +1.2 held-out Sharpe on Bybit 2023-25 → **+0.03** on Binance 2021-23 |
| **4** | **Triple-barrier event system**, 160 formulas / 8 divisions, 5.49-year panel | Architecture works, cost problem solved — **but 5 winners all collapse OOS** |

## Round 4 in one line

The **architecture** was the right fix and it worked: event-driven triple
barrier with volatility-scaled SL/TP turned expectancy from *negative
everywhere* into **+160 … +237 bps per trade at 54–57 % win rate on 5–7 day
holds**. Cost as a fraction of the target move went from >100 % to ~0.1 %.

Then two corrections killed it:
1. **Sample uniqueness** — `n_eff = T/h`, so t 3.27 → 1.36 vs a 2.69 threshold.
2. **The 5.49-year test** — the effect collapsed (+237 → +28 bps); it never
   existed before 2023 and *loses* in the 2021 mania. Regime-conditional artifact.

Final measured bound: required SR/trade **0.159**, observed **0.027** — **5.9× short**.
Closing it needs 190 years of data or a genuinely 5.9× stronger edge.

See [`WINNERS.md`](WINNERS.md) and [`lab/reports/BARRIER_SYSTEM.md`](lab/reports/BARRIER_SYSTEM.md).

## The pattern (three times, same shape)

| lead | in-sample | out-of-sample |
|---|---|---|
| F_XVENUE cross-venue funding | net Sharpe +3.0 | 40 % of P&L from one **delisted** coin → red-team veto |
| `oi_leverage_state` | held-out Sharpe **+1.22** | **+0.03** on a different venue and regime |
| low-vol / jump family | +237 bps, t 2.82 | **+28 bps**, t 0.36 on 5.5 y |

Three independent leads, three collapses on genuinely out-of-sample data. That
consistency is itself the finding: **the 2023-2025 crypto sample contains
patterns that do not generalise**, and free-data directional alpha at retail
cost is bounded.

---

## What is built and reusable

| layer | module |
|---|---|
| **Barriers** | `lab/engine/barriers.py` — Yang-Zhang / Rogers-Satchell / Parkinson σ, first-passage TP/SL solver, wick-floored stops, sample uniqueness |
| **Backtest** | `lab/engine/event_backtest.py` — event-driven, cooldown, concurrency caps, min-hold; `lab/engine/backtest.py` — continuous, R9-enforced |
| **Cost** | `lab/exec/cost_model.py` — per-fill: fee (from `config/account.yaml`) + per-coin half-spread + √-impact + queue `p_fill` + adverse selection + walk-the-book. BTC reconciles to 0.7 % |
| **Sizing** | `lab/exec/vol.py` — Yang-Zhang, EWMA, GJR-GARCH, vol-targeting |
| **Filtering** | `lab/exec/kalman.py` — dynamic hedge ratio, local-level smoother, innovation z |
| **Execution** | `lab/exec/execution.py` — Almgren-Chriss split, maker-first ladder, netting; `timing.py` — cost-aware entry gate |
| **Universe** | `lab/exec/universe.py` — frozen rolling liquidity filter |
| **Formulas** | `lab/divisions/d1…d8` — 160 formulas, each with its paper reference and the author's defaults |
| **Pipeline** | `lab/research/tournament.py`, `pipeline.py`, `champion_validate.py` (deflated-t, placebo, surrogate, held-out) |
| **Integrity** | append-only ledger (`lab/ledger.py`), sealed test partition, numeric firewall + leak canary git hooks |

**If a genuine signal ever arrives, the machine to trade it correctly is finished.**

---

## NEXT — ranked by expected value

### 1. Market making / liquidity provision  ← recommended
Stop predicting direction. Quote both sides, **earn** the spread instead of
paying it. This flips the equation on the signals that were `COST_BOUND`:
`imbvol_fade` has a **gross Sharpe of 4.3** and dies only on turnover — as a
market-maker quote skew, turnover is the business, not the cost.

- Framework: **Avellaneda-Stoikov 2008** + **Guéant-Lehalle-Fernandez-Tapia 2013**
- Inventory control: `vol.py` (GJR-GARCH) + `kalman.py` — already built
- Adverse selection: already calibrated in `maker_model.py`
- ML layer that is actually well-posed: gradient-boosted **toxic-flow / adverse-selection**
  prediction over 5–30 s → widen or pull quotes. Clean labels, millions of samples.
- Backtest on the aggTrades tape already on disk (passive fills already simulated)
- Est. 2–4 weeks to a paper-tradeable system

### 2. Buy the data that removes the `DATA_LIMITED` bounds
- Run an **L2 order-book recorder** for 3 months (free, costs time), or
- Vendor: Kaiko / Amberdata / Laevitas (~$500–5 000/mo)
- Only then is directional ML worth trying: DeepLOB (Zhang-Zohren-Roberts 2019),
  Kolm et al 2023 — these work **on real L2**, not on ±1 % depth bands

### 3. Lower the cost base
VIP 1–2 fee tier + maker-only + co-location moves the cost floor down ~40 %.
Then re-check the `COST_BOUND` signals (`imbvol_fade`, `slope_amplify`).

### 4. Accept a lower-Sharpe mandate
Funding carry as a hedged beta position: ~0.6 net taker Sharpe, positive carry,
known risks. Not alpha — a real allocation, honestly labelled.

### Not recommended
- **More formulas on the same data.** 218 mechanisms tested; the bound is
  structural (`t = SR × √(T/h)`), not a search-coverage problem.
- **Deep learning on returns.** Crypto has 5 years × ~14 effective independent
  assets. Gu-Kelly-Xiu-style ML works in equities with 100 years × 6 000 names —
  the sample here is ~1 000× too small.

---

## Quick start for whoever picks this up

```bash
export PYTHONPATH=.
export CRYPTO_LAB_LEDGER_SQLITE=./ledger/lab.sqlite

python3 tests/test_lab.py          # 10/10 engine integrity
python3 tests/test_cost_model.py   # 11/11 cost model
python3 tests/calibration.py       # planted edge caught, zero-edge control rejected

python3 lab/data/long_panel.py     # rebuild the 5.49y panel (needs network)
python3 lab/research/tournament.py D2_VOL   # run one division
```

Read in this order: `README.md` → `WINNERS.md` → `lab/reports/BARRIER_SYSTEM.md`
→ `DATA_INVENTORY.md` → `CLAUDE.md` (the rules that keep it honest).
