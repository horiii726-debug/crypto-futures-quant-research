# Research Factory — full-queue report (specs L–P)

Ran the full formula library as a pipeline. `data/test` never opened
(`test_looks = 0` in every ledger). No failure deleted, no gate loosened. New
families declared before results were seen. DSR is per-family, never pooled.

---

## Result

**58 distinct mechanisms** (F001–F094, duplicates collapsed) — `lab/research/REGISTRY.md`.
**Zero survivors.** The pattern from Rounds 1–3 holds under a systematic sweep:
no formula produces a cross-sectional or time-series edge that survives the
N3–N9 ladder net of the per-coin cost model, on the accessible data (free
Binance / Bybit history, 2021–2025).

| batch | family class | formulas | new runs | survivors | outcome |
|---|---|---|---|---|---|
| 1 | microstructure / order flow | 14 | 0 | 0 | F_MICRO*/F_BOOK bounded — COST_BOUND (gross edge up to 4.3 Sharpe, killed by turnover). F002/F011/F012 need L2 or reduce to bounded signed-flow. |
| 2 | volatility | 11 | 0 | 0 | sizing/gating only (`lab/exec/vol.py`). Directional F_VOL bounded, NO_EDGE. |
| 3 | trend / momentum / reversal | 9 | 0 | 0 | F_XSEC/F_DIR/F_TRANSFORM/F_DEP/F_PAIRS bounded — best DSR 0.14. |
| **4** | **crypto-native** | **10** | **6** | **0** | **F_CN4: F060 leverage ratio (placebo p 0.24), F061 OI divergence (placebo p 0.12), F063 funding momentum (IC 0.0009), F068 liq magnet — all NO_EDGE.** |
| — | term-structure (F064) | 1 | 24 cfg | 0 | **F_TERM: BTC/ETH quarterly-vs-perp slope. Best maker Sharpe +0.73 but surrogate p 0.55 (noise). UNDERPOWERED (2 assets).** |
| 5 | state / filtering | 8 | 0 | 0 | wrappers (`lab/exec/kalman.py`). F081 Kalman-pairs catastrophic, F087 Johansen bounded. |
| 6 | cross-asset / stat | 5 | 0 | 0 | F_XCOIN bounded — best DSR 0.09. |

New this round: **30 trials** (`ledger/round3.sqlite`) — F_CN4 (6) + F_TERM (24),
both bounded.

---

## What ran fresh and why it failed (spec O taxonomy)

| formula | screen | gate | root cause |
|---|---|---|---|
| F060 leverage ratio | ratio 148 @ D1 | placebo p 0.24, surrogate p 0.23 | **NO_EDGE** — a random cross-sectional ranking does as well. Confirms Round 3 J4 (Binance 2021-23: Sharpe +0.03). |
| F061 OI divergence (4-quadrant) | ratio 4.2 @ D1 | placebo p 0.12, surrogate p 0.003 | **NO_EDGE** — clears the surrogate noise ceiling on \|IC\| but a random ranking matches the net Sharpe. |
| F063 funding momentum | ratio 0.76 | — | **NO_EDGE** — IC 0.0009; the *change* in the funding trend carries nothing. |
| F068 liq magnet | ratio 0.11 | — | **NO_EDGE** |
| F064 term-structure (×4 signals, 24 cfg) | — | surrogate p ~0.5 everywhere | **NO_EDGE / DATA_LIMITED** — 2 assets, curve shape adds nothing over the (bounded) funding level. |

## Recorded but not run (data we do not have)

- **F002 multi-level OFI, F011 Hawkes, F012 propagator** — need a live L2
  order-book feed; bookDepth is ±1…±5 % bands. (Hawkes/propagator also reduce to
  lagged signed order flow = bounded F_MICRO_OFI.)
- **F065 cross-venue funding (+OKX)** — F_XVENUE (Binance–Bybit) is already
  VETOed (concentrated in delisted names); a third leg does not revive it.
- **F066/F067 liquidation** — Binance coin-M `liquidationSnapshot` is BTC/ETH
  only → F_LIQD, already UNDERPOWERED.
- **F069 stablecoin supply flow** — no on-chain data source ingested.
- **F022 EGARCH, F024 HAR-RV, F026 bipower, F029 vol-of-vol, F030 realized
  kernel, F084/F085/F086** — sizing/gating refinements, not directional signals;
  the directional vol family (F_VOL) is bounded.

---

## Cumulative state of the project

| | |
|---|---|
| distinct mechanisms in the library | 58 |
| signal families tested & bounded | 24 (+ F_CN4, F_TERM this round) |
| machine-counted trials (all ledgers) | 814 |
| **survivors** | **0** |
| red-team vetoes | 1 (F_XVENUE) |
| underpowered verdicts | 2 (F_LIQD, F_TERM) |
| **test-partition looks** | **0** |
| S0 self-tests | 10/10 · cost-model 11/11 · calibration PASS |

**The strongest lead of the whole project (`oi_leverage_state` / F060) failed
its cross-venue out-of-sample test in Round 3 and its placebo test here. There
is no surviving candidate.**

The bound is structural, and now systematically documented: at every horizon
reachable with free data and retail cost (~5–15 bps per side, per-coin), the
information the data contains is smaller than the cost of trading on it. This
holds across microstructure, volatility, trend, cross-sectional, cross-asset,
funding, open-interest, liquidation, cointegration and term-structure mechanisms.

## What could still change the answer (all require new inputs, not new formulas)

1. **A live L2 order-book + full tape recorder** — every microstructure bound
   says "real edge at the touch, we can't see it". Highest-value data investment.
2. **Longer, multi-venue OI/funding history via a paid vendor** (Laevitas,
   Amberdata, Coinalyze) — to move F_OI / F_CN4 past the single-regime problem.
3. **A lower cost base** — VIP fee tier + co-located maker infra would move the
   `cost_bps` floor down ~40 %; some COST_BOUND microstructure signals
   (F_BOOK `imbvol_fade`, gross Sharpe 4.3) would need re-checking.
4. **Options / vol-surface data** — the only major mechanism class not touched.
