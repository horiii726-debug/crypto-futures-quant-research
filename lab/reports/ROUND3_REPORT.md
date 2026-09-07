# Research Round 3 — report (specs A–J)

Executed in order. `data/test` never opened. No failure deleted, no gate
loosened. `UNIVERSE_RULE.md` written and committed before use.

---

## Headline

**J4 is decisive: `oi_leverage_state` does not generalise.** On Bybit
2023-2025 it looked like a +1.2 held-out Sharpe near-miss; on **Binance OI
2021-2023 (the LUNA + FTX regime) it is noise** — maker Sharpe **+0.03 to
+0.12**, surrogate p **0.47**. The one open candidate is now bounded as a
sample-specific pattern. **The search has no surviving candidate.**

---

## G — regime attribution (not a trial)

`oi_leverage_state` P&L split by regime, all states causal:

| dimension | result |
|---|---|
| BTC trend (up/down/flat) | **EVEN** — P&L tracks time in each bucket |
| funding sign (pos/neg) | **EVEN** |
| realised-vol tercile | **WEAK in low-vol** — Sharpe ~0.1 there (vs 2.5–3.6 in mid/high), not negative |

Read: not a pure regime bet, but the edge needs market dispersion and vanishes
in calm markets. (`lab/reports/REGIME_ATTRIB.md`)

## H — universe filter (pre-registered, committed)

`lab/reports/UNIVERSE_RULE.md` + `lab/exec/universe.py`, frozen:
`median_30d(half_spread) ≤ 1.0 bps` AND `median_30d(ADV) ≥ 50 M` AND
`listed ≥ 90 d`. Yields 13–20 liquid coins over 2024–2025; CRV, HOOK, ZEC,
thin/distressed names excluded. Rolling, in-sample, no lookahead.

### H3 — `oi_leverage_state` on the frozen universe

| | maker net SR | taker net SR |
|---|---|---|
| static liquid-28, full sample | +2.26 | +1.96 |
| **rolling filter, full sample** | **+1.04** | **+0.88** |
| static-28, held-out 30% | +1.22 | +0.90 |
| **rolling filter, held-out 30%** | **+1.11** | **+0.91** |

The inflated full-sample number came from the broader static list trading names
that later went illiquid. The **held-out** figure (the decision-relevant one) is
unchanged by a strict pre-registered filter — but see J4.

## A — cost model finalised

`cost_bps()` now: fee from `config/account.yaml` (**A1** — no hidden retail
guess; a bug-check note in the file), per-coin half-spread, square-root impact,
queue `p_fill`, adverse selection, **walk-the-book from the real ±1…±5 %
bookDepth ladder** (**A6**, `lab/data/book_ladder.py`, 28 coins). Every fill can
be written to the ledger `fills` table one row at a time
(`log_fill` / `reconcile_fill` / `fills_summary`). **11/11 unit tests pass**
incl. `cost(BTC) < cost(SOL) < cost(CRV)` and BTC reconciliation 0.7 %.

## D — volatility (`lab/exec/vol.py`)

Yang-Zhang, EWMA (λ=0.94), **GJR-GARCH(1,1)** (MLE, asymmetric), and
`vol_target_weight` — **sizing/timing only, never direction.** Tested:
weight shrinks 3× into a vol spike.

## E — Kalman (`lab/exec/kalman.py`)

Dynamic hedge ratio (Q/R = 1e-5, one param), local-level smoother (MA
replacement, adaptive lag), normalised-innovation z-score for **gating, not
direction**. Tested: recovers a time-varying β to |error| 0.014.

## C / I — execution (`lab/exec/execution.py`)

Order split (Almgren-Chriss: 4 slices → impact ÷√4), maker-first ladder
(20 s TIF, 3 attempts, then taker; per-attempt `p_fill` from the queue model),
netting (send the delta, never round-trip the same side). Wired into
`paper_trader.py` so an armed run logs `p_fill_predicted` vs `filled` — the
**I2** validation of the queue model. **I4** target `p_fill ≥ 0.6` is measured,
not assumed (ETH $20 k ladder predicts ~1.0; thin coins lower).

## F — entry timing (`lab/exec/timing.py`)

Adaptive threshold `c·σ̂`, cost-aware gate (execute a slice only if it beats the
window-median cost by a margin), spread-state gate (skip if spread > 2× 30 d
median), **forced taker at window end**. The only metric is Δ realised cost bps
vs the fixed-offset baseline — never Sharpe. On synthetic data the saving is
~0 (consistent with the P2.2 imbvol-timing result: ~0.3 bps, not material).
Real-data evaluation is limited by intrabar book coverage (~8 months, 12 coins).

## J — multi-venue ingest (the blocker)

- **`lab/data/binance_metrics.py`** — Binance vision `metrics` → OI in **USD**,
  hourly. BTC from **2020-09**, most majors from **2021-12** (covers LUNA
  2022-05 and FTX 2022-11 — regimes the Bybit 2 y sample never saw).
- **`lab/data/multi_venue.py`** — one UTC hourly grid, OI in USD, funding
  annualised, delisted kept.
- **J3 validation:** Bybit vs Binance OI level-correlation in the 1-month
  overlap: **8/19 coins > 0.8** (BTC 0.85, XRP 0.93, LINK 0.94, AVAX 0.89,
  DOT 0.86, BCH 0.92, XLM 0.89, APT 0.93; ETH 0.44, BNB 0.15, AAVE 0.07,
  NEAR −0.38 fail). OI-change correlation is ~0.25 for all coins including BTC.
  Short overlap + genuine venue divergence; some contribution to J4 but not
  enough to explain it.
- **J4 — `oi_leverage_state` re-run on Binance OI 2021-2023** (same signal,
  same params, not a trial):

| window | maker Sharpe (ann) |
|---|---|
| full 2021-01 … 2023-09, 8 OI-validated coins | **+0.12** |
| full, all 23 coins | **+0.03** |
| LUNA 2022 H1 | +1.17 |
| **FTX 2022 H2** | **−1.09** |
| recovery 2023 | +0.17 |

walk-forward [0.0, −0.12, +1.05, −0.51] · surrogate p **0.47** · max DD −0.33.

**The Bybit 2023-2025 +1.2 held-out Sharpe does not reproduce.**
`oi_leverage_state` is a sample-specific pattern. F_OI bound extended
(`ledger/lab.sqlite`).

---

## Where this leaves the project

- **23 families / bounds** (22 + the `oi_leverage_state` cross-venue bound).
- **No surviving candidate.** The strongest lead of the whole project failed
  its cross-venue / cross-regime out-of-sample test.
- New infrastructure delivered and tested: `cost_model` (per-fill, walk-the-
  book, config fees), `universe`, `vol`, `kalman`, `execution`, `timing`,
  multi-venue OI back to 2020-2021.

## K — new-trial gate (NOT opened)

Spec K opens "kalau J berhasil". **J did not succeed**, so K is not triggered
automatically. `lab/reports/ROUND3_GATE.md` holds the pre-registered K design
(30-trial hard budget, separate ledger, pre-screen `edge_bps > 3·cost_bps` &
turnover ≤ 0.05 & horizon ≥ 4 h; candidates: perp-vs-quarterly basis /
term-structure, funding-momentum persistence, cross-venue OI divergence,
liquidation-cascade aftermath 24–72 h). **Your call whether to open it** — the
multi-venue data now makes the cross-venue-OI and funding-momentum candidates
feasible.
