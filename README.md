# crypto-futures-quant-research

A from-scratch quantitative research engine for crypto perpetual futures, and the
**negative result** it produced: across **58 distinct signal mechanisms** and
**814 pre-registered trials**, no strategy survives a proper out-of-sample,
cost-aware, multiple-testing-adjusted evaluation on free exchange data.

The point of the repo is the **process**: an append-only experiment ledger, a
pre-registered gate ladder, a per-trade cost model calibrated on real tape, and a
discipline that makes it hard to fool yourself. The conclusion — that retail-cost
crypto perp alpha from free data is bounded below the cost floor at every
reachable horizon — is stated with the evidence attached.

> Research code. Not investment advice. No strategy here is tradeable; that is the finding.

---

## The result in one table

| | |
|---|---|
| distinct signal mechanisms tested | **58** (F001–F094, duplicates collapsed) |
| signal families | 24 (cross-sectional, microstructure, volatility, funding, open-interest, liquidation, cointegration, term-structure, cross-asset) |
| machine-counted trials (all ledgers) | **814** |
| **surviving strategies** | **0** |
| red-team vetoes | 1 (cross-venue funding spread — real inefficiency, un-tradeable) |
| underpowered verdicts | 2 (BTC/ETH-only liquidation, BTC/ETH-only term structure) |
| **test-partition looks** | **0** (never opened) |
| self-tests | 10/10 · cost-model 11/11 · leak calibration PASS |

The strongest lead of the whole project — `oi_leverage_state` (short the coins
whose open interest is large relative to traded volume) — showed a **+1.2
held-out Sharpe on Bybit 2023–2025 data**, then **failed** when re-run on Binance
open-interest data over the 2021–2023 LUNA/FTX regime (**+0.03 Sharpe, surrogate
p = 0.47**) and failed a placebo test. It is a sample-specific pattern, not an
edge. Full write-up: [`lab/reports/FACTORY_REPORT.md`](lab/reports/FACTORY_REPORT.md).

---

## Why believe the negative result

The engine is calibrated to catch a real edge and reject a fake one:

- **Leak calibration** (`tests/calibration.py`): a synthetic dataset with a
  planted IC = 0.35 is classified `CANDIDATE`; the same pipeline on a zero-edge
  control is classified `NOT A CANDIDATE` (it loses ~cost and fails the noise
  ceiling). Both must pass on every commit.
- **The cost model reconciles**: `cost_bps("BTCUSDT", …)` round-trip = 5.70 bps
  vs the 5.66 bps measured by simulating limit orders on the real aggTrades tape
  (0.7% error). Per-coin, size-aware (square-root impact), with a real
  walk-the-book from the ±1…±5% order-book ladder.
- **Execution off-by-one is enforced by a bit-identical kernel test**: a signal
  formed on bar `t` earns bar `t+2`'s close-to-close return; `oracle(t+1)` must
  NOT profit.
- **Multiplicity is paid for**: the Deflated Sharpe Ratio reads the cumulative
  trial count *from the ledger* (the machine counts, not the researcher) and is
  scoped per family.

---

## The gate ladder (every trial runs all of it)

```
N3  cheap screen        edge_bps / cost_bps ≥ 3 at ≥1 timeframe  (all 6 TFs logged)
N4  full backtest       event-driven, per-fill cost, maker-first execution, rolling universe filter
N5  purged K-fold CV    embargo = 1.5 × max(horizon, hold)
N6  placebo             shuffle the cross-sectional ranking, 300+ draws, p < 0.05
N7  surrogate           block-bootstrap the forward returns, 300+ draws, p < 0.05
N8  PBO                 probability of backtest overfit (CSCV) < 0.5
N9  DSR                 deflated Sharpe ≥ 0.95 vs the family's ledger trial count
N10 sealed OOS          one look per family, only after every gate passes  (never reached — no candidate)
```

On failure, a root cause is written from a fixed taxonomy: `COST_BOUND`,
`NO_EDGE`, `OVERFIT`, `REGIME_DEPENDENT`, `DATA_LIMITED`, `IMPLEMENTATION`,
`METRIC_MISMATCH`.

---

## Layout

```
CLAUDE.md               constitution — the non-negotiable rules (R1–R10)
SYSTEM_SPEC.md          consolidated design spec
lab/
  ledger.py             append-only SQLite; UPDATE/DELETE rejected by triggers
  exec/
    cost_model.py       per-trade, per-coin: fee · half-spread · sqrt-impact ·
                        queue p_fill · adverse selection · walk-the-book
    universe.py         frozen liquidity filter (spread ≤ 1bp, ADV ≥ 50M, listed ≥ 90d)
    vol.py              Yang-Zhang · EWMA · GJR-GARCH · vol-targeting  (sizing only)
    kalman.py           dynamic hedge ratio · local-level smoother · innovation z
    execution.py        Almgren-Chriss order split · maker-first ladder · netting
    timing.py           entry timing inside a fixed rebalance window
    maker_model.py      limit-order fills simulated on the real trade tape
  stats/                dsr · pbo (CSCV) · mcs (+SPA+DM) · surrogate · power
  engine/               numba 3-tier bit-identical kernels · purged CV · backtest
  features/             crypto_native.py (F060–F070) · micro · …
  research/
    pipeline.py         the N3–N9 harness + failure taxonomy
    REGISTRY.md         all 58 mechanisms, their family, gate status, root cause
    formula_registry.py the registry as data
    campaign.py         the S1–S7 cross-sectional campaign (305 trials)
    pairs_study.py      F_PAIRS — Johansen · OU · Kalman · Avellaneda-Lee · GJR-GARCH
    oi_study.py oi_leverage_hardened.py oi_crossvenue_rescore.py   F_OI + candidate
    carry_study.py xvenue_study.py liq_study.py term_structure.py batch4.py
  data/                 binance_vision · binance_metrics (OI to 2020-09) ·
                        multi_venue · book_ladder · quarterly · tape · liqsnap
  live/paper_trader.py  calendar-time paper harness (gated behind human sign-off)
  guards/               test_seal · numeric_firewall · leak_canary  (git hooks)
data/{raw,processed,train,valid,test}    test/ is SEALED — regenerable, not committed
ledger/                 *.sqlite (local, gitignored) — schema in lab/ledger.py
lab/reports/            FACTORY_REPORT · ROUND{2,3}_REPORT · RESCORE_305 · BATCH_{1..6} · …
tests/                  test_lab.py (10) · test_cost_model.py (11) · calibration.py
```

---

## Reproduce

```bash
pip install numpy pandas scipy numba statsmodels arch pyarrow pyyaml

export PYTHONPATH=.
export CRYPTO_LAB_LEDGER_SQLITE=./ledger/lab.sqlite

python3 tests/test_lab.py          # 10/10 — engine integrity
python3 tests/test_cost_model.py   # 11/11 — cost model + BTC reconciliation
python3 tests/calibration.py       # planted-edge caught, zero-edge control rejected

# rebuild the cost model calibration (needs data.binance.vision access, free)
python3 lab/data/book_ladder.py
python3 lab/exec/universe.py

# run a batch (writes lab/reports/BATCH_4.md + ledger/round3.sqlite)
python3 lab/research/batch4.py
python3 lab/research/formula_registry.py   # regenerate REGISTRY.md
```

Data (`data/`) is not committed — it is ~30 GB and fully regenerable from
`data.binance.vision` (klines, aggTrades, bookDepth, metrics/OI, funding — all
free, no key) and the Bybit public API. The ledgers (`ledger/*.sqlite`) are
local and gitignored; the schema is in `lab/ledger.py`.

---

## What would change the answer (new data, not new formulas)

1. A live L2 order-book + full tape recorder — every microstructure bound says
   "real edge at the touch, we can't see it" (F_BOOK `imbvol_fade` has a gross
   Sharpe of 4.3, destroyed by turnover).
2. Longer multi-venue OI / funding history via a paid vendor — to get F_OI past
   the single-regime problem.
3. A lower cost base (VIP fees + co-located maker infra) — moves the `cost_bps`
   floor down ~40%.
4. Options / vol-surface data — the one major mechanism class untouched here.

---

## License

MIT. See [`LICENSE`](LICENSE).
