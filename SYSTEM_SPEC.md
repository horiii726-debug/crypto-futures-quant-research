# CRYPTO QUANT FACTORY — SYSTEM SPEC v2 (backtest + live)

Consolidation of `CRYPTO_QUANT_FACTORY_FINAL_CLAUDE_CODE.txt` and
`CRYPTO_QUANT_FACTORY_MASTER_PROMPT.txt`, cleaned by a quant dev. Where the two
source files contradicted or were not implementable, the resolution and the
reason are marked **[FIX]**. This file supersedes both source files and
`CLAUDE.md` for anything they disagree on.

---

## 0. MISSION

Build ONE research-to-live engine for crypto perpetual futures. The goal is a
strategy with **positive net expectancy that survives cost, execution, OOS and
live reality** — not a good-looking backtest.

- No seed strategy. No formula is assumed correct.
- Never force a survivor. Never relax a gate. Never invent data/fees/specs.
- Backtest profit ≠ live-ready.

Autonomous loop: `OBSERVE → DIAGNOSE → ROOT CAUSE → FIX/NEW HYPOTHESIS → TEST →
VALIDATE → REPEAT`. Do not stop on a failed batch, small Sharpe, small IC,
negative net PnL, thin data, or high cost.

**Stop only when:**
1. a candidate passes every gate through S7 → `READY_FOR_PAPER`; or
2. every item in the *enumerated* objective queue (§7) has a verdict or a
   `BOUND`; or
3. a step genuinely cannot be automated (live capital, exchange account
   creation, funding a wallet, a calendar-time wait).

Live capital always needs human sign-off.

---

## 1. THE TEN RULES (constitution — non-negotiable)

| # | Rule |
|---|------|
| R1 | The LLM never states a metric as fact. Every number in a report resolves to a ledger `experiment_id`. |
| R2 | `data/test/` is sealed. Opened only by `unseal_test.py`, one look per family, no `--force`, logged. |
| R3 | Trial count is computed by the machine from the ledger and **never reset**. DSR reads it. See **[FIX 1]** for what counts as a trial. |
| R4 | Discovery and validation are walled. A material parameter change = a new hypothesis (`parent_id` linked). |
| R5 | `UNDERPOWERED` and `UNKNOWN` are valid verdicts. A null on a weak sample does not refute the hypothesis. |
| R6 | Failures are never deleted. Root cause + lesson + `next_action` stored for every one. |
| R7 | Live-parity: a feature that cannot be computed in real time from the production feed is invalid for a live strategy. |
| R8 | Live capital deployment requires human sign-off. |
| R9 | A signal formed at time/bar `t` is executed no earlier than `t+1`. In the backtest the position that earns the return realised at bar `k` was decided at `k − (execution_lag + 1)` (see `lab/engine/backtest.py`). |
| R10 | Survivorship: a delisted symbol stays in the sample for every period the historical universe definition includes it. |

### Forbidden
Loosen a gate to pass · tune on the test set · hide a negative result · reset
the trial count · pick the window that happens to pass · invent
data/fees/specs · call a backtest "profitable" or "live-ready".

### Escalation ladder (climb on failure, never sideways)
L1 fix sample quality → L2 change timeframe/horizon (within the F-1 feasible
set) → L3 re-spec the hypothesis → L4 new mechanism → L5 new target → L6 write
a `BOUND`, move to the next objective. A **bounded** family stays frozen until
new admissible information appears.

---

## 2. **[FIX 1] MULTIPLICITY — the biggest design flaw in the source files**

The source files ask for "hundreds to thousands of formulas" **and** a single
global cumulative trial count that DSR reads. Those are mathematically
incompatible: at 3000 trials DSR needs an annualised Sharpe near 4 to pass, so
nothing can ever survive — the search defeats itself. Resolution:

**Two-tier counting.**

1. **Cheap screen** (rough IC, turnover, signal frequency, cost feasibility,
   1-fold holdout sanity) runs on hundreds/thousands of formulas and **does NOT
   increment the trial count.** It only ranks and discards. It never touches
   `data/valid` beyond a single rough holdout and never touches `data/test`.
2. **Full evaluation** (purged CV + embargo + placebo + surrogate + DSR + PBO +
   cost stress) runs only on the screen's survivors and **each one is a
   trial.** These are pre-registered in small batches.

**Budgets and DSR scope.**
- Each family (§6) has a `max_full_trials` budget in `config/budgets.yaml`
  (default 40, priority families 60–80). Exhaust it → family frozen, write a
  `BOUND`.
- **DSR is evaluated at the family level** against that family's full-trial
  count (this is the number that gates a candidate at G5).
- A **portfolio-level DSR** against the *global* full-trial count is computed
  once, only for a candidate that has already passed G5 at the family level,
  as the final multiplicity check before `READY_FOR_PAPER`. A candidate that
  cannot clear portfolio DSR is `NO_EDGE` (over-searched), not resurrected.
- The cheap screen's count is logged separately (`screen_evals`) for honesty
  but is not a DSR input.

**Pre-registration.** Before any full batch: write the hypothesis record
(§5 schema) and the exact grid to the ledger. Adding a grid point after seeing
results = a new hypothesis (R4).

---

## 3. VENUE

### **[FIX 2] Research venue ≠ production venue**

The source files name **Bybit** as production but every free multi-year
historical archive (klines, aggTrades, premium index, some book/depth) is on
**Binance** (`data.binance.vision`). Bybit's public history
(`public.bybit.com`) has trades but sparse order-book / OI history. Resolution:

- **Research venue: Binance USDT-M perp.** Deepest free archive; used for
  discovery and validation of *signal* existence.
- **Production venue: Bybit USDT perp** (or Binance — a `config/venue.yaml`
  choice). All *execution* assumptions (fees, tick, funding, fill model, cost
  model) use the production venue's official spec. A candidate is only
  `READY_FOR_PAPER` after its edge is re-checked under production-venue
  execution assumptions.
- Never mix venues in one dataset without a provenance row and a written
  methodological reason. Cross-venue lead-lag is allowed as an explicit
  hypothesis with both venues recorded.
- If a signal exists on Binance data but the production venue lacks the data
  to run it live (e.g. a DOM feature and Bybit book history is thin) → the
  candidate is `PARKED` with a live-recorder requirement, not shipped.

### Contract spec — pull from the official API, never guess

`config/venue.yaml` fields (`maker_fee`, `taker_fee`, `funding_interval`,
`funding_cap`, `tick_size`, `min_notional`, `step_size`, `leverage_tier`,
`maintenance_margin`, `liquidation_fee`, `mark_price_basis`) are **NULL until
filled from the venue's official REST/docs**. NULL blocks anything that needs
the value. Per-symbol values live in a keyed map, not a scalar.

- Binance: `fapi.binance.com/fapi/v1/exchangeInfo`, `/fapi/v1/fundingInfo`,
  official USDⓈ-M fee schedule.
- Bybit: `api.bybit.com/v5/market/instruments-info?category=linear`,
  `/v5/market/tickers`, Bybit fee-rate docs (VIP 0: maker 0.02% / taker
  0.055% for linear perps — verify at build time, it changes).

---

## 4. DATA

### 4.1 Required streams (priority order for a microstructure engine)

`OHLCV` · `TRADES/TAPE + aggressor side` · `ORDER BOOK / DOM / L2` ·
`SPREAD / book-top` · `OPEN INTEREST` · `FUNDING` · `LIQUIDATION` ·
`premium index / basis`.

### 4.2 **[FIX 3] Data availability is a hard constraint — enumerate it first**

Discovery step before any download: probe availability, coverage, missingness,
storage. Then generate a data plan. Known state (Binance, verified this
project — re-verify, it drifts):

| stream | free history | status |
|---|---|---|
| klines 1m | 2019/2020 → now | OK |
| aggTrades (tape + aggressor) | full | OK — the core microstructure source |
| premiumIndexKlines / markPriceKlines | full | OK — basis / carry |
| fundingRate | full | OK |
| bookTicker (BBO) | **ends ~2024-04** | PARTIAL — calibration only, or live recorder |
| bookDepth (DOM ±1–5%) | **ends ~2024-05** | PARTIAL — ~8 months usable |
| openInterestHist | **~30 days** (REST) | `DATA_UNAVAILABLE` for backtest |
| long/short ratio | **~30 days** | `DATA_UNAVAILABLE` for backtest |
| liquidation stream | real-time only | `DATA_UNAVAILABLE` for backtest |

**Where history is missing: start a live recorder now** so the dataset exists
for the next cycle. Do not fabricate. Mark the family `PARKED` with the
recorder as `next_action`.

### 4.3 Timeframes

Primary: **M3, M5, M15**. Microstructure is processed at **event level** from
the tape/book, then aggregated *causally* to M3/M5/M15. Every hypothesis is
tested at all three; the timeframe is chosen by robustness/OOS, never by
highest return.

### 4.4 Quality gate (data-warden VETO)

timestamp integrity & timezone · duplicates · gaps · symbol rename/mapping ·
contract changes · tick-size changes · maintenance windows · outliers · trade
classification consistency · book continuity · funding anomalies · OI/liq
discontinuity. Every dataset row carries: `venue, symbol, timestamp, timezone,
source, schema_version, contract_spec, quality_status`.

### 4.5 Universe

Explicit, versioned, pre-defined entry/exit/ranking/rebalance/liquidity rules.
Delisted symbols retained per R10. Never chosen from test results.

### 4.6 Partitions

Chronological. `train` / `valid` / `test`. `test` sealed until S7. SHA-256 of
each partition in the ledger.

---

## 5. COST MODEL — audited BEFORE any edge is judged

### **[FIX 4] No fixed bps. The maker model is a prerequisite, not an option.**

The source files forbid a universal 8/10/14 bps cost. Correct. Per-trade cost =

```
commission(maker|taker, venue tier)
+ spread cost          (measured half-spread, per symbol/vol/liquidity state)
+ slippage             (size vs top-of-book / ADV)
+ market impact        (size vs depth; from bookDepth or a calibrated model)
+ adverse selection    (maker fills are informed-selected; measured from tape)
+ funding              (over the holding period, on net exposure)
```

- **Maker and taker separated.** Fill probability modelled. Partial fills
  modelled when the strategy needs them.
- **Hybrid execution** (`lab/exec/maker_model.py`): rest at the touch,
  `p_fill` and adverse selection **simulated on the real tape**, taker
  fallback after a wait, opportunity cost of the wait charged. No 100%
  maker-fill assumption.
- Store separately every run: `gross_pnl, commission, spread_cost, slippage,
  impact, funding, net_pnl`.
- Cost model not yet verified → `COST_UNCERTAIN`, **not** `NO_EDGE`.
- A run that concludes `COST_FAILURE` triggers a mandatory audit: re-pull
  official fees, re-measure spread/slippage/fill/impact, check double-counting
  and maker/taker logic. If a cost-model bug is found: **invalidate the
  affected results and rerun legitimately.**
- Cost verified but high → objectively test lower-turnover specs, longer
  feasible horizons, realistic maker execution, liquidity filters, a trade
  threshold, position-size caps. **Never change the cost to make a strategy
  look profitable.**

### F-1 feasibility gate (execution-analyst)

For each (timeframe × execution-mode) compute the cost floor and the required
IC / hit-rate. `required_IC ≈ turnover × cost_floor / dispersion`. Any pair
whose required IC exceeds a stated ceiling (default 0.05 for cross-sectional,
0.08 for single-asset directional) is **INFEASIBLE and closed** — a `BOUND`,
not a family the search keeps trying. Rebuild F-1 whenever the cost model or
execution model changes.

*(Project finding: taker-only closes M3–M4h and 1d. Hybrid execution reopens
1d. M3/M5/M15 need maker execution AND a signal whose half-life exceeds the
holding period — otherwise they stay closed.)*

---

## 6. FORMULA / HYPOTHESIS CORPUS

Hundreds+ of candidates through the **cheap screen** (§2 tier 1). Families
(the enumerated objective space — this is what "bounded" is measured against):

`F_VOL` `F_STATE` `F_FILTER` `F_TS` `F_MICRO_OFI` `F_MICRO_VPIN`
`F_MICRO_BOOK` `F_MICRO_QUEUE` `F_MICRO_LIQ` `F_FUND` `F_BASIS` `F_OI`
`F_DIR` `F_ENTRY` `F_EXIT` `F_MOMENTUM` `F_REVERSAL` `F_BREAKOUT`
`F_DIST` `F_PATH` `F_INFO` `F_XASSET` `F_XVENUE` `F_META_ML`

Each family: methods from the literature (identifiers must resolve;
UNRETRIEVED papers are not summarised; falsification search mandatory).
Literature formula → store source + equation. Combined formula →
`NEW_HYPOTHESIS`.

### Hypothesis schema (`schemas/hypothesis.schema.json`)
`claim · mechanism · math_form · inputs · target · horizon · assumptions ·
failure_conditions · live_parity · null_hypothesis · effect_size_of_interest ·
source_ids · lessons_reviewed · bounded_search_space`.

### Microstructure priority (needs mechanism, not brute force)
OFI + OI-acceleration · aggressor sequence + spread state · microprice + book
imbalance · queue/resiliency + tape pressure · liquidation shock + OI/price
response · funding shock + order flow. Each combination is a hypothesis with a
written mechanism.

---

## 7. OBJECTIVE QUEUE (what "bounded" means)

The search space is **finite and enumerated**:

```
objective = family × timeframe(M3,M5,M15) × mode(time-series, cross-sectional)
            × execution(taker, hybrid-maker)
```

Each objective has a `max_full_trials` budget. The queue is **bounded** when
every objective has: a `SURVIVOR`, or a `BOUND` (with economic + data +
statistical reasoning and the closed sub-space), or `DATA_UNAVAILABLE/PARKED`
with a recorder plan. "Thousands of formulas" is not a bounded space — the
*families* are.

---

## 8. STAGES — what is autonomous vs not

| Stage | Content | Autonomous? |
|---|---|---|
| S0 | Build + verify engine. Synthetic control/planted self-tests. 10 S0 tests + kernels IDENTICAL + seal blocks + leak canary. E2E calibration: zero-edge ⇒ not a candidate, planted ⇒ candidate. | Yes |
| S1 | Data: venue spec, universe, partitions+hashes, **cost model**, F-1. | Yes (recorders run over calendar time) |
| S2 | Corpus: cheap screen over the families. | Yes |
| S3 | Alpha research: primary signal, M3/M5/M15, ts + xs. | Yes |
| S4 | Strategy construction (entry/exit/sizing/risk/exec). ML meta-layer only on a signal that already passed. | Yes |
| S5 | Backtest: portfolio, live-parity execution, per-trade cost. | Yes |
| S6 | Validation: purged CV, embargo, walk-forward, DSR, PBO, MCS/SPA, power, sensitivity, regime split, placebo, surrogate, cost/slippage/turnover stress. | Yes |
| S7 | OOS: one sealed test look per family, logged. | Yes |
| **S8** | **Paper trading — calendar-time (weeks+). Engine STOPS at `READY_FOR_PAPER`; a separate paper harness (`paper/`) runs live and logs. Human compares paper vs backtest distributions.** | **No — calendar time** |
| S9 | Execution testing: realised fee/spread/slippage/fill/latency/reject/adverse selection vs model. | No — live account |
| S10 | Live review: full gate checklist + small size + hard limits + kill switch + human sign-off. | No — human |

Never skip a stage. Never relax a gate.

---

## 9. AGENTS (all write to the ledger; work on the same artifacts)

`director` (orchestration, budget, queue, ladder — no compute, no verdict,
cannot overturn a veto) · `librarian` · `theorist` (reads lessons first) ·
`data-warden` **[VETO]** · `research-coder` (no parameter-picking from results)
· `experiment-runner` (only role that unseals test) · `validator` **[VETO]** ·
`red-team` **[VETO]**, no positive claims · `risk-officer` **[VETO]** ·
`execution-analyst` **[VETO]**.

Gate order: G0 source · G1 data · G2 code · G3 discovery · G3.5 power ·
G4 OOS/CV · G5 multiplicity (family DSR) · G6 robustness · G7 cost ·
G8 risk · G9 red-team · G10 portfolio-DSR + human.

---

## 10. FAILURE TAXONOMY (never FAIL→DELETE)

`VALID · TESTING · DATA_FAILURE · IMPLEMENTATION_FAILURE · COST_FAILURE ·
COST_UNCERTAIN · UNDERPOWERED · NO_EDGE · OVERFIT · PARKED · SURVIVOR`.

Every failure row: `root_cause · evidence · affected_dataset ·
affected_formula · next_action`.

- `DATA_FAILURE` / `IMPLEMENTATION_FAILURE` → fix, rerun. **Do not kill the
  candidate.**
- `COST_FAILURE` → audit (§5), test lower-turnover / maker.
- `UNDERPOWERED` → add sample only if justified.
- `OVERFIT` → reject.
- `NO_EDGE` after all relevant validation → reject.

---

## 11. RISK (risk-officer VETO)

max drawdown · daily loss · position/leverage limits · correlation
concentration (**modelled to spike toward 1 in stress — diversification is
assumed to fail in a crash**) · liquidation risk · tail risk · P(breach) ·
P(ruin) · Monte Carlo / block bootstrap. `$25,000` account simulation for any
`READY_FOR_PAPER` candidate.

---

## 12. LIVE-READINESS CHECKLIST (S10, human)

data valid · no leakage · dynamic execution model · positive net expectancy ·
robust OOS · acceptable drawdown · regime stability · cost stress passed ·
slippage stress passed · no obvious overfit · paper/live parity acceptable ·
risk limits valid · kill switch valid. Then: small size, hard limits,
monitoring, kill switch, human sign-off. **No autonomous deployment of real
capital.**

---

## 13. COMPUTE

- **[FIX 5]** Python + `numba` (3-tier bit-identical kernels, already built).
  **Rust/C++ is deferred until a profiler proves a specific bottleneck** — the
  source files themselves say "optimise only proven bottlenecks". A system
  with no edge does not need a Rust rewrite.
- Parallel execution · cached/reused features (never compute a feature twice) ·
  vectorised · columnar (parquet) · incremental recompute · memory-map large
  tapes.
- Backend equivalence tested; a mismatch cannot be used for a research result.

---

## 14. PER-EXPERIMENT OUTPUT (ledger)

`experiment_id · hypothesis_id · formula · source/provenance · research_venue ·
production_venue · symbol · timeframe · target · horizon · features ·
execution_mode · gross_return · net_return · commission · spread_cost ·
slippage · impact · funding · turnover · IC · CV_IC · Sharpe · Sortino ·
ProfitFactor · MaxDrawdown · DSR(family) · PBO · placebo_p · surrogate_p · OOS ·
regime_stability · cost_sensitivity · status · root_cause · next_action`.

---

## 15. START ORDER

1. Build/verify S0 → green (already done in this repo).
2. S1: fill `venue.yaml` (Binance research + Bybit production) from official
   APIs; build the **per-trade cost model** and the **maker execution model**;
   rebuild F-1; enumerate the objective queue (§7); start live recorders for
   `DATA_UNAVAILABLE` streams.
3. S2 cheap screen → S3 → … → S7, per the objective queue, autonomously.
4. Report to the user ONLY at: `SURVIVOR` / `READY_FOR_PAPER` /
   `READY_FOR_LIVE_REVIEW` / whole queue bounded / genuine human decision.
   Everything negative stays in the ledger.

**Target: an edge that survives cost, execution, OOS and live reality —
not a good-looking backtest.**
