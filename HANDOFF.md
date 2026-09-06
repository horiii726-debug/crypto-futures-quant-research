# HANDOFF — context for a fresh chat / new AI session

Read this + `SYSTEM_SPEC.md` + `CLAUDE.md` before doing anything. This file is
the honest state of the project: what exists, how it works, what was tried,
what was learned, what NOT to repeat, what to do next.

Repo: `/root/crypto-lab/` · git history: 6 commits, HEAD `260ff91`.

---

## 1. WHAT EXISTS (and works)

A research-to-S7 engine in Python (+ numba). All of S0 is green and stays
green.

### Engine (`lab/engine/`)
| file | what it does |
|---|---|
| `kernels.py` | numba 3-tier kernels (python/numpy/numba), **bit-identical, tolerance 0**. `verify_kernels()` must print IDENTICAL. |
| `labels.py` | triple-barrier labels, average uniqueness, overlap factor. |
| `cv.py` | purged K-fold + embargo + CPCV. `leakage_check()`. |
| `backtest.py` | portfolio, cost-aware, **R9 execution**: `pos[t]=signal[t-execution_lag-1]`, earns bar `t`. `_check_costs()` refuses to run while `venue.yaml slippage_bps` is null. `VenueNotConfigured`. |
| `runner.py` | enumerate a pre-registered grid, log every trial to the ledger BEFORE the result is known. |
| `signals.py` | spec→feature resolver with a module whitelist. |

### Ledger (`lab/ledger.py`)
Append-only SQLite. `UPDATE`/`DELETE` blocked by triggers on all result
tables. Tables: `sources hypotheses datasets trials experiments metrics
verdicts lessons test_looks bounds universe`.
`Ledger.trial_count(family=…)` is the ONLY source of the number DSR reads (R3).
Env: `XAU_LAB_LEDGER_SQLITE=./ledger/lab.sqlite`.

### Stats (`lab/stats/`)
`dsr.py` (Deflated Sharpe; trial count from ledger, refuses a manual
`n_trials` unless `allow_manual_trials`) · `pbo.py` (CSCV) · `mcs.py` (Model
Confidence Set + SPA + Diebold-Mariano; selftest: 200 iid models must NOT
collapse to one winner) · `power.py` (required_n / effective_n with overlap +
autocorr correction) · `surrogate.py` (block-bootstrap label-shuffle noise
threshold).

### Guards (`lab/guards/`) — wired as Claude Code hooks in `.claude/settings.json`
`test_seal.py` (blocks any Bash touching `data/test/`, exit 2) ·
`numeric_firewall.py` (blocks untagged numbers in `reports/`) ·
`unseal_test.py` (the ONLY path into test; one look per family; no `--force`;
logs to ledger; needs a human token) · `leak_canary.py` (future-perturbation
+ prefix-stability, tol 1e-10; + heuristic look-ahead scan) ·
`ledger_checkpoint.py` (dangling `[EXP-…]` detector).

### Data (`lab/data/`)
| file | what |
|---|---|
| `binance_vision.py` | `data.binance.vision` client: symbol discovery (incl. delisted), klines, aggTrades, funding, `index_klines` (premiumIndex/markPrice). `NotFound` on 403/404 (no retry). |
| `ingest.py` | 1m → quality gate → resample (1h/4h/1d) → chronological partitions → SHA-256 to ledger. Negative-month cache for delisted symbols. |
| `universe.py` | point-in-time universe with survivorship. |
| `spread.py` | Corwin-Schultz + Abdi-Ranaldo (OHLC estimators — **unreliable at 1h**, cross-check only) + `realised_spread_aggtrades` (Roll + quoted, from the real tape — **authoritative**). |
| `microstructure.py` | bookDepth + bookTicker readers + aggregate stats. |
| `f1.py`, `f1_hybrid.py` | F-1 feasibility: required IC/hit-rate per horizon, taker vs hybrid cost floor. |

### Execution model (`lab/exec/maker_model.py`) — **[the P1 deliverable]**
Limit-at-touch. `p_fill` and adverse selection **simulated on the real
aggTrades tape**; queue-ahead = 0.7 × real best-bid-qty (from bookTicker);
taker fallback after the wait (45 s entry / 10 s exit); opportunity/wait cost
charged. **No 100% maker-fill assumption.** `ADVERSE_SELECTION_FLOOR_BPS =
0.75`.

Calibrated numbers (`data/processed/exec_summary.json`): **universe hybrid
round-trip ≈ 8.5 bps mean / 7.6 bps median** vs taker ~10.8–17. By tier:
BTC 5.7, SOL 7.6, WLD 5.5, HOOK 9.6, DAR 14.3 bps.

### Research (`lab/research/`)
`panel.py` (Panel container + ~41 causal cross-sectional features across 12
divisions) · `study.py` (one hypothesis: signal→positions→t+1 backtest→purged
CV→placebo→surrogate→family-DSR→PBO; `EXEC_MODE` taker|hybrid) ·
`gates.py` (G3–G7 verdict logic) · `campaign.py` (per-family tournament with
pre-registered grids + autonomous escalation) · `main_s1_s7.py` (S1→S7
orchestrator) · `montecarlo.py` ($25k risk sim: block-bootstrap +
fixed-fraction + halve-after-drawdown + crash-correlation shock — **built,
never used, no candidate**).

### Corpus (`lab/rag/formulas.sqlite`)
**76 papers, 131 formulas, 18 divisions.** 63 ADMISSIBLE, 10 PARKED,
3 REJECT_DATA. By division: F_XSEC 23, F_XCOIN 17, F_VOL 16, F_FUND 12,
F_FLOW 12, F_DIR 8, F_STATE 6, F_TRANSFORM 5, F_SIZE/F_PATH/F_LIQ/F_DIST/
F_DEP/F_BOOK 4 each, F_EXIT 3, F_MULTI/F_ENTRY 2, F_FORMULA 1.

### Data on disk (`data/`, ~2.5 GB)
- `processed/klines_1m/` — 55 symbols, 1m, 2023-09…2025-08 (survivorship:
  20 delisted/migrated names retained: MATIC→POL, RNDR→RENDER, OM, WAVES,
  AGIX, DAR, KLAY, LINA, AMB, HOOK, LOOM, IDEX, KEY, MYRO, STORJ, COMBO, FTM,
  BOND, OCEAN, …).
- `processed/panels/prio50_2y/{1h,4h,1d}/` — aligned close/volume/etc.
- `data/{train,valid,test}/prio50_2y/` — **test is sealed, `test_looks`=0**.
- `processed/bookdepth_daily/` — 30 coins, 260 days (2023-09…2024-05, all the
  DOM history there is).
- aggTrades: **only sampled days cached** (~22 coins × ~14 days) — full tape
  NOT downloaded yet.

---

## 2. LEDGER STATE (do not reset — R3)

`trials 305 · experiments 132 · hypotheses 132 · verdicts 131 (130 FAIL,
1 UNDERPOWERED) · lessons 131 · bounds 44 · test_looks 0`.

The 305 trials span 13 families (F_XSEC 117, F_XCOIN 36, F_FLOW 32, F_FUND 30,
F_DIR 30, then F_TRANSFORM/F_VOL/F_STATE/F_PATH/F_MULTI/F_DEP/F_LIQ/F_BOOK).

---

## 3. WHAT WAS TESTED — and the RESULT (the bounds)

### 3.1 Approach used so far
**Cross-sectional, long/short decile, daily & 3-day rebalance, on 1h/1d/3d
OHLCV + funding panels, Binance USDT-M, 2 years, measured costs.** Full gate
ladder each config: purged CV + embargo, placebo (random ranking), surrogate
(block-shuffle), DSR (cumulative machine count), family verdict.

### 3.2 Result: **NO_SURVIVOR across 13 families / 305 trials. Test never opened.**

The recurring signature (this is the key finding):
1. A handful of features have a real in-sample IC ~0.03–0.05 and beat the
   placebo (there IS faint structure).
2. Their **cross-validated IC flips sign or collapses** (e.g. `xsec_low_vol`
   +0.051 in-sample → −0.025 out-of-fold). Overfit.
3. The few with a positive CV IC have a raw IC ~0.005–0.014 and **fail the
   surrogate test** (indistinguishable from block-shuffled noise).
4. **Best DSR anywhere = 0.35** (`transform_hurst`@3d), gate needs 0.95. The
   cumulative trial count makes the bar unreachable.
5. Net Sharpe after the measured cost floor: best ~+1.2, only on configs that
   fail (2)/(3)/(4).

### 3.3 F-1 feasibility (measured)
- Taker-only cost floor ~12–14 bps round-trip (2× taker fee = 10 bps
  dominates). Closes 5m / 15m / 1h / 4h / 1d.
- **Hybrid execution (~8.5 bps) reopens 1d** for turnover ≤ ~1.4
  (required IC 0.041 < 0.05 ceiling). 4h and shorter stay closed.
- Only 3d is feasible taker-only.

### 3.4 Data gaps found (hard constraints)
- `F_OI`, `F_LIQD` → **REJECT_DATA.** Binance publishes ~30 days of OI /
  long-short history (REST), bulk `metrics` files stop 2022, liquidation
  stream is real-time only. No free multi-year history exists.
- `F_BOOK`, `F_LIQ` (real DOM) → **UNDERPOWERED.** bookTicker ends ~2024-04,
  bookDepth ends ~2024-05. Only ~8 months. 260-day test: best |IC| 0.028,
  beats placebo (p=0.017), fails surrogate (p=0.21).
- `F_VOL / F_STATE / F_SIZE / F_DIST / F_EXIT / F_FILTER` → no standalone
  alpha to evaluate; they are overlays that need a passing base signal.

### 3.5 P4 (universe → top 100 / 3 yr): NOT pursued — reasoned bound.
F-1 math: cost floor unchanged, dispersion ~unchanged at scale → required IC
unchanged (~0.04–0.05) → same result, plus more trials = harsher DSR. The 2022
bear makes XS predictability net-of-cost worse, not better. Extending the
universe extends the bounds, it does not overturn them.

---

## 4. WHAT NOT TO REPEAT

1. **Do NOT re-run the 305 cross-sectional OHLCV configs.** They are in the
   ledger with verdicts and bounds. Re-running them just inflates the trial
   count against future candidates.
2. **Do NOT keep shotgunning cross-sectional price/volume features.** That
   family space is bounded (F_XSEC, F_XCOIN, F_DIR, F_MOMENTUM, F_REVERSAL on
   OHLCV alone). Best CV-stable IC found: ~0.014. The cost floor needs ~0.04+.
3. **Do NOT test M3/M5/M15 with taker execution.** F-1 says required IC
   0.2–0.8 there. It is closed. Only test those timeframes with the maker
   execution model AND a signal whose half-life > holding period.
4. **Do NOT try to backtest F_OI / F_LIQD / real-DOM on historical bulk
   data.** It does not exist. Start a live recorder instead.
5. **Do NOT loosen a gate, reset the trial count, or open `data/test/`
   outside `unseal_test.py`.**

---

## 5. WHAT TO DO NEXT (priority order — this is the open frontier)

The cross-sectional / OHLCV / bulk-data approach is bounded. Genuinely
unexplored, in order of expected value:

### 5.1 Tape microstructure at M3/M5/M15 with maker execution
The tape (`aggTrades`) has FULL history and was only used via a crude
`taker_buy_base` kline proxy. Build the real event-level engine:
- **OFI** (order-flow imbalance), **cumulative delta**, **signed volume**,
  **aggressor run length**, **large-trade imbalance**, **trade intensity /
  Hawkes**, **VPIN** (volume-bucketed), **Kyle's λ** from real signed volume,
  **effective/realised spread** per bar, **microprice proxy** from trade
  bounce, **trade-sign autocorrelation** (Lillo-Farmer).
- Aggregate causally to M3/M5/M15. Test time-series per asset first.
- Execute with `lab/exec/maker_model.py` (earn the spread, not pay it).
- Mechanism examples (each needs a written hypothesis): OFI persistence at
  seconds→minutes; aggressor-imbalance exhaustion → reversal; microprice
  tilt → next-bar drift.
- **Realistic expectation:** intraday OFI has |IC| 0.05–0.15 in the
  literature but a half-life of seconds→minutes. The edge is real; whether it
  survives *your* fill rate + adverse selection + latency is the open
  question. This is where the answer is, if there is one.

### 5.2 Delta-neutral funding / basis carry (a risk premium, not "alpha")
- `premiumIndexKlines` has full history → real perp-vs-index basis.
- Long spot + short perp when funding/basis positive; harvest funding,
  delta-neutral. Portfolio-level, low turnover, needs spot+perp execution.
- Only tested so far as a *cross-sectional signal*, never as a *harvest*.
- Historically Sharpe ~6 (2020-23), decayed, went negative parts of 2025 —
  test it honestly, it may be `NO_EDGE` now, but it is a distinct objective.

### 5.3 Cross-venue (Binance vs Bybit vs OKX vs Hyperliquid)
- Funding-rate divergence, basis dislocation, lead-lag. Persistent,
  capacity-limited. Needs a second venue's data feed.

### 5.4 Live recorders (start now, pay off in weeks)
- Record Binance/Bybit WS: `bookTicker`, partial `depth`, `forceOrder`
  (liquidations), `openInterest`, `markPrice`. Build the historical dataset
  for F_MICRO_BOOK / F_MICRO_QUEUE / F_MICRO_LIQ / F_OI that bulk data cannot
  provide.

### 5.5 Single-asset directional at 3d (feasible, lightly explored)
- 3d is F-1-feasible. Slow TS momentum / trend / carry blend on BTC/ETH.
  Modest Sharpe target (0.5–1.0). A small pre-registered study (3–5
  mechanisms, not 300) has a fair shot because the family DSR bar is
  achievable at low trial counts.

---

## 6. FIRST CONCRETE STEPS FOR THE NEW CHAT

1. Read `SYSTEM_SPEC.md` §2 ([FIX 1] multiplicity), §5 (cost model), §7
   (objective queue). Adopt the two-tier trial counting and per-family DSR —
   the single-global-DSR design is why nothing survives.
2. Fill `config/venue.yaml` for **both** Binance (research) and Bybit
   (production) from the official APIs. Nothing runs until it's non-null.
3. Build `lab/data/tape.py` — event-level aggTrades feature engine (§5.1).
   Download full aggTrades for ~10 liquid coins × ~12–15 months (~20 GB,
   disk has 63 GB free).
4. Build `lab/features/micro/` — OFI, VPIN, cumulative delta, Kyle λ,
   microprice, aggressor runs — all causal, all `assert_causal`-clean.
5. Rebuild F-1 for (M3/M5/M15 × maker-execution). Confirm which are feasible.
6. Run S3 on the feasible micro objectives, time-series mode, maker
   execution, small pre-registered batches. Full gate ladder.
7. In parallel: start the live WS recorders (§5.4).
8. Report to the user ONLY at SURVIVOR / READY_FOR_PAPER / whole-queue-bounded
   / genuine human decision.

---

## 7. HOW THE SYSTEM RUNS (quick reference)

```bash
cd /root/crypto-lab
export PYTHONPATH=. XAU_LAB_LEDGER_SQLITE=./ledger/lab.sqlite PYTHONHASHSEED=0

python3 tests/test_lab.py            # S0: 10/10 must pass (isolated ledger)
python3 lab/engine/kernels.py        # numba IDENTICAL
python3 -c "from lab.ledger import Ledger; print(Ledger().trial_count())"  # 305

# a research run is driven by lab/research/main_s1_s7.py (S1..S7);
# new work should extend campaign.HYPOTHESES + panel.REGISTRY, or add a
# dedicated study module like lab/research/_book_test.py.
```

Reports so far: `reports/FINAL_REPORT.md` (the full P1–P4 result),
`reports/F1_HYBRID.md`, `reports/S1_S7_REPORT.md`, `reports/S0_VERIFICATION.md`.

---

## 8. ONE-PARAGRAPH SUMMARY FOR THE NEW AI

> A full research-to-S7 quant engine exists and is verified (`/root/crypto-lab`,
> S0 green). It has tested **13 formula families across 305 machine-counted
> trials** on 2 years of real Binance USDT-M perp data with a **measured**
> per-trade cost model and a **tape-calibrated hybrid maker execution model**
> (~8.5 bps round-trip vs ~11–17 taker). Result: **NO_SURVIVOR** — the
> cross-sectional OHLCV+funding approach is **bounded** below the cost floor at
> every feasible horizon (only 1d taker-hybrid and 3d are feasible; the best
> CV-stable IC found was ~0.014 vs ~0.04 needed). The test partition was never
> opened. `F_OI`/`F_LIQD`/real-DOM are `DATA_UNAVAILABLE` (no free multi-year
> history — needs a live recorder). **The open frontier is: (1) event-level
> tape microstructure (OFI/VPIN/cumulative-delta/Kyle-λ/microprice) at
> M3/M5/M15 executed as a maker; (2) delta-neutral funding/basis carry;
> (3) cross-venue dislocations; (4) live recorders for the missing streams.**
> Do NOT repeat the 305 cross-sectional configs. Fix the single-global-DSR
> design first (`SYSTEM_SPEC.md` [FIX 1]) or nothing will ever survive.
