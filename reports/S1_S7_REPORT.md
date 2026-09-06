# Crypto Quant Factory — S1→S7 autonomous run (Option A)

**Result: ZERO systems cleared the gate ladder. Reported in full with bounds — a valid outcome.**

All figures below are machine-computed and stored in the append-only ledger
(`ledger/lab.sqlite`). Every research metric traces to a ledger experiment
`[EXP-xxxxxx]`; the aggregates here are read back from `metrics`, `verdicts`,
`bounds`. The test partition was **never unsealed** — `test_looks` count is 0.

Scope executed: S1 full · S2 limited (50 papers / 66 formulas, 4 families) ·
S3–S6 full pipeline for F_XSEC, F_FUND, F_FLOW, F_XCOIN · S7 (no survivor → no
test look). Stopped before S8 as instructed.

---

## S1 — DATA

| item | value |
|---|---|
| source | data.binance.vision bulk files + fapi.binance.com spec (hashed to ledger `sources`) |
| universe | 55 USDT-M perps, tag `prio50_2y` — **survivorship intact**: 20 names delisted/migrated during the window are retained (MATICUSDT→POL ends 2024-09, RNDRUSDT→RENDER ends 2024-07, OMUSDT, WAVESUSDT, AGIXUSDT, DARUSDT, KLAYUSDT, LINAUSDT, AMBUSDT, HOOKUSDT, LOOMUSDT, IDEXUSDT, KEYUSDT, MYROUSDT, STORJUSDT, COMBOUSDT, FTMUSDT, BONDUSDT, OCEANUSDT, …) |
| history | 2023-09-01 … 2025-08-31 (24 months), 1-minute klines, resampled to 1h/4h/1d |
| full ever-listed set | 1028 symbols enumerated (874 USDT-M); 216 USDT-M perps fully delisted before the window — the survivorship gap that biases naïve studies |
| quality gate | 55/55 kept. BTCUSDT: 0 missing minutes over 1,052,640 bars, 0 OHLC violations. Delisting shows as a clean series end, not "missing data". |
| partitions (chronological, hashed to `datasets`) | train 2023-09-01…2025-02-28 · valid 2025-03-01…2025-05-31 · **test 2025-06-01…2025-08-31 (sealed)** |
| train:1h sha256 | `bd19f304da0c109f…` (13,128 bars × 55) |
| valid:1h sha256 | `2b300b1436a0ff6e…` (2,208 × 55) |
| test:1h sha256 | `d554f89a61a8a552…` (2,208 × 55) — sealed, hash recorded, never read |

### Measured transaction costs (not assumed)

Realised **half-spread from aggTrades** (22 symbols, trade-by-trade, spanning
the liquidity range): p25 **0.55 bps**, median **0.75 bps**, p60 **0.85 bps**,
p75 **1.13 bps**, max (clean) **2.10 bps** (one estimator blow-up >5 bps
dropped). OHLC estimators (Corwin-Schultz, Abdi-Ranaldo) were computed as a
cross-check but are unreliable on 1h crypto bars (range is volatility-dominated)
— **not used** to set the number.

`config/venue.yaml`: `slippage_bps` = **1.35** (= p60 half-spread 0.85 + 0.5 bp
impact). `maker_fee` 0.0002 / `taker_fee` 0.0005 (VIP 0, official). Funding cap
±2%, mean |funding| **1.17 bps** per interval (20 symbols).

### F-1 feasibility (measured costs) — `reports/F1_CRYPTO.md`

Round-trip cost floor is **~12–14 bps**, dominated by 2× taker fee (10 bps).
Required IC = turnover (1.6×) × cost_floor / cross-sectional dispersion.

| horizon | cost floor (bps) | dispersion D (bps) | required IC | required hit-rate | FEASIBLE |
|---|---|---|---|---|---|
| 5m  | 12.2 | 17 | 1.12 | 0.9997 | **NO — closed** |
| 15m | 12.2 | 30 | 0.65 | 0.789 | **NO — closed** |
| 1h  | 12.3 | 60 | 0.33 | 0.645 | **NO — closed** |
| 4h  | 12.5 | 123 | 0.16 | 0.575 | **NO — closed** |
| 1d  | 14.0 | 313 | 0.071 | 0.531 | **NO — closed** |
| 3d  | 17.5 | 567 | **0.049** | 0.521 | **yes (marginal)** |

**Only the 3-day horizon is feasible**, and only barely (required IC 0.049 vs a
generous 0.05 admissible ceiling). 5m–1d are **closed to research** for this
universe/period — bounds written to the ledger (`bounds`, family `ALL`).

Cause: taker-only execution. A maker-only or hybrid model would roughly halve
the fee term and reopen 1d (and marginally 4h) — that is the single highest-
leverage change to this result, and it needs an execution model + queue
simulation, which is S9 work.

---

## S2 — FORMULA CORPUS (limited)

`lab/rag/formulas.sqlite`: **50 papers**, **66 formulas**, 4 priority families.
40 ADMISSIBLE, 9 PARKED (need aggTrades/depth), 1 REJECT_DATA. Sources: arXiv,
SSRN, ScienceDirect, RePEc — identifiers recorded; scan depth = abstract for
search-surfaced, method for textbook-standard forms; falsification notes carried
(e.g. crypto momentum decayed post-2021; carry Sharpe went negative in 2025;
short-term reversal is largely a bid-ask-bounce artifact).

| family | papers | formulas | of which computable now |
|---|---|---|---|
| F_XSEC | 17 | 23 | 23 |
| F_FUND | ~10 | 12 | 12 |
| F_FLOW | ~11 | 12 | 4 (rest PARKED for true tape/VPIN) |
| F_XCOIN | ~12 | 17 | 17 |

---

## S3–S6 — ALPHA RESEARCH (cross-sectional primary)

107 pre-registered configs run through the full ladder on train+valid
(purged 6-fold CV + embargo). Every fit is a ledger trial; **DSR reads the
cumulative count (107) — never reset.** Placebo = random cross-sectional
ranking. Surrogate = block-shuffled forward returns.

**Every config FAILED.** The recurring signature — no config satisfies more
than two of the five requirements at once:

1. **A few features have real in-sample IC ~0.03–0.05 and beat the placebo**
   (placebo_p ≤ 0.05): `xsec_low_vol` |IC| 0.051, `xsec_idio_vol` 0.052,
   `xsec_st_reversal` 0.043, `xsec_max` 0.038, `flow_imb_momentum`@3d 0.032,
   `xcoin_peer_return`@3d 0.038, `fund_carry`@3d 0.029.
2. **…but their cross-validated IC collapses or flips sign** — e.g.
   `xsec_low_vol` +0.051 in-sample → **−0.025** out-of-fold; `xsec_idio_vol`
   +0.052 → **−0.030**; `xcoin_peer_return` −0.038 → −0.067. Classic overfit:
   the full-sample IC is a fluke.
3. **The few with a positive CV IC have a raw IC too small to matter** (0.005–
   0.014) and **fail the surrogate test** (surrogate_p 0.5–0.9 →
   indistinguishable from block-shuffled noise): `xsec_momentum`@3d cvIC +0.10 /
   raw IC 0.014 / surr_p 0.51; `xcoin_pca_residual`@3d cvIC +0.13 / raw IC
   0.008 / surr_p 0.50.
4. **DSR against the cumulative machine trial count is decisive.** No config
   anywhere reaches the 0.95 gate. The single highest DSR (0.55,
   `xsec_momentum`@1d, scored early when the cumulative count was ~6) belongs to
   a config whose |IC| is 0.010 — it fails the economic-significance floor. By
   the time the identical mechanism is re-tested at higher trial counts its DSR
   is ~0.2.
5. Net annualised Sharpe after the measured 12–17 bps cost floor: best ~+1.2
   (`xsec_residual_momentum`@3d), but that config fails (3) and (4).

Nothing simultaneously satisfies: economic IC ≥ 0.03 **and** stable positive
CV IC **and** clears the surrogate noise ceiling **and** DSR ≥ 0.95 **and**
net Sharpe ≥ family floor.

### Per-family verdict

| family | configs | best in-sample \|IC\| | best CV-stable config | best net Sharpe (ann) | best DSR | verdict |
|---|---|---|---|---|---|---|
| **F_XSEC**  | 58 | 0.052 (`idio_vol`@3d — CV −0.04) | `momentum`@3d cvIC +0.10 / rawIC 0.014 | +1.24 (`residual_mom`@3d) | 0.55 (fails IC floor) | **FAIL / bounded** |
| **F_FUND**  | 15 | 0.029 (`fund_carry`@3d) | `fund_momentum`@1d cvIC +0.05 / rawIC 0.009 | +1.06 | 0.24 | **FAIL / bounded** |
| **F_FLOW**  | 16 | 0.032 (`flow_imb_momentum`@3d) | `flow_imb_momentum` cvIC +0.04 / rawIC 0.011 | +1.12 | 0.24 | **FAIL / bounded** |
| **F_XCOIN** | 18 | 0.038 (`peer_return`@3d — CV −0.07) | `pca_residual`@3d cvIC +0.13 / rawIC 0.008 | +0.93 | 0.10 | **FAIL / bounded** |

Evidence: ledger experiments `EXP-*` for each family (`experiments` table,
`hyp_id` linked); every config's metrics in `metrics`; every mechanism's
verdict in `verdicts` (48 FAIL) with a `lessons` root-cause row (48).

Placebo: the *directional* signals (momentum, reversal, imbalance-momentum,
peer-return) mostly **do** beat random ranking (placebo_p 0.007–0.05) — there is
faint structure. It is just far too small to trade through Binance taker costs
at the only feasible horizon. Every FAIL has a lesson row with root cause in
`lessons` (48 lessons); every family has a `bounds` row.

### Escalation ladder (autonomous)

Executed within each family without asking: L2 (horizon change) — every feature
tested at both 1d and 3d; L3 (re-spec) — pre-registered parameter grids;
L4 (new mechanism) — 5–11 distinct mechanisms per family from the corpus.
L5/L6 reached: targets and mappings exhausted for the admissible feature set →
**BOUND written, family closed**.

---

## S7 — OUT-OF-SAMPLE

**No family produced a survivor, so no test look was taken.** The sealed test
partition (`test:*` hashes in `datasets`) is untouched — `test_looks` = 0. Each
family retains its single permanent look for a future, better-motivated
candidate.

---

## RISK / $25,000 simulation

**Not applicable — nothing reached paper-trading candidacy.** The Monte Carlo
engine (`lab/research/montecarlo.py`: block-bootstrap + fixed-fraction +
halve-after-drawdown + explicit crash-correlation shock) is built and unit-
tested, ready for the first candidate that clears S7. Applied to the *least-bad*
config (which still fails the gates), it returns P(ruin) > 0 and negative median
terminal equity — i.e. it correctly says "do not fund this."

---

## BOUNDS (the actual result)

Written to `ledger` `bounds`:

- **Horizons 5m / 15m / 1h / 4h / 1d — CLOSED.** Required IC to overcome the
  measured Binance USDT-M taker cost floor (0.071 at 1d, rising to 1.12 at 5m)
  exceeds the 0.05 ceiling on plausible admissible cross-sectional crypto IC.
  Taker-only execution is the binding constraint.
- **F_XSEC** — 58 configs, 11 mechanisms, 2 horizons: best in-sample |IC| ≤ 0.057,
  best CV-stable IC ≤ 0.10 with raw IC ≤ 0.014, best DSR ≤ 0.21 vs 107 trials,
  no net-positive Sharpe after cost that also passes multiplicity. Cross-
  sectional price/volume predictability on this universe/period is bounded
  below the 3-day cost floor.
- **F_FUND** — 8 mechanisms: best |IC| ≤ 0.029 (funding carry at 3d), does not
  clear surrogate noise or DSR. Funding-based cross-sectional selection adds
  nothing tradeable here (consistent with the literature's 2024–25 carry
  decay).
- **F_FLOW** — bar-level aggressor proxy only (true VPIN/tape PARKED): best
  |IC| ≤ 0.032, fails CV/surrogate/DSR. No evidence that kline-derived order-
  flow imbalance is tradeable at feasible horizons.
- **F_XCOIN** — 5 mechanisms (BTC lead-lag, peer strength, spillover momentum,
  PCA-residual reversal, dispersion switch): best |IC| ≤ 0.044, best DSR ≤ 0.11.
  Lead-lag / spillover structure is real but sub-cost at 3d.

## What would move this

1. **Maker/hybrid execution model** (S9) — halving the fee term reopens 1d and
   marginally 4h; this is the highest-value next step.
2. **Depth data (Tahap 4)** for genuine F_BOOK / F_LIQ / true F_FLOW — a
   different information set, currently PARKED.
3. **Wider universe + longer history** (Tahap 1c: 100 coins, 3 yr) — more
   cross-sectional breadth raises effective N and dispersion.
4. **Meta-labeling** is not applicable yet: it requires a primary signal that
   already passes on its own, and none did.
