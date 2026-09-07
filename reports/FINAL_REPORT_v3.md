# Crypto Perp Quant Factory — terminal report (v3, supersedes v2)

**Stop condition B reached: every objective in the enumerated queue is bounded.**
**One genuine human-decision item survives review (F_OI · `oi_leverage_state`).**
**Test partition never opened — `test_looks = 0` — across 745 machine-counted trials.**

All numbers below are computed by the machine and stored in the append-only
`ledger/lab.sqlite` (UPDATE/DELETE blocked by trigger). Ledger state:

| | |
|---|---|
| hypotheses | 189 |
| experiments | 189 |
| full trials (DSR inputs, family-scoped) | 745 |
| cheap screens (`counts_as_trial=0`) | 432 |
| verdicts | 160 (157 FAIL · 1 PASS→VETOed · 1 UNDERPOWERED · 1 red-team VETO) |
| lessons written | 157 |
| bounds (distinct family × objective) | 33 |
| **test-set looks** | **0** |
| S0 self-tests | 10/10 green · calibration PASS (control rejected, planted edge caught) |

---

## 1. Bottom line

Across **20 signal families** and **745 pre-registered trials** on 2 years of real
Binance USDT-M + Bybit data with **measured** (not guessed) execution cost, **no
strategy clears the full ladder** (purged CV + embargo → CPCV/PBO → family-scoped
Deflated Sharpe → block-bootstrap surrogate → placebo → cost stress → red team).

The search is **not** "we didn't look hard enough". The result is a **structural
bound**: at every horizon we can actually trade, the cost floor exceeds the
information the data contains.

| research horizon | measured taker cost floor | required \|IC\| to break even | best \|IC\| ever observed (in-sample) |
|---|---|---|---|
| 5 min  | 12.2 bps | 1.12  | 0.057 |
| 15 min | 12.2 bps | 0.65  | 0.057 |
| 1 h    | 12.3 bps | 0.33  | 0.057 |
| 4 h    | 12.5 bps | 0.16  | 0.057 |
| 1 d    | 14.0 bps | 0.071 | 0.057 |
| 3 d (hybrid maker) | 14.3 bps | **0.040** | 0.057 |

The only feasible cell is **≥1-day holding with hybrid maker execution**, and even
there the best *cross-validated* \|IC\| (~0.014) is a third of what is required.

The two things that looked positive at first pass both failed review:

- **F_XVENUE** (Binance–Bybit funding spread): real inefficiency, gross ~14 %/yr,
  but the OOS-stable P&L is **~40 % one delisted coin (IDEXUSDT)** and lives
  entirely in names with distressed liquidity. **Red-team VETO.**
- **F_OI · `oi_leverage_state`**: passed placebo, surrogate, 6/6 CV folds, 4/4
  walk-forward windows and a held-out 30 % slice (see §3). Failed the
  pre-registered gate on rank-IC (0.004 < 0.010) and family DSR (0.72 < 0.95).
  **This is the one item that is a judgement call, not a refutation.**

---

## 2. Every family, bounded

### 2.1 Cross-sectional price / volume / funding — 13 families, 305 trials (prior runs, re-confirmed)
`F_XSEC F_XCOIN F_FUND F_FLOW F_DIR F_TRANSFORM F_DEP F_PATH F_MULTI F_VOL F_LIQ F_STATE` and daily `F_BOOK`.
Best in-sample \|IC(1-bar)\| ≤ 0.057, best net annualised Sharpe ≤ 1.3, best DSR ≤ 0.35 against 100–200+ cumulative trials. Cross-sectional predictability at feasible horizons is below the cost floor for this universe/period.

### 2.2 F_CARRY — delta-neutral funding carry
55 coins, 2 y, real premium-index basis, hysteresis, 2-leg cost + hedge tracking error, 1.25× capital.
40 configs. The ~10 %/yr funding is a **short-perp risk premium, not alpha**: surrogate p = 1.00 (a static short basket does as well). Net Sharpe ≤ 2.55 maker / ≤ 0.60 taker, decaying. A human may harvest it as a hedged beta allocation — that is a discretionary position, not a system.

### 2.3 F_MICRO_OFI / F_MICRO_BOOK — time-series tape microstructure (M5/M15, maker exec)
10 liquid coins, OFI / cum-delta / VPIN / Kyle-λ / microprice-from-VWAP / big-trade imbalance. **No config passed even the cheap screen** (rough \|IC\| ≥ 0.008 + net-positive per-bar + turnover < 2).

### 2.4 F_MICRO_OFI_EVENT — event-level OFI on the aggTrades tape (10 s–5 min)
BTC/SOL/DOGE, 28 configs total. Best gross OFI edge ≤ 0.001 bps/bar, gross Sharpe negative at every window, best maker net Sharpe −39. **OFI from *aggregate* trades carries no tradeable directional edge at any accessible timescale.** Real OFI/queue work needs a live L2 book feed (we have none).

### 2.5 F_OI — Bybit 2-year hourly open interest, cross-sectional (8 mechanisms)
`oi_delta_rev/mom`, `oi_price_confirm`, `oi_short_cover`, `oi_extreme_rev`, `oi_per_volume`, `oi_accel_rev`, `oi_leverage_state`. 45+ configs (105 F_OI trials logged). Best \|IC(1h)\| ≤ 0.013, best maker net Sharpe ≤ 1.18, best family DSR ≤ 0.19. **Open-interest dynamics add no cross-sectional edge over price/volume after cost** — *except* `oi_leverage_state`, escalated to a hardened study (§3).

### 2.6 F_BOOK — cross-sectional DOM depth signals (bookDepth ±1 %/±5 % bands, 5 m/15 m)
12 coins, 2023-09 → 2024-05 (all the bookDepth history that exists), 7 mechanisms, **120 configs**.
Best \|IC(1-bar)\| ≤ 0.024, best **gross** Sharpe ≤ **4.3** (`imbvol_fade`, `slope_amplify`, surrogate p = 0.012 — a *real* 5-minute signal), best **maker net** Sharpe ≤ **−5.9**. The intraday book-imbalance edge is real and completely inside the spread: turnover destroys 20–100× the gross edge. Tradeable book microstructure is at the touch / sub-second and needs full L2.

### 2.7 F_LIQD — forced-liquidation timing of BTC/ETH (coin-M liquidationSnapshot)
BTCUSD_PERP + ETHUSD_PERP hourly, 2023-09 → 2024-09, 5 signal forms, 60 configs. Best gross Sharpe 0.89, best maker net 0.14, surrogate p ≥ 0.05. Binance publishes coin-M `liquidationSnapshot` **only for these two correlated instruments**, so the effective sample is one BTC/ETH stress series → **UNDERPOWERED and bounded**.

### 2.8 F_XVENUE — Binance–Bybit funding-spread pair
55 coins, 2 y, maker & taker. The spread **is** a real mean-reverting inefficiency (surrogate p = 0.01, autocorr 0.53), gross ~14 %/yr. Bounded as a strategy: (a) for liquid coins < 1 bp/8h, net-negative after 4-leg cost; (b) the only OOS-stable config concentrates in illiquid/delisted small-caps (IDEXUSDT ≈ 40 % of gross P&L) where maker fills are not achievable; (c) taker cost-stress Sharpe +0.22; (d) capacity ~low-five-figures USD. **RED-TEAM VETO.**

---

## 3. The one open decision — F_OI · `oi_leverage_state`

**Signal.** `sig = −zscore_xsec( log( OI · price / rolling_mean_168h(quote_volume) ) )`
— i.e. **short the coins whose open interest is large relative to their traded
volume, long the coins where it is small.** Mechanism: high OI/volume = crowded /
high effective leverage = fragile to funding and liquidation cascades →
underperforms.

**Hardened validation** (`lab/research/oi_leverage_hardened.py`), liquid-28
universe, config frozen on the first 70 % of the panel, last 30 % looked at once:

| metric | selection window (first 70 %) | **held-out (last 30 %)** | full 2 y |
|---|---|---|---|
| maker net Sharpe (ann) | +2.59 | **+1.25** | +2.27 |
| taker net Sharpe (ann) | +2.53 | **+1.19** | +2.21 |
| gross Sharpe (ann) | +2.77 | +1.43 | +2.45 |
| turnover / bar | 0.008 | 0.007 | 0.008 |

- Walk-forward Sharpe by quarter: **[3.07, 2.30, 2.05, 1.54]** — all positive, decaying.
- Purged 6-fold CV: **6/6 folds positive**, mean fold Sharpe +2.25.
- Placebo (per-coin sign flip, 300×): **p = 0.003**.
- Block-shuffle surrogate (300×): **p = 0.020**.
- Monthly Sharpe: median +1.5, 75 % of months positive, worst month −3.6.
- Per-coin P&L: spread across 28 names (biggest: SUI +0.18, NEAR +0.17, SOL +0.10;
  LDO −0.14, LTC −0.08). **Not a one-coin artifact.**

**Why it still fails the pre-registered ladder**

1. **Rank-IC 0.004 < 0.010 gate.** The relationship is a monotone *tercile* spread
   (top quintile +0.35 %/24h vs bottom +0.11 %/24h, spread t ≈ 3.7 after
   deflating for 24 h overlap), not a smooth full-panel rank-IC. The gate metric
   and the signal shape are mismatched — arguable, but the gate was pre-registered
   and is not being moved.
2. **Family DSR 0.72 < 0.95.** Partly my fault: F_OI was run three times during
   this session (105 trials); a clean ~50-trial count would give DSR ≈ 0.85 — still
   short of 0.95.
3. **Single regime.** 2 years is all the hourly OI history Bybit has. Position
   autocorrelation is 0.95 / 0.71 / 0.54 / 0.36 at 1d / 1w / 1m / 3m — a **slow,
   near-static long/short book**. Only 0.27 correlated with a 90-day price-momentum
   book, so it is *not* simply disguised momentum, but the decaying walk-forward
   (3.07 → 1.54) is consistent with a positioning regime that is weakening.

**The decision (needs you):**

- **(a) Freeze F_OI.** Treat `oi_leverage_state` as a sub-threshold curiosity in a
  single regime. Consistent with the constitution as written.
- **(b) Promote it to a forward paper test.** It is the strongest lead in a
  20-family / 745-trial search, survives every robustness check except two
  technical gates, works after *taker* cost, has trivial turnover and ~low-seven-
  figure capacity. Run it live-parity on a paper account for 60–90 days
  (calendar-time, per R8/S8); if the held-out +1.2 Sharpe holds, escalate to a
  small real allocation.
- **(c) Rebuild the count and re-gate.** Rebuild a clean F_OI ledger (~50 trials,
  no reruns), re-run only `oi_leverage_state`, and if DSR clears ~0.9 accept it as
  a marginal SURVIVOR into S8.

My recommendation: **(b)** — it costs only time and it is the single result in
this whole project worth putting real forward data behind.

---

## 4. What is built and proven (engine is live-ready; no strategy is)

- **Data**: `data.binance.vision` bulk client (klines, aggTrades, bookDepth,
  bookTicker, premiumIndex, markPrice — free, no key) + Bybit API (funding 2 y, OI
  2 y hourly, coin-M liquidations). Rust hot-path kernels (`rustkernels/`, rayon)
  parse tape → 16-col micro bars and bookDepth → 8-col DOM bars ~20–30× faster
  than Python. Mass download → Rust process → delete-raw pipeline keeps disk flat.
- **Universe**: 55 coins incl. 20 delisted/migrated names retained for every
  period they traded (R10 survivorship).
- **Execution model**: `lab/exec/maker_model.py` — limit-at-touch with `p_fill` +
  adverse selection **simulated on the real aggTrades tape**, taker fallback, no
  100 %-fill assumption. Hybrid round-trip **8.5 bps** measured, not guessed.
- **Ladder**: purged K-fold + embargo + CPCV; PBO/CSCV; MCS + SPA + DM;
  block-bootstrap surrogate; placebo; **family-scoped Deflated Sharpe** with the
  trial count read *by the machine* from the ledger (R3).
- **Integrity**: append-only ledger, `data/test/` sealed (one look per family,
  no `--force`), R9 off-by-one execution convention (`pos[t] = signal[t−2]`)
  enforced by a bit-identical 3-tier kernel test, numeric firewall + leak canary
  as commit hooks.
- **S8–S10** (paper trading, execution testing, human sign-off) require a live
  account and calendar time **by design** (R8). The engine stops at
  `READY_FOR_PAPER`.

## 5. If you want to keep searching — the honest options

1. **A live L2 order-book recorder.** Every microstructure bound above says "real
   edge exists at the touch, we can't see it". Recording full depth for 20 coins
   for 3 months is the single highest-value next data investment.
2. **Longer OI history** to break the F_OI single-regime problem — needs a paid
   vendor (Laevitas, Amberdata) or starting a recorder now.
3. **Accept lower Sharpe.** F_CARRY as a hedged beta position (~0.6 net taker
   Sharpe, positive-carry, known risks) is a *real* allocation if the mandate
   allows ~1 Sharpe rather than requiring pure alpha.
4. **Options / basis term-structure** — not in the current data scope.
