# REPRICING_RULE — frozen before any code changed (P0.1)

This rule is frozen. The re-score (P0.5) runs against exactly this specification.
Nothing below may be changed after re-score results are seen.

## 1. Scope of the change

**Only the cost number changes.** Everything else is byte-for-byte identical to
the RESEARCH ROUND 1 runs:

| kept identical | |
|---|---|
| signals / features | unchanged |
| parameters, grids | unchanged |
| universe (coins) | unchanged |
| date ranges, partitions | unchanged |
| execution convention | R9 unchanged (`pos[t] = signal[t-2]`) |
| trial count | unchanged — this is not a new trial |
| DSR denominators | unchanged |

The re-scored result is **accepted as-is, up or down**. No tuning, no
window-picking, no gate loosening follows the re-score.

## 2. Cost source of record — per coin, NOT pooled

Cost for every fill comes from `lab/exec/cost_model.py :: cost_bps(coin, side,
notional, ts, mode)`. Its calibration inputs, frozen:

| component | source (frozen) | fallback when coin absent |
|---|---|---|
| **fee** | Binance USDT-M perp VIP-0: taker 4.5 bps, maker 1.8 bps (post-BNB-discount schedule as of 2024). `mode` selects the leg. | none needed — same for all coins |
| **half-spread** | per-coin, first hit wins: `exec_calibration.json` → `spread_measured.json` → `spread_extra.json` → `aggtrades_micro.json` → `spread_estimates.realised_by_symbol` | liquidity tier (see §4) |
| **adverse selection** (maker) | per-coin entry+exit `adverse_selection_bps` from `exec_calibration.json` (5 coins) | `as_bps = max(0.75, 0.35 · half_spread_bps)` — the ROUND 1 floor, scaled by spread |
| **p_fill** (maker) | queue model §3, calibrated from `bookdepth_features.json` near-touch notional + per-coin trade arrival rate | tier default: tier-1 0.92, tier-2 0.85, tier-3 0.72 (from the 5 calibrated coins' entry/exit mean) |
| **impact** | square-root law §3, `σ_daily` and `ADV` from 1h klines (all 55 coins) | Y = 0.5 default coefficient |
| **residual slippage** | walk-the-book on `bookdepth` near-touch ladder for the portion of `notional` beyond L1 depth | 0 when `notional` ≤ near-touch depth |

**No per-coin parameter is fitted.** Half-spread, adverse-selection and depth are
*measured* per coin; `Y` (impact), the fee schedule, and the tier fallbacks are
**global constants**. There is exactly one impact coefficient `Y` for the whole
universe.

## 3. Formulas (frozen)

**Square-root impact** (Bouchaud / Almgren-Chriss):

    impact_bps = 1e4 · Y · σ_daily · sqrt( Q / ADV )
    Y = 0.5 ;  σ_daily = std of daily log returns, 30d rolling ;
    ADV = median daily quote-volume, 30d ;  Q = order notional (USD)

**Maker fill probability** (queue depletion, Poisson arrivals):

    Q_ahead  = near-touch same-side notional at order entry (bookdepth)
    λ        = opposing-aggressor notional arrival rate (aggTrades, per second)
    T        = time-in-force (entry 45s, exit 10s — the ROUND 1 calibration values)
    p_fill   = P( Poisson(λ·T) · avg_trade_notional > Q_ahead )
             ≈ 1 - Γ_cdf( Q_ahead / avg_trade_notional ; λ·T )
    p_fill is clipped to [0.05, 0.99]; higher realised σ over the prior hour
    shrinks p_fill multiplicatively by exp(-2·(σ_1h/σ_1h_median - 1)^+).

**Effective one-way cost:**

    maker_leg   = maker_fee + adverse_sel_bps - half_spread_bps    (you EARN the spread if filled passively)
    taker_leg   = taker_fee + half_spread_bps + impact_bps + residual_slippage_bps
    total_oneway_bps = p_fill · maker_leg + (1 - p_fill) · taker_leg
    total_oneway_bps = max(total_oneway_bps, taker_fee + 0.1)      # never cheaper than fee+ε

    mode="taker"  -> forces p_fill = 0
    mode="maker"/"hybrid" -> uses the queue p_fill

Round-trip = entry one-way + exit one-way (exit uses the 10s TIF → lower p_fill).

## 4. Liquidity tiers (fallback only)

Assigned once from 30d median quote-volume, frozen:

| tier | 30d median ADV (USD) | half-spread fallback | example |
|---|---|---|---|
| 1 | ≥ 300 M | 0.4 bps | BTC, ETH, SOL |
| 2 | 30 M – 300 M | 1.2 bps | mid alts |
| 3 | < 30 M | 4.0 bps | thin / delisted |

A coin that has a *measured* half-spread never uses the tier value for spread;
the tier only supplies `p_fill` fallback and is recorded in the fill row.

## 5. Reconciliation gate (P0.4)

`cost_bps("BTCUSDT","buy", notional=25_000, ts=<mid-sample>, mode="hybrid")`
round-trip must land within **15%** of `exec_calibration.json` BTC hybrid
round-trip (5.662 bps) → acceptance band **[4.81, 6.51] bps**.
If outside: STOP, fix the bug, do not proceed to re-score.

## 6. Ledger

Every fill writes one row: `coin, ts, side, notional, mode, fee_bps,
spread_bps, impact_bps, slippage_bps, adverse_sel_bps, p_fill, tier,
total_bps`. Stored per-fill; aggregation happens only when reading back.
