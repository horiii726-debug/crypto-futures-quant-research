# UNIVERSE_RULE — frozen before any run (spec B)

Pre-registered. Not changed after results are seen. Snooping = shifting any
threshold below to make a result pass.

## Admission test (in-sample only, rolling, no lookahead)

A coin is **in** the tradeable universe for a rebalance at time `t` iff, using
only data strictly before `t`:

1. `median_30d( half_spread_bps )  <=  1.0`
2. `median_30d( ADV_usd )          >=  50_000_000`   (50 M USD / day)
3. the coin has been listed **>= 90 days** as of the start of the 30-day window
   (i.e. >= 120 days of history before `t`)

All three must hold. A coin that fails any test is excluded for that rebalance
and re-checked at the next one (a coin can enter/leave over time — R10
survivorship still applies to anything it held while listed).

## Definitions (frozen)

| quantity | definition |
|---|---|
| `half_spread_bps` | **tick-level Roll (1984) effective half-spread** where measured (`aggtrades_micro.json` / `spread_measured.json` / `spread_extra.json`, 25 coins: `1e4·sqrt(max(0,-cov(Δp_t,Δp_{t-1})))/mid` on the trade tape). For a coin without a measured tape spread, a **liquidity-tier proxy from ADV**: tier-1 (30d ADV ≥ 300 M) → 0.4 bps, tier-2 (30–300 M) → 1.2 bps, tier-3 (< 30 M) → 4.0 bps. Binance bookDepth carries only the ±1%…±5% ladder (no touch), and Roll on 1-minute bars over-estimates the touch spread ~2× with a coin-dependent factor (0.5–5.0), so it is not used directly. |
| `ADV_usd` | median of daily summed `quote_volume` over the trailing 30 days |
| `listed` | first bar present in `data/processed/klines_1m/{coin}.parquet` |

## Impact coefficient

`Y = 0.5`, **one global value**, never fitted per coin. (Per-coin calibration is
allowed only if a coin has enough of its own large prints to fit `Y` with a
standard error < 0.15; none currently qualify, so all use 0.5.)

## Expected effect (informational, not a tuning target)

At these thresholds, thin / distressed names drop out (CRV half-spread ~8.7 bps,
HOOK ADV ~10 M, ZEC/XMR/DASH/DAR/AMB all sub-30 M) and the liquid majors stay
in (BTC, ETH, SOL, BNB, XRP, DOGE, LINK, AVAX, ADA, LTC, DOT, …). The exact
membership is whatever the frozen test returns — it is computed, not chosen.

## Where this rule is enforced

`lab/exec/universe.py :: eligible(coin, ts) -> bool` and
`tradeable_at(ts) -> list[str]`. Any strategy that trades a cross-section reads
its universe from here.
