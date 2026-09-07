# OI universe re-score (spec H3)

_Not a trial. oi_leverage_state, same signal/params/dates. Coin set: static liquid-28 vs frozen `universe.tradeable_at()` (rolling)._

| variant | gross SR | maker net SR | taker net SR | turnover | avg names |
|---|---|---|---|---|---|
| static_28 | +2.45 | +2.26 | +1.96 | 0.0077 | 19.4 |
| dynamic_universe | +1.16 | +1.04 | +0.88 | 0.0070 | 10.1 |
| held_out_static | +1.43 | +1.22 | +0.90 | 0.0066 | 18.8 |
| held_out_dynamic | +1.25 | +1.11 | +0.91 | 0.0073 | 12.1 |

**Δ net Sharpe (dynamic − static):**
- full sample : maker -1.21, taker -1.08
- held-out 30% : maker -0.11, taker +0.01

