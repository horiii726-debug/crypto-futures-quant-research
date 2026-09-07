# REGIME_ATTRIB — oi_leverage_state (spec G)

_Not a trial. Existing P&L split by regime bucket (all states causal)._

**Overall:** 17544 bars, Sharpe 2.26, P&L 0.7386, maxDD -0.0704, turnover 0.0077

## btc_trend

| bucket | n_bar | frac | Sharpe(ann) | P&L | P&L share | maxDD | turnover |
|---|---|---|---|---|---|---|---|
| flat | 7680 | 0.44 | 2.91 | 0.4252 | 0.58 | -0.0675 | 0.0079 |
| up | 8952 | 0.51 | 1.78 | 0.2988 | 0.4 | -0.0704 | 0.0076 |
| down | 912 | 0.05 | 1.11 | 0.0146 | 0.02 | -0.0308 | 0.0067 |

→ **EVEN — P&L tracks time spent in each bucket; regime-robust on this dimension**

## vol_bucket

| bucket | n_bar | frac | Sharpe(ann) | P&L | P&L share | maxDD | turnover |
|---|---|---|---|---|---|---|---|
| mid | 6312 | 0.36 | 3.56 | 0.4446 | 0.6 | -0.0533 | 0.0085 |
| high | 5616 | 0.32 | 2.45 | 0.2833 | 0.38 | -0.0657 | 0.0074 |
| low | 5616 | 0.32 | 0.13 | 0.0107 | 0.01 | -0.0975 | 0.007 |

→ **WEAK in ['low'] (Sharpe there ~0) — edge needs the other regimes; still net-positive everywhere**

## fund_sign

| bucket | n_bar | frac | Sharpe(ann) | P&L | P&L share | maxDD | turnover |
|---|---|---|---|---|---|---|---|
| pos | 16096 | 0.92 | 2.22 | 0.6778 | 0.92 | -0.0899 | 0.0078 |
| neg | 1425 | 0.08 | 2.87 | 0.0608 | 0.08 | -0.0264 | 0.006 |
| na | 23 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

→ **EVEN — P&L tracks time spent in each bucket; regime-robust on this dimension**

