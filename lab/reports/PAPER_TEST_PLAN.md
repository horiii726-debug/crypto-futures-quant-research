# PAPER_TEST_PLAN — oi_leverage_state (RESEARCH ROUND 2 · P1)

Status: **harness built, NOT armed.** R8 requires a human sign-off before any
order (paper, testnet, or live) is placed. This document is the checklist that
sign-off unlocks.

## What is being tested

| | |
|---|---|
| strategy | `oi_leverage_state`, frozen config `n=168, q=0.30, hold=24h` |
| universe | liquid-28 (`lab.research.oi_leverage_hardened.LIQUID`) |
| allocation | $25,000, gross leverage 1.0, market-neutral |
| rebalance | hourly, 2-bar execution delay (R9) |
| execution | hybrid: post-only limit at touch, taker fallback after TIF |
| venue | Binance USDT-M perp (price/exec) + Bybit OI (signal input) |

## Why (both must hold to call it a success)

1. **3rd regime.** Bybit hourly OI is only ~2 years — `oi_leverage_state`'s single
   biggest weakness. 60–90 calendar days of forward data is a genuinely
   out-of-sample regime the backtest never saw.
2. **cost-model validation.** Every order logs `predicted_cost_bps` (from
   `lab/exec/cost_model.py`) next to `realized_cost_bps`. If the model is
   systematically off by > 2 bps the RESCORE_305 conclusions need revisiting.

## Per-order log (`ledger/paper_orders.jsonl`)

`ts, coin, side, target_w, delta_w, notional, intended_price, actual_fill,
predicted_cost_bps, realized_cost_bps, p_fill_predicted, filled, latency_ms, mode`

## Run

```
# dry-run (no keys, no orders) — verify the book each hour
python3 lab/live/paper_trader.py

# armed (Binance testnet or a paper account) — AFTER sign-off
BINANCE_TRADE_KEY=... BINANCE_TRADE_SECRET=... \
  python3 lab/live/paper_trader.py --arm --mode hybrid
# schedule hourly via cron / systemd timer

# fortnightly review
python3 lab/live/paper_trader.py --review
```

## Review cadence — every 2 weeks, no intervention between

| check | pass condition |
|---|---|
| realised net Sharpe (annualised, so far) | ≥ 0 by week 4, ≥ +0.8 by week 12 |
| cost-model error | \|mean(realized − predicted)\| ≤ 2 bps |
| fill rate (hybrid) | ≥ 0.80 |
| turnover | ≤ 0.01 / bar (matches backtest 0.007) |
| max drawdown | ≤ 12% (backtest worst month −3.6%) |

## Decision after 90 days

- held-out backtest was maker +1.25 / taker +1.19. If paper lands **≥ +0.8**
  net and the cost model held → escalate to a small **real** allocation
  ($10–25k), keep logging.
- If paper is **< 0** or the cost model was off by > 3 bps → **freeze F_OI**,
  write the lesson, done.
- Anything between → extend the paper test another 90 days.

## Sign-off

- [ ] risk-officer: allocation, leverage, drawdown limit approved
- [ ] execution-analyst: hybrid fill assumptions match the venue's real book
- [ ] human (R8): explicit go
