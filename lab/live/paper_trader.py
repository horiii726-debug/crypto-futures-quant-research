"""RESEARCH ROUND 2 · P1 — paper-trading harness for oi_leverage_state.

Calendar-time, small allocation. Does NOT go live on its own: R8 requires a human
sign-off, so this only runs when explicitly started with `--arm` and real keys.
Until then `dry_run` just computes and logs the intended book.

Two purposes (per the brief):
  1. gather an out-of-sample 3rd regime — Bybit hourly OI is only 2 years, the
     single-regime risk is oi_leverage_state's main weakness;
  2. validate lab/exec/cost_model.py against real fills.

Per-order log (JSONL, ledger/paper_orders.jsonl): ts, coin, side, target_w,
notional, intended_price, actual_fill, predicted_cost_bps, realized_cost_bps,
p_fill_predicted, filled, latency_ms, mode.

Review every 2 weeks. No intervention, no tuning between reviews.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "ledger" / "paper_orders.jsonl"
STATE = ROOT / "ledger" / "paper_state.json"

from lab.exec.cost_model import cost_bps                       # noqa: E402
from lab.research.oi_leverage_hardened import LIQUID, _lev_signal, _positions  # noqa: E402

# frozen strategy config = the hardened best (n=168, q=0.3, hold=24h)
CFG = dict(n=168, q=0.30, hold_h=24)
AUM_USD = 25_000.0            # small allocation, matches the montecarlo account
TARGET_GROSS_LEV = 1.0
REBALANCE_H = 1


def _live_oi_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pull the last ~200h of hourly close + quote-vol (Binance) and OI (Bybit)
    for the liquid universe, live. Falls back to the cached research panel when
    no keys are configured (dry-run)."""
    from lab.venue import keys
    if not keys.have("bybit", "read"):
        from lab.research.oi_study import _panel
        c, q, o = _panel("1h")
        cols = [x for x in LIQUID if x in c.columns]
        return c[cols].tail(400), q[cols].tail(400), o[cols].tail(400)
    # live path (implemented when keys are present)
    raise NotImplementedError("live OI/price pull — wire to keys.binance_signed / keys.bybit_signed")


def target_book() -> pd.Series:
    c, q, o = _live_oi_panel()
    sig = _lev_signal(c, q, o, n=CFG["n"])
    sig = sig.where(c.notna().sum(axis=1) >= 8)
    pos = _positions(sig, q=CFG["q"])
    pos = pos.rolling(CFG["hold_h"], min_periods=1).mean()
    w = pos.iloc[-1].dropna()
    w = w / w.abs().sum() * TARGET_GROSS_LEV if w.abs().sum() else w
    return w


def _record(row: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


def step(*, arm: bool, mode: str = "hybrid"):
    now = datetime.now(timezone.utc).isoformat()
    prev = json.loads(STATE.read_text()) if STATE.exists() else {"book": {}}
    tgt = target_book()
    px = _live_oi_panel()[0].iloc[-1]

    for coin, w_new in tgt.items():
        w_old = prev["book"].get(coin, 0.0)
        dw = w_new - w_old
        if abs(dw) < 1e-4:
            continue
        side = "buy" if dw > 0 else "sell"
        notional = abs(dw) * AUM_USD
        # spec C1: split; spec C2/I1: maker-first ladder prediction
        from lab.exec.execution import split_order, plan_ladder, LadderConfig
        slices = split_order(notional, n_slices=4)
        lad = plan_ladder(coin, side, slices[0], now, cfg=LadderConfig())
        pc = cost_bps(coin, side, notional, now, mode, leg="entry")
        intended = float(px.get(coin, np.nan))
        row = {"ts": now, "coin": coin, "side": side, "target_w": float(w_new),
               "delta_w": float(dw), "notional": notional, "n_slices": len(slices),
               "intended_price": intended, "actual_fill": None,
               "predicted_cost_bps": pc["total_bps"],
               "predicted_cost_bps_ladder": lad.predicted_cost_bps,
               "realized_cost_bps": None,
               "p_fill_predicted": lad.p_fill_predicted, "filled": False,
               "latency_ms": None, "mode": mode, "armed": arm}
        if arm:
            row = _place_real_order(row)          # fills in actual_fill / realized_cost / latency
        _record(row)

    STATE.write_text(json.dumps({"book": {k: float(v) for k, v in tgt.items()},
                                 "ts": now}, indent=1))
    print(f"[{now}] {'ARMED' if arm else 'dry-run'}  book of {len(tgt)} names, "
          f"gross {tgt.abs().sum():.2f}, top: "
          f"{', '.join(f'{k}{v:+.2f}' for k, v in tgt.sort_values().tail(3).items())}")


def _place_real_order(row: dict) -> dict:
    """Only reached with --arm and real trade keys. Post-only limit at touch,
    taker fallback after TIF, then reconcile realised cost."""
    raise NotImplementedError(
        "R8: live order placement requires human sign-off. Wire to "
        "keys.binance_signed('/fapi/v1/order', ...) here and remove this guard "
        "only after the paper-review checklist in lab/reports/PAPER_TEST_PLAN.md.")


def review() -> dict:
    if not LOG.exists():
        return {"status": "no orders logged yet"}
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    filled = df[df["filled"]] if "filled" in df else df.iloc[:0]
    out = {"orders": len(df), "filled": int(len(filled)),
           "predicted_cost_bps_mean": float(df["predicted_cost_bps"].mean()),
           "date_range": [df["ts"].min(), df["ts"].max()]}
    if len(filled):
        out["realized_cost_bps_mean"] = float(filled["realized_cost_bps"].mean())
        out["cost_model_error_bps"] = float((filled["realized_cost_bps"]
                                             - filled["predicted_cost_bps"]).mean())
        out["fill_rate"] = float(len(filled) / len(df))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="store_true", help="place real paper/testnet orders (needs keys + R8)")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "taker", "maker"])
    ap.add_argument("--review", action="store_true")
    a = ap.parse_args()
    if a.review:
        print(json.dumps(review(), indent=1, default=float))
    else:
        step(arm=a.arm, mode=a.mode)
