"""F-1 feasibility: required IC and required hit-rate per horizon, given the
MEASURED cost floor. A horizon whose required IC exceeds what admissible
crypto cross-sectional effects plausibly deliver (~0.05 gross IC ceiling) is
INFEASIBLE and closed to research.

cost floor per rebalance (round-trip, long-short book):
    2 * taker_fee                        (enter+exit, taker both sides, conservative)
  + 2 * half_spread                      (measured, CS/AR + aggTrades validated)
  + funding_over_hold                    (|mean funding| * hold/interval, per net leg)
  + impact                              (small linear term, clip vs ADV)

edge model (rank long/short, decile):
    gross return per rebalance  ~ IC * D      (D = cross-sec std of h-returns)
    net                         = IC * D  -  turnover * cost_floor
    => required IC              = turnover * cost_floor / D
hit-rate model (directional):
    required p                  = 0.5 + cost_floor / (2 * R)   (R = E|h-return|)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
IC_CEILING = 0.05          # generous ceiling for admissible XS crypto IC (gross)
HITRATE_CEILING = 0.56


def load_panel_close(tag: str, horizon: str) -> pd.DataFrame:
    return pd.read_parquet(PROC / "panels" / tag / horizon / "close.parquet")


def dispersion_and_move(close: pd.DataFrame, k_bars: int) -> tuple[float, float]:
    fwd = close.shift(-k_bars) / close - 1.0
    D = float(fwd.std(axis=1).median())          # cross-sectional dispersion
    R = float(fwd.abs().stack().median())         # typical absolute move
    return D, R


def f1_table(tag: str, panel_freq_hours: float, half_spread_prop: float,
             mean_abs_funding_8h: float, venue: dict,
             horizons=("5m", "15m", "1h", "4h", "1d", "3d"),
             turnover_per_rebalance: float = 1.6) -> dict:
    taker = float(venue["fees"]["taker_fee"])
    # base panel available is 1h; approximate finer horizons by scaling dispersion
    base = load_panel_close(tag, "1h")
    hbars = {"5m": 1 / 12, "15m": 0.25, "1h": 1, "4h": 4, "1d": 24, "3d": 72}
    rows = []
    for h in horizons:
        hrs = hbars[h]
        kb = max(1, round(hrs / 1.0))  # in 1h bars
        D1, R1 = dispersion_and_move(base, max(kb, 1))
        if hrs < 1:  # sub-hour: scale by sqrt-time from the 1h estimate
            D = D1 * np.sqrt(hrs / 1.0)
            R = R1 * np.sqrt(hrs / 1.0)
        else:
            D, R = D1, R1
        hold_intervals = hrs / 8.0
        funding_cost = mean_abs_funding_8h * max(hold_intervals, 0.0) * 0.5  # net leg
        impact = 0.5e-4  # 0.5 bp linear impact assumption for <=0.3% ADV clips
        cost_floor = 2 * taker + 2 * half_spread_prop + funding_cost + impact
        req_ic = turnover_per_rebalance * cost_floor / D if D > 0 else np.inf
        req_hit = 0.5 + cost_floor / (2 * R) if R > 0 else np.inf
        feasible = (req_ic <= IC_CEILING) and (req_hit <= HITRATE_CEILING)
        rows.append({
            "horizon": h, "hold_hours": hrs,
            "cost_floor_bps": round(cost_floor * 1e4, 2),
            "  taker_x2_bps": round(2 * taker * 1e4, 2),
            "  spread_x2_bps": round(2 * half_spread_prop * 1e4, 2),
            "  funding_bps": round(funding_cost * 1e4, 2),
            "dispersion_D_bps": round(D * 1e4, 1),
            "typ_move_R_bps": round(R * 1e4, 1),
            "required_IC": round(req_ic, 4),
            "required_hit_rate": round(req_hit, 4),
            "FEASIBLE": bool(feasible),
        })
    return {"tag": tag, "half_spread_bps": round(half_spread_prop * 1e4, 3),
            "mean_abs_funding_8h_bps": round(mean_abs_funding_8h * 1e4, 3),
            "ic_ceiling": IC_CEILING, "hit_rate_ceiling": HITRATE_CEILING,
            "turnover_assumed": turnover_per_rebalance,
            "rows": rows,
            "feasible_horizons": [r["horizon"] for r in rows if r["FEASIBLE"]],
            "infeasible_horizons": [r["horizon"] for r in rows if not r["FEASIBLE"]]}


if __name__ == "__main__":
    print("f1 module ready")
