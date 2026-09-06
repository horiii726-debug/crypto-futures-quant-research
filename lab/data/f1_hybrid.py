"""F-1 recomputed with the maker/hybrid execution cost floor.

Uses lab.exec.maker_model calibrations (real aggTrades tape) to get a
per-liquidity-tier hybrid round-trip cost, then rebuilds the required-IC /
required-hit-rate table and reports which horizons reopen vs taker-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
from lab.data.f1 import load_panel_close, dispersion_and_move, IC_CEILING, HITRATE_CEILING  # noqa

HBARS = {"5m": 1 / 12, "15m": 0.25, "1h": 1, "4h": 4, "1d": 24, "3d": 72}


def build(tag: str, *, taker_half_spread_bps: float, hybrid_roundtrip_bps: float,
          taker_fee: float, mean_abs_funding_8h: float,
          turnover_per_rebalance: float = 1.2,
          horizons=("5m", "15m", "1h", "4h", "1d", "3d")) -> dict:
    base = load_panel_close(tag, "1h")
    taker_rt = 2 * (taker_fee + taker_half_spread_bps * 1e-4)
    rows = []
    for h in horizons:
        hrs = HBARS[h]
        kb = max(1, round(hrs))
        D1, R1 = dispersion_and_move(base, kb)
        if hrs < 1:
            D, R = D1 * np.sqrt(hrs), R1 * np.sqrt(hrs)
        else:
            D, R = D1, R1
        funding_cost = mean_abs_funding_8h * max(hrs / 8.0, 0.0) * 0.5
        taker_floor = taker_rt + funding_cost + 0.5e-4
        hybrid_floor = hybrid_roundtrip_bps * 1e-4 + funding_cost + 0.5e-4
        def verdict(floor):
            ric = turnover_per_rebalance * floor / D if D > 0 else np.inf
            rhr = 0.5 + floor / (2 * R) if R > 0 else np.inf
            return ric, rhr, (ric <= IC_CEILING and rhr <= HITRATE_CEILING)
        tric, trhr, tfe = verdict(taker_floor)
        hric, hrhr, hfe = verdict(hybrid_floor)
        rows.append({
            "horizon": h, "hold_hours": hrs,
            "taker_cost_floor_bps": round(taker_floor * 1e4, 2),
            "hybrid_cost_floor_bps": round(hybrid_floor * 1e4, 2),
            "dispersion_D_bps": round(D * 1e4, 1),
            "required_IC_taker": round(tric, 4), "FEASIBLE_taker": bool(tfe),
            "required_IC_hybrid": round(hric, 4), "FEASIBLE_hybrid": bool(hfe),
            "required_hit_hybrid": round(hrhr, 4),
        })
    reopened = [r["horizon"] for r in rows if r["FEASIBLE_hybrid"] and not r["FEASIBLE_taker"]]
    return {
        "tag": tag,
        "taker_half_spread_bps": taker_half_spread_bps,
        "hybrid_roundtrip_bps": hybrid_roundtrip_bps,
        "taker_roundtrip_bps": round(taker_rt * 1e4, 2),
        "rows": rows,
        "feasible_taker": [r["horizon"] for r in rows if r["FEASIBLE_taker"]],
        "feasible_hybrid": [r["horizon"] for r in rows if r["FEASIBLE_hybrid"]],
        "reopened_by_hybrid": reopened,
    }


def to_md(res: dict) -> str:
    L = ["# F-1 (maker/hybrid execution) — measured costs\n",
         f"[SPEC] Universe `{res['tag']}`. Taker round-trip "
         f"{res['taker_roundtrip_bps']} bps vs **hybrid round-trip "
         f"{res['hybrid_roundtrip_bps']:.2f} bps** (limit-at-touch, p_fill and "
         f"adverse selection simulated on the real aggTrades tape, taker "
         f"fallback after the wait). NO 100% maker-fill assumption.\n",
         "| horizon | taker floor (bps) | hybrid floor (bps) | disp D (bps) | req IC taker | req IC hybrid | FEASIBLE taker | FEASIBLE hybrid |",
         "|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        L.append(f"| {r['horizon']} | {r['taker_cost_floor_bps']} | {r['hybrid_cost_floor_bps']} | "
                 f"{r['dispersion_D_bps']} | {r['required_IC_taker']} | {r['required_IC_hybrid']} | "
                 f"{'yes' if r['FEASIBLE_taker'] else 'no'} | "
                 f"{'**yes**' if r['FEASIBLE_hybrid'] else 'no'} |")
    L.append(f"\n**Feasible (taker-only):** {', '.join(res['feasible_taker']) or 'NONE'}")
    L.append(f"**Feasible (hybrid):** {', '.join(res['feasible_hybrid']) or 'NONE'}")
    L.append(f"**Reopened by hybrid execution:** {', '.join(res['reopened_by_hybrid']) or 'none'}")
    return "\n".join(L)


if __name__ == "__main__":
    print("f1_hybrid ready")
