"""F-1 feasibility for intraday microstructure at M3/M5/M15 with maker vs
taker execution. Uses realised per-bar volatility from the tape panel as the
'dispersion' a directional TS signal can capture.

required_hit_rate = 0.5 + cost_floor / (2 * R)   with R = E|bar close-to-close move|
required_edge_bps = turnover_per_bar * cost_floor   (must be < gross edge/bar)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_EXEC = json.loads((ROOT / "data" / "processed" / "exec_summary.json").read_text())
MAKER_RT = _EXEC["hybrid_roundtrip_bps_mean"] * 1e-4
TAKER_RT = 2 * (0.0005 + 0.85e-4)


def build(tape_panels: dict[int, dict], hit_ceiling: float = 0.545) -> dict:
    rows = []
    for bar_min, panel in tape_panels.items():
        cc = []
        for df in panel.values():
            r = np.log(df["close"]).diff().abs()
            cc.append(r[np.isfinite(r)])
        R = float(pd.concat(cc).median())          # typical |bar move|
        for mode, rt in (("taker", TAKER_RT), ("maker", MAKER_RT)):
            # 1-bar hold: enter+exit each bar => 1 round trip / bar => turnover ~2
            for hold in (1, 3, 6, 12):
                rt_per_bar = rt / hold             # amortise the round trip over the hold
                req_hit = 0.5 + (rt_per_bar / 2) / (2 * R) if R > 0 else np.inf
                req_edge_bps = rt_per_bar * 1e4
                feasible = req_hit <= hit_ceiling and req_edge_bps < R * 1e4 * 0.6
                rows.append({
                    "bar_min": bar_min, "mode": mode, "hold_bars": hold,
                    "typ_move_bps": round(R * 1e4, 2),
                    "roundtrip_bps": round(rt * 1e4, 2),
                    "cost_per_bar_bps": round(req_edge_bps, 2),
                    "required_hit_rate": round(req_hit, 4),
                    "FEASIBLE": bool(feasible),
                })
    feas = sorted({(r["bar_min"], r["mode"], r["hold_bars"]) for r in rows if r["FEASIBLE"]})
    return {"rows": rows, "feasible": [list(x) for x in feas],
            "maker_roundtrip_bps": round(MAKER_RT * 1e4, 2),
            "taker_roundtrip_bps": round(TAKER_RT * 1e4, 2)}


def to_md(res: dict) -> str:
    L = ["# F-1 — intraday microstructure (M3/M5/M15), maker vs taker\n",
         f"[SPEC] Maker round-trip {res['maker_roundtrip_bps']} bps (tape-calibrated "
         f"hybrid), taker {res['taker_roundtrip_bps']} bps. A directional bar signal is "
         f"feasible if the required hit-rate <= 0.545 and the per-bar cost < 60% of the "
         f"typical bar move.\n",
         "| bar | mode | hold | typ move (bps) | cost/bar (bps) | req hit-rate | FEASIBLE |",
         "|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        L.append(f"| {r['bar_min']}m | {r['mode']} | {r['hold_bars']} | {r['typ_move_bps']} | "
                 f"{r['cost_per_bar_bps']} | {r['required_hit_rate']} | "
                 f"{'**yes**' if r['FEASIBLE'] else 'no'} |")
    L.append(f"\n**Feasible (bar, mode, hold):** {res['feasible'] or 'NONE'}")
    return "\n".join(L)
