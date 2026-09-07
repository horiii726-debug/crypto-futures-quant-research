"""RESEARCH ROUND 2 · P2.2 — imbvol_fade as an EXECUTION-TIMING overlay.

NOT a trial. No directional hypothesis: the overlay may not change position sign
or size; it only chooses WHICH 5-minute bar inside an already-fixed rebalance
window to execute in. If the window ends, execution is forced.

Baseline  : execute every rebalance at a fixed +2-bar offset (what the studies do).
Overlay   : within a 12-bar (1h) window after the signal, execute in the bar where
            imbvol_fade most favours our intended trade side
            (buy when the book says the next tick is up; sell when down).
            Forced execution at bar 12 if never favourable.

Metric (the only one): mean realised 5m execution-return edge, bps, overlay minus
baseline, measured on real 5m close-to-close moves. Positive = the overlay fills
at better prices for the same trade.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "data" / "processed" / "bookdepth_bars"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT",
        "LINKUSDT", "AVAXUSDT", "ADAUSDT", "LTCUSDT", "BCHUSDT", "INJUSDT"]
WINDOW = 12          # 1h in 5-min bars
K = 3


def _zx(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def _load():
    from lab.data.ingest import _load_1m
    months = [str(p) for p in pd.period_range("2023-09", "2024-05", freq="M")]
    price, imb = {}, {}
    for s in SYMS:
        p = BOOK / f"{s}_5m.parquet"
        if not p.exists():
            continue
        b = pd.read_parquet(p)
        d = _load_1m(s, months)
        if d.empty:
            continue
        px = d.set_index("dt")["close"].resample("5min", label="left", closed="left").last()
        price[s] = px.reindex(b.index)
        imb[s] = b["imb_vol"].reindex(b.index)
    C = pd.DataFrame(price).sort_index()
    V = pd.DataFrame(imb).reindex(C.index)
    return C, V


def run() -> dict:
    C, V = _load()
    sgn = np.sign(np.log(C).diff(K))
    # imbvol_fade: >0 => expect price UP next => good bar to BUY
    tilt = -_zx(V.rolling(K, min_periods=1).mean() * sgn)
    ret5 = C.pct_change()

    rng = np.random.default_rng(7)
    # simulate rebalances every 12 bars; random intended side per coin per window
    idx = np.arange(0, len(C) - WINDOW - 1, WINDOW)
    base_edges, ovl_edges = [], []
    for s in C.columns:
        t = tilt[s].values
        r = ret5[s].values
        for start in idx:
            side = rng.choice([-1.0, 1.0])          # intended trade side (fixed, not chosen by overlay)
            w = slice(start, start + WINDOW)
            tw = t[w] * side                        # favourable when >0
            rw = r[w]
            if not np.isfinite(rw).any():
                continue
            # baseline: fixed +2 offset
            b_bar = 2
            # overlay: first bar where tilt favours the trade; else forced at end
            fav = np.where(np.isfinite(tw) & (tw > 0.3))[0]
            o_bar = int(fav[0]) if len(fav) else WINDOW - 1
            # execution edge = -(side * price move from window-open to fill bar)
            # (buying after a dip = positive edge; the mid we pay vs the window's open)
            base_px_move = np.nansum(rw[1:b_bar + 1]) if b_bar >= 1 else 0.0
            ovl_px_move = np.nansum(rw[1:o_bar + 1]) if o_bar >= 1 else 0.0
            base_edges.append(-side * base_px_move * 1e4)
            ovl_edges.append(-side * ovl_px_move * 1e4)
    base = float(np.nanmean(base_edges))
    ovl = float(np.nanmean(ovl_edges))
    out = {"n_rebalances": len(base_edges),
           "baseline_exec_edge_bps": base,
           "overlay_exec_edge_bps": ovl,
           "delta_bps": ovl - base,
           "delta_bps_se": float(np.nanstd(np.array(ovl_edges) - np.array(base_edges))
                                 / np.sqrt(len(base_edges)))}
    print(f"  n rebalances       : {out['n_rebalances']}")
    print(f"  baseline exec edge : {base:+.3f} bps")
    print(f"  overlay  exec edge : {ovl:+.3f} bps")
    print(f"  delta (overlay-base): {out['delta_bps']:+.3f} ± {out['delta_bps_se']:.3f} bps")
    verdict = ("overlay saves cost — fold into execution layer" if out["delta_bps"] > 2 * out["delta_bps_se"]
               else "no material execution saving from imbvol timing")
    out["_verdict"] = verdict
    print(f"  VERDICT: {verdict}")
    return out


if __name__ == "__main__":
    r = run()
    json.dump(r, open(ROOT / "lab" / "reports" / "IMBVOL_OVERLAY.json", "w"), indent=1, default=float)
    print("\nwrote lab/reports/IMBVOL_OVERLAY.json")
