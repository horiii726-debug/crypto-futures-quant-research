"""bookDepth (DOM) -> causal intraday bar features.

Raw: timestamp, percentage (-5..-1,+1..+5 = % from mid), depth (cum base qty
to that band), notional (cum quote value). ~1 snapshot / 30s, 2023-01..2024-05.

Per bar (5m/15m), using only snapshots with ts <= bar close:
  depth_imb_1pct   (bid_notional_-1 - ask_notional_+1)/(sum)
  depth_imb_5pct   same at the +-5% band
  book_slope       near(+-1%) / far(+-5%) cumulative notional, avg of both sides
  total_depth_z    total +-5% notional, later z-scored
  depth_withdraw   change in total depth vs previous bar (liquidity pulling)
  microprice_tilt  (bid_qty_1 - ask_qty_1)/(sum) -> expected drift
  imb_vol          within-bar std of the 1% imbalance (book instability)
"""
from __future__ import annotations
import io, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from lab.data.binance_vision import CACHE

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed" / "bookdepth_bars"


def _read_day(sym, day):
    key = f"data/futures/um/daily/bookDepth/{sym}/{sym}-bookDepth-{day}.zip"
    p = CACHE / key
    if not p.exists():
        return None
    with zipfile.ZipFile(io.BytesIO(p.read_bytes())) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw))
    if "percentage" not in df.columns:
        df.columns = ["timestamp", "percentage", "depth", "notional"]
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ("percentage", "depth", "notional"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["ts"])


def _bar_feats(df: pd.DataFrame, bar_min: int) -> pd.DataFrame:
    # pivot: index=snapshot ts, columns=percentage band, values=notional
    piv = df.pivot_table(index="ts", columns="percentage", values="notional", aggfunc="last")
    dpiv = df.pivot_table(index="ts", columns="percentage", values="depth", aggfunc="last")
    cols = piv.columns
    b1 = -1.0 if -1.0 in cols else min([c for c in cols if c < 0], default=np.nan)
    a1 = 1.0 if 1.0 in cols else min([c for c in cols if c > 0], default=np.nan)
    b5 = min(cols); a5 = max(cols)
    imb1 = (piv[b1] - piv[a1]) / (piv[b1] + piv[a1])
    imb5 = (piv[b5] - piv[a5]) / (piv[b5] + piv[a5])
    slope = ((piv[b1] / piv[b5]).replace([np.inf, -np.inf], np.nan)
             + (piv[a1] / piv[a5]).replace([np.inf, -np.inf], np.nan)) / 2
    total = piv[b5] + piv[a5]
    mp = (dpiv[b1] - dpiv[a1]) / (dpiv[b1] + dpiv[a1])
    snap = pd.DataFrame({"imb1": imb1, "imb5": imb5, "slope": slope,
                         "total": total, "mp": mp})
    rule = f"{bar_min}min"
    g = snap.resample(rule, label="left", closed="left")
    bar = pd.DataFrame({
        "depth_imb_1pct": g["imb1"].mean(),
        "depth_imb_5pct": g["imb5"].mean(),
        "imb_vol": g["imb1"].std(),
        "book_slope": g["slope"].median(),
        "total_depth": g["total"].mean(),
        "microprice_tilt": g["mp"].mean(),
        "n_snap": g["imb1"].count(),
    })
    bar["depth_withdraw"] = bar["total_depth"].pct_change()
    return bar[bar["n_snap"] > 0]


def bar_features(sym, days, bar_min):
    out = []
    for d in days:
        raw = _read_day(sym, d)
        if raw is None or len(raw) < 100:
            continue
        try:
            out.append(_bar_feats(raw, bar_min))
        except Exception:
            continue
    if not out:
        return pd.DataFrame()
    df = pd.concat(out).sort_index()
    return df[~df.index.duplicated(keep="last")]


def build_panel(syms, days, bar_min, cache=True):
    PROC.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in syms:
        f = PROC / f"{s}_{bar_min}m.parquet"
        if cache and f.exists():
            out[s] = pd.read_parquet(f)
            continue
        df = bar_features(s, days, bar_min)
        if len(df):
            df.to_parquet(f)
            out[s] = df
    return out


if __name__ == "__main__":
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-03-01", "2024-03-05")]
    df = bar_features("ETHUSDT", days, 15)
    print(df.tail(6).to_string())
