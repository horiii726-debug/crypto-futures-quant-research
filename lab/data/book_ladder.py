"""Per-coin order-book ladder from raw Binance bookDepth zips (spec A6).

bookDepth rows: timestamp, percentage, depth, notional
  percentage in {-5,-4,-3,-2,-1, 1,2,3,4,5} = cumulative depth within that % of mid

Output data/processed/book_ladder.json:
  {coin: {"pct_bands":[1,2,3,4,5],
          "cum_notional_usd_median":[...],      # median cumulative $ within +-k%
          "incremental_notional_usd_median":[...],# per-band $ (cum diff)
          "l1_notional_usd_est": float,          # ~ first slice, from the 1% band
          "n_snapshots": int}}
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "data" / "futures" / "um" / "daily" / "bookDepth"
OUT = ROOT / "data" / "processed" / "book_ladder.json"
BANDS = [1, 2, 3, 4, 5]


def _coin(sym: str, max_days: int = 120) -> dict | None:
    d = RAW / sym
    if not d.exists():
        return None
    files = sorted(d.glob(f"{sym}-bookDepth-*.zip"))
    if not files:
        return None
    step = max(1, len(files) // max_days)
    files = files[::step]
    cum = {b: [] for b in BANDS}       # cumulative notional within +-b% (avg of bid+ask side)
    n = 0
    for f in files:
        try:
            with zipfile.ZipFile(f) as z:
                raw = z.read(z.namelist()[0])
            df = pd.read_csv(io.BytesIO(raw))
        except Exception:
            continue
        df["ap"] = df["percentage"].abs()
        g = df.groupby(["timestamp", "ap"])["notional"].mean().unstack("ap")
        for b in BANDS:
            if b in g.columns:
                cum[b].append(g[b].dropna().values)
        n += g.shape[0]
    if n == 0:
        return None
    cum_med = [float(np.median(np.concatenate(cum[b]))) if cum[b] else np.nan for b in BANDS]
    inc = [cum_med[0]] + [cum_med[i] - cum_med[i - 1] for i in range(1, len(BANDS))]
    return {"pct_bands": BANDS, "cum_notional_usd_median": cum_med,
            "incremental_notional_usd_median": inc,
            "l1_notional_usd_est": float(cum_med[0] * 1e-2),   # ~1% of the 1% band ~ touch
            "n_snapshots": int(n)}


def build() -> dict:
    out = {}
    for d in sorted(RAW.iterdir()):
        if not d.is_dir():
            continue
        r = _coin(d.name)
        if r:
            out[d.name] = r
            print(f"  {d.name:10} 1%%=${r['cum_notional_usd_median'][0]/1e6:8.1f}M  "
                  f"5%%=${r['cum_notional_usd_median'][4]/1e6:8.1f}M  n={r['n_snapshots']}", flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    build()
    print(f"\nwrote {OUT}  ({len(json.loads(OUT.read_text()))} coins)")
