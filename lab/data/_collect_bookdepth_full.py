"""Full daily bookDepth for the 2023-09 .. 2024-05 overlap window (all the
bookDepth there is), ~15 coins. Small files. -> daily F_BOOK / F_LIQ features."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab.data.microstructure import book_depth  # noqa

OUT = ROOT / "data" / "processed" / "bookdepth_daily"
OUT.mkdir(parents=True, exist_ok=True)
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT",
        "AVAXUSDT", "ADAUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "INJUSDT", "LDOUSDT",
        "CRVUSDT", "APTUSDT", "FILUSDT", "SUIUSDT", "ARBUSDT", "OPUSDT",
        "WLDUSDT", "DARUSDT", "HOOKUSDT", "AMBUSDT", "IDEXUSDT", "KEYUSDT",
        "STORJUSDT", "LOOMUSDT", "SUSHIUSDT", "ZENUSDT"]
DAYS = [d.strftime("%Y-%m-%d") for d in pd.date_range("2023-09-01", "2024-05-17", freq="D")]


def daily_book_features(bd: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the ~1-min bookDepth snapshots to a daily row per symbol:
    depth imbalance (bid vs ask notional at +-1%), book slope, near-touch $."""
    if bd.empty:
        return pd.DataFrame()
    bd = bd.copy()
    bd["day"] = bd["ts"].dt.floor("D")
    piv = bd.pivot_table(index=["day", "ts"], columns="percentage", values="notional", aggfunc="last")
    cols = list(piv.columns)
    b1 = -1 if -1 in cols else max([c for c in cols if c < 0], default=None)
    a1 = 1 if 1 in cols else min([c for c in cols if c > 0], default=None)
    b5 = min(cols) if cols else None
    if b1 is None or a1 is None:
        return pd.DataFrame()
    imb = (piv[b1] - piv[a1]) / (piv[b1] + piv[a1])
    slope = piv[b1] / piv[b5] if b5 in piv else pd.Series(index=piv.index, dtype=float)
    near = (piv[b1] + piv[a1]) / 2
    df = pd.DataFrame({"depth_imb": imb, "book_slope": slope, "near_notional": near})
    df = df.reset_index()
    g = df.groupby("day").agg(depth_imb=("depth_imb", "mean"),
                              depth_imb_std=("depth_imb", "std"),
                              book_slope=("book_slope", "median"),
                              near_notional=("near_notional", "median"))
    return g


def one(sym):
    f = OUT / f"{sym}.parquet"
    if f.exists():
        return sym, "cached", len(pd.read_parquet(f))
    try:
        bd = book_depth(sym, DAYS)
        if bd.empty:
            return sym, "no_data", 0
        g = daily_book_features(bd)
        g.to_parquet(f)
        return sym, "ok", len(g)
    except Exception as e:  # noqa
        return sym, f"err:{type(e).__name__}:{e}"[:70], 0


def main():
    t0 = time.time()
    print(f"bookDepth daily: {len(SYMS)} syms x up to {len(DAYS)} days (2023-09..2024-05)")
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(one, s): s for s in SYMS}
        for fu in as_completed(futs):
            s, st, n = fu.result()
            print(f"  {s:10} {st:12} days={n} ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
