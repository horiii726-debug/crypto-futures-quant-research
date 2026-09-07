"""indexPriceKlines (the underlying index the perp tracks) for the universe.
Combined with perp close (from klines_1m panels) -> a REAL basis, not the
smoothed premium index. -> hardened carry study."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lab.data.binance_vision as bv  # noqa
from lab.data.ingest import month_range  # noqa

OUT = ROOT / "data" / "processed" / "basis"
SYMS = [s for s in json.load(open(ROOT / "data/raw/_spec/universe_pool.json")) if s != "SLERFUSDT"]
MONTHS = month_range("2023-09", "2025-08")


def one(sym):
    f = OUT / f"{sym}.parquet"
    if not f.exists():
        return sym, "no_basis_file"
    d = pd.read_parquet(f)
    if "index_close" in d.columns:
        return sym, "cached"
    try:
        ix = bv.index_klines(sym, MONTHS, kind="indexPriceKlines", interval="1h")
        mk = bv.index_klines(sym, MONTHS, kind="markPriceKlines", interval="1h")
    except Exception as e:  # noqa
        return sym, f"err:{e}"[:50]
    if ix.empty:
        return sym, "no_index"
    d["index_close"] = ix.set_index("dt")["close"].reindex(d.index)
    if not mk.empty:
        d["mark_close"] = mk.set_index("dt")["close"].reindex(d.index)
    # perp close from the 1m panel (1h) if available
    for part in ("train", "valid", "test"):
        p = ROOT / "data" / part / "prio50_2y" / "1h" / "close.parquet"
        if p.exists():
            pc = pd.read_parquet(p)
            if sym in pc.columns:
                d["perp_close"] = pd.concat(
                    [d.get("perp_close", pd.Series(dtype=float)), pc[sym]]
                ).groupby(level=0).last().reindex(d.index)
    d.to_parquet(f)
    return sym, "ok"


def main():
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(one, SYMS))
    from collections import Counter
    print(Counter(s for _, s in res), f"{time.time()-t0:.0f}s")
    for sym, s in res:
        if s.startswith("err") or s in ("no_index", "no_basis_file"):
            print(f"  {sym}: {s}")


if __name__ == "__main__":
    main()
