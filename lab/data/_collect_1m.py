"""Download + cache 1m klines for the research universe (data.binance.vision)."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lab.data.binance_vision as bv  # noqa
from lab.data.ingest import _load_1m, month_range  # noqa

SPEC = ROOT / "data" / "raw" / "_spec"
MONTHS = month_range("2023-09", "2025-08")


def one(sym):
    cache = ROOT / "data" / "processed" / "klines_1m" / f"{sym}.parquet"
    try:
        df = _load_1m(sym, MONTHS)
        n = len(df)
        cache_mb = cache.stat().st_size / 1e6 if cache.exists() else 0
        return sym, "ok", n, round(cache_mb, 1)
    except Exception as e:  # noqa
        return sym, f"err:{type(e).__name__}:{e}"[:80], 0, 0


def main():
    pool = json.load(open(SPEC / "universe_pool.json"))
    pool = [s for s in pool if s != "SLERFUSDT"]
    print(f"downloading 1m klines: {len(pool)} symbols x {len(MONTHS)} months")
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(one, s): s for s in pool}
        for fu in as_completed(futs):
            r = fu.result()
            done += 1
            print(f"  [{done}/{len(pool)}] {r[0]:15} {r[1]:12} rows={r[2]:>8} "
                  f"{r[3] if len(r) > 3 else ''}MB ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
