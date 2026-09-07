"""Parallel tape-panel builder: (symbol, month) chunks across a process pool,
per-chunk parquet, then concat to per-(symbol,bar) panels."""
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab.data.tape import bar_features, PROC  # noqa

CHUNK = PROC / "_chunks"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT",
        "LINKUSDT", "AVAXUSDT", "ADAUSDT", "LTCUSDT"]
MONTHS = [str(p) for p in pd.period_range("2024-08", "2025-07", freq="M")]
BARS = [3]


def chunk_task(sym, month, bar):
    out = CHUNK / f"{sym}_{month}_{bar}m.parquet"
    if out.exists():
        return sym, month, bar, "cached", 0
    per = pd.Period(month, freq="M")
    days = [d.strftime("%Y-%m-%d")
            for d in pd.date_range(per.start_time, per.end_time, freq="D")]
    try:
        df = bar_features(sym, days, bar)
    except Exception as e:  # noqa
        return sym, month, bar, f"err:{e}"[:60], 0
    if len(df):
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    return sym, month, bar, "ok", len(df)


def main():
    CHUNK.mkdir(parents=True, exist_ok=True)
    tasks = [(s, m, b) for s in SYMS for m in MONTHS for b in BARS]
    print(f"{len(tasks)} (symbol,month,bar) chunks, pool=6")
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(chunk_task, *t): t for t in tasks}
        for fu in as_completed(futs):
            s, m, b, st, n = fu.result()
            done += 1
            if st != "cached":
                print(f"  [{done}/{len(tasks)}] {s} {m} {b}m {st} n={n} ({time.time()-t0:.0f}s)", flush=True)
    # concat chunks -> per (symbol, bar) panel
    for s in SYMS:
        for b in BARS:
            parts = sorted(CHUNK.glob(f"{s}_*_{b}m.parquet"))
            if not parts:
                continue
            import numpy as np
            df = pd.concat([pd.read_parquet(p) for p in parts]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
            rat = df["vwap"] / df["vwap"].shift(1)
            df["twap_ret"] = np.log(rat.where(rat > 0))
            df["cum_delta"] = df["ofi_usd"].cumsum()
            df.to_parquet(PROC / f"{s}_{b}m.parquet")
            print(f"  panel {s} {b}m -> {df.shape} {df.index.min()}..{df.index.max()}")
    print(f"DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
