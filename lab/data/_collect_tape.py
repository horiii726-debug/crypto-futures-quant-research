"""Full aggTrades (tape) download for the microstructure engine.
~10 liquid coins x 12 months. Files kept zipped in data/raw cache; parsed
lazily by lab/data/tape.py."""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab.data.binance_vision import fetch_file, NotFound  # noqa

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT",
        "LINKUSDT", "AVAXUSDT", "ADAUSDT", "LTCUSDT"]
DAYS = [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-08-01", "2025-07-31", freq="D")]


def one_file(sym, d):
    key = f"data/futures/um/daily/aggTrades/{sym}/{sym}-aggTrades-{d}.zip"
    dst = ROOT / "data" / "raw" / key
    if dst.exists():
        return sym, d, "cached", dst.stat().st_size
    try:
        blob = fetch_file(key, verify=False)
        return sym, d, "ok", len(blob)
    except NotFound:
        return sym, d, "404", 0
    except Exception as e:  # noqa
        return sym, d, f"err:{type(e).__name__}", 0


def main():
    tasks = [(s, d) for s in SYMS for d in DAYS]
    print(f"tape download: {len(SYMS)} syms x {len(DAYS)} days = {len(tasks)} files")
    t0 = time.time()
    done = ok = bytes_ = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(one_file, s, d): (s, d) for s, d in tasks}
        for fu in as_completed(futs):
            s, d, st, n = fu.result()
            done += 1
            ok += st in ("ok", "cached")
            bytes_ += n
            if done % 200 == 0 or st.startswith("err"):
                print(f"  [{done}/{len(tasks)}] {s} {d} {st}  "
                      f"{bytes_/1e9:.1f}GB  {time.time()-t0:.0f}s", flush=True)
    print(f"DONE {ok}/{done} ok, {bytes_/1e9:.1f}GB, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
