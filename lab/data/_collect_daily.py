"""Driver: daily-bar collection for every USDT-M perp ever listed, using
data.binance.vision monthly 1d files ONLY (no REST -> no 418 ban)."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lab.data.binance_vision as bv  # noqa

PROC = ROOT / "data" / "processed" / "daily"
SPEC = ROOT / "data" / "raw" / "_spec"


def one(sym):
    f = PROC / f"{sym}.parquet"
    if f.exists():
        return sym, "cached", len(pd.read_parquet(f))
    try:
        files = bv.list_files(sym, kind="klines", freq="monthly", interval="1d")
        months = sorted({k.split("-1d-")[1].replace(".zip", "") for k in files if "-1d-" in k})
        if not months:
            return sym, "nofiles", 0
        df = bv.klines(sym, months, interval="1d", freq="monthly")
    except Exception as e:  # noqa
        return sym, f"err:{type(e).__name__}", 0
    if len(df):
        f.parent.mkdir(parents=True, exist_ok=True)
        keep = ["dt", "open_time", "open", "high", "low", "close", "volume",
                "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
        df[keep].to_parquet(f)
    return sym, "ok", len(df)


def main():
    alls = json.load(open(SPEC / "all_symbols_ever.json"))
    usdt = [s for s in alls if s.endswith(("USDT", "BUSD", "USDC"))]
    print(f"collecting daily bars for {len(usdt)} perps via data.binance.vision")
    t0 = time.time()
    done = ok = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(one, s): s for s in usdt}
        for fu in as_completed(futs):
            sym, status, n = fu.result()
            done += 1
            ok += status in ("ok", "cached")
            if not status.startswith(("ok", "cached")) or done % 100 == 0:
                print(f"  [{done}/{len(usdt)}] {sym} {status} n={n} ({time.time()-t0:.0f}s)", flush=True)
    files = list(PROC.glob("*.parquet"))
    print(f"DONE: {len(files)} symbols with data, {ok}/{done} ok, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
