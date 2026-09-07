"""Bybit linear-perp open-interest history (hourly, ~2y) for the universe.
Bybit's OI API pages back to 2023-09 - this is the multi-year OI history that
Binance does NOT provide. -> F_OI family (previously DATA_UNAVAILABLE)."""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "bybit_oi"; OUT.mkdir(parents=True, exist_ok=True)
SYMS = [s for s in json.load(open(ROOT / "data/raw/_spec/universe_pool.json")) if s != "SLERFUSDT"]
START = int(pd.Timestamp("2023-09-01", tz="UTC").timestamp() * 1000)
END = int(pd.Timestamp("2025-08-31 23:00", tz="UTC").timestamp() * 1000)

def _get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "crypto-lab/1.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(1.0 + i)
    return None

def one(sym):
    f = OUT / f"{sym}.parquet"
    if f.exists():
        return sym, "cached", len(pd.read_parquet(f))
    rows, end = [], END
    while end > START:
        url = (f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={sym}"
               f"&intervalTime=1h&limit=200&endTime={end}")
        d = _get(url)
        if not d or d.get("retCode") != 0:
            break
        lst = d["result"]["list"]
        if not lst:
            break
        rows += lst
        oldest = int(lst[-1]["timestamp"])
        if oldest <= START or oldest >= end:
            break
        end = oldest - 1
        time.sleep(0.08)
    if not rows:
        return sym, "no_data", 0
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_numeric(df["timestamp"]).astype("int64")
    df["oi"] = pd.to_numeric(df["openInterest"], errors="coerce")
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").sort_values("ts")[["dt", "oi"]]
    df.to_parquet(f)
    return sym, "ok", len(df)

def main():
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=5) as ex:
        res = list(ex.map(one, SYMS))
    from collections import Counter
    print(Counter(s for _, s, _ in res), f"{time.time()-t0:.0f}s")
    for sym, s, n in res:
        if s not in ("ok", "cached"):
            print(f"  {sym}: {s}")
    ok = [(sym,n) for sym,s,n in res if s in ("ok","cached")]
    if ok:
        print("sample lengths:", sorted(n for _,n in ok)[:5], "...", sorted(n for _,n in ok)[-3:])

if __name__ == "__main__":
    main()
