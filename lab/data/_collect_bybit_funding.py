import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "bybit_funding"; OUT.mkdir(parents=True, exist_ok=True)
SYMS = [s for s in json.load(open(ROOT / "data/raw/_spec/universe_pool.json")) if s != "SLERFUSDT"]
START_MS = int(pd.Timestamp("2023-09-01", tz="UTC").timestamp() * 1000)
END_MS = int(pd.Timestamp("2025-08-31", tz="UTC").timestamp() * 1000)

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
    rows, end = [], END_MS
    while end > START_MS:
        url = ("https://api.bybit.com/v5/market/funding/history?category=linear"
               f"&symbol={sym}&endTime={end}&limit=200")
        d = _get(url)
        if not d or d.get("retCode") != 0:
            break
        lst = d["result"]["list"]
        if not lst:
            break
        rows += lst
        oldest = int(lst[-1]["fundingRateTimestamp"])
        if oldest <= START_MS or oldest >= end:
            break
        end = oldest - 1
        time.sleep(0.12)
    if not rows:
        return sym, "no_data", 0
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_numeric(df["fundingRateTimestamp"]).astype("int64")
    df["funding_bybit"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").sort_values("ts")[["dt", "funding_bybit"]]
    df.to_parquet(f)
    return sym, "ok", len(df)

def main():
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(one, SYMS))
    from collections import Counter
    print(Counter(s for _, s, _ in res), f"{time.time()-t0:.0f}s")
    for sym, s, n in res:
        if s not in ("ok", "cached"):
            print(f"  {sym}: {s}")

if __name__ == "__main__":
    main()
