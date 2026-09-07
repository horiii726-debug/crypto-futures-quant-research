"""Binance vision `metrics` -> OI(USD) hourly panel, back to 2020-09 (spec J1).

metrics CSV columns: create_time, symbol, sum_open_interest,
  sum_open_interest_value (USD), count_toptrader_long_short_ratio,
  sum_toptrader_long_short_ratio, count_long_short_ratio,
  sum_taker_long_short_vol_ratio

5-minute snapshots -> resampled to 1h (last).
"""
from __future__ import annotations

import io
import json
import re
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "binance_oi_hist"
BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"


def _list_days(sym: str) -> list[str]:
    prefix = f"data/futures/um/daily/metrics/{sym}/"
    marker, days = "", []
    for _ in range(20):
        url = f"{S3}?delimiter=/&prefix={prefix}&marker={marker}"
        x = urllib.request.urlopen(url, timeout=30).read().decode()
        ks = [k for k in re.findall(r"<Key>([^<]+)</Key>", x) if k.endswith(".zip")]
        for k in ks:
            m = re.search(r"metrics-(\d{4}-\d{2}-\d{2})\.zip", k)
            if m:
                days.append(m.group(1))
        if "<IsTruncated>true" not in x or not ks:
            break
        marker = re.findall(r"<Key>([^<]+)</Key>", x)[-1]
    return sorted(set(days))


def _day(sym: str, d: str) -> pd.DataFrame | None:
    url = f"{BASE}/{sym}/{sym}-metrics-{d}.zip"
    try:
        raw = urllib.request.urlopen(url, timeout=30).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    except Exception:
        return None
    df["ts"] = pd.to_datetime(df["create_time"], utc=True)
    df = df.set_index("ts")
    keep = df[["sum_open_interest_value", "sum_open_interest",
               "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]].apply(
        pd.to_numeric, errors="coerce")
    return keep


def build_symbol(sym: str, start: str = "2020-09-01", end: str = "2023-10-01") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    days = [d for d in _list_days(sym) if start <= d <= end]
    if not days:
        return {"symbol": sym, "status": "no_days"}
    parts = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for r in ex.map(lambda d: _day(sym, d), days):
            if r is not None and not r.empty:
                parts.append(r)
    if not parts:
        return {"symbol": sym, "status": "empty"}
    m = pd.concat(parts).sort_index()
    m = m[~m.index.duplicated()]
    h = m.resample("1h").last()
    h = h.rename(columns={"sum_open_interest_value": "oi_usd",
                          "sum_open_interest": "oi_contracts",
                          "count_long_short_ratio": "lsr_accounts",
                          "sum_taker_long_short_vol_ratio": "taker_lsr"})
    p = OUT / f"{sym}.parquet"
    h.to_parquet(p)
    return {"symbol": sym, "status": "ok", "rows": len(h),
            "start": str(h.index[0]), "end": str(h.index[-1]),
            "oi_usd_median": float(h["oi_usd"].median())}


UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
            "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT", "BCHUSDT",
            "MATICUSDT", "UNIUSDT", "AAVEUSDT", "FILUSDT", "XLMUSDT", "ICPUSDT",
            "NEARUSDT", "ATOMUSDT", "ETCUSDT", "TRXUSDT", "APTUSDT"]


if __name__ == "__main__":
    import sys
    syms = sys.argv[1:] or UNIVERSE
    res = []
    for s in syms:
        r = build_symbol(s)
        res.append(r)
        print(f"  {s:10} {r.get('status'):8} rows={r.get('rows','-')} "
              f"{r.get('start','')[:10]}..{r.get('end','')[:10]}", flush=True)
    json.dump(res, open(OUT / "_index.json", "w"), indent=1)
    print("done ->", OUT)
