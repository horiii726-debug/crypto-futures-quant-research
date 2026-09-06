"""Point-in-time tradable universe with survivorship (R10).

Two passes:
  1. cheap daily bars for EVERY USDT-M perp that ever listed (REST klines),
     -> monthly rank by trailing 30d USDT volume.
  2. the research universe = union of each month's top-N. Delisted names that
     were top-N in their time stay in (that is the whole point).

The monthly membership table is written to the ledger `universe` table and
LOCKED (a sha) before any research runs.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "data" / "raw" / "_spec"
PROC = ROOT / "data" / "processed"
FAPI = "https://fapi.binance.com/fapi/v1/klines"


def _get(url, timeout=30, retries=4):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "crypto-lab/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa
            last = e
            time.sleep(1.0 + i)
    raise RuntimeError(f"{url}: {last}")


def daily_bars_rest(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    out, cur = [], start_ms
    while cur < end_ms:
        url = f"{FAPI}?symbol={symbol}&interval=1d&startTime={cur}&limit=1500"
        rows = json.loads(_get(url))
        if not rows:
            break
        out += rows
        nxt = rows[-1][0] + 86_400_000
        if nxt <= cur:
            break
        cur = nxt
        if len(rows) < 1500:
            break
        time.sleep(0.12)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out, columns=["open_time", "open", "high", "low", "close",
                                    "volume", "close_time", "quote_volume", "trades",
                                    "tbb", "tbq", "ig"])
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = df["open_time"].astype("int64")
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["dt", "open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]


def collect_daily(symbols: list[str], start: str, end: str,
                  cache: Path | None = None) -> dict[str, pd.DataFrame]:
    cache = cache or (PROC / "daily")
    cache.mkdir(parents=True, exist_ok=True)
    s_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    e_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    res = {}
    for i, sym in enumerate(symbols):
        f = cache / f"{sym}.parquet"
        if f.exists():
            res[sym] = pd.read_parquet(f)
            continue
        try:
            df = daily_bars_rest(sym, s_ms, e_ms)
        except RuntimeError:
            df = pd.DataFrame()
        if len(df):
            df.to_parquet(f)
            res[sym] = df
        if (i + 1) % 50 == 0:
            print(f"  daily {i+1}/{len(symbols)}")
    return res


def monthly_universe(daily: dict[str, pd.DataFrame], top_n: int = 50,
                     min_days_listed: int = 21) -> pd.DataFrame:
    """For each month M: rank symbols by trailing 30d quote volume as of the
    last day of M-1, take top_n that have >= min_days_listed of history."""
    panel = []
    for sym, df in daily.items():
        d = df[["dt", "quote_volume"]].copy()
        d["symbol"] = sym
        panel.append(d)
    alld = pd.concat(panel, ignore_index=True).sort_values("dt")
    alld["month"] = alld["dt"].dt.to_period("M")
    months = sorted(alld["month"].unique())[1:]  # need a prior month for the rank
    rows = []
    for m in months:
        asof = (m.start_time.tz_localize("UTC"))
        window_lo = asof - pd.Timedelta(days=30)
        w = alld[(alld["dt"] >= window_lo) & (alld["dt"] < asof)]
        vol = w.groupby("symbol")["quote_volume"].sum()
        cnt = w.groupby("symbol")["quote_volume"].count()
        elig = vol[cnt >= min_days_listed].sort_values(ascending=False)
        for rank, (sym, v) in enumerate(elig.head(top_n).items(), 1):
            rows.append({"month": str(m), "symbol": sym, "rank": rank,
                         "trailing_30d_quote_vol": float(v)})
    return pd.DataFrame(rows)


def lock_universe(um: pd.DataFrame, all_symbols_usdt: list[str], ledger) -> dict:
    import hashlib
    blob = um.sort_values(["month", "rank"]).to_csv(index=False)
    sha = hashlib.sha256(blob.encode()).hexdigest()
    members = set(um["symbol"])
    # currently-listed set for delisting tags
    try:
        ei = json.load(open(SPEC / "exchangeInfo.json"))
        live = {s["symbol"] for s in ei["symbols"] if s["status"] == "TRADING"}
    except Exception:
        live = set()
    for _, r in um.iterrows():
        status = "in" if r["symbol"] in live else "delisted"
        ledger.add_universe_row(r["month"], r["symbol"], status,
                                reason=f"rank{int(r['rank'])} 30d_vol={r['trailing_30d_quote_vol']:.0f}")
    (PROC / "universe_monthly.csv").write_text(blob)
    return {
        "sha256": sha,
        "n_months": um["month"].nunique(),
        "n_distinct_symbols": len(members),
        "n_delisted_in_universe": len(members - live),
        "delisted_members": sorted(members - live),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    alls = json.load(open(SPEC / "all_symbols_ever.json"))
    usdt = [s for s in alls if s.endswith("USDT")]
    print(f"{len(usdt)} USDT-M perps ever; collecting daily bars...")
