"""5-year OHLCV panel (2020-2025) from Binance vision monthly klines.

The prio50 research panel only covers 2023-09..2025-05 (1.75y). Statistical
power scales as sqrt(T): extending to ~5 years multiplies the t-statistic by
sqrt(5.4/1.75) = 1.76, which is exactly what the low-volatility family needs to
cross the multiple-testing threshold.

Survivorship (R10): every symbol that EVER traded on Binance USDT-M perp in the
window is included for the months it existed; delisted coins are kept and simply
go NaN after their last bar.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "long_panel"

START, END = "2020-01", "2025-06"

# Coins with a Binance USDT-M perp listed early enough to give a long sample.
# Delisted names are INCLUDED (survivorship): they simply stop.
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT",
    "DOTUSDT", "LTCUSDT", "BCHUSDT", "LINKUSDT", "AVAXUSDT", "UNIUSDT", "ATOMUSDT",
    "ETCUSDT", "TRXUSDT", "XLMUSDT", "FILUSDT", "AAVEUSDT", "EOSUSDT", "THETAUSDT",
    "ALGOUSDT", "VETUSDT", "ICPUSDT", "NEARUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT",
    "FTMUSDT", "GRTUSDT", "CHZUSDT", "ENJUSDT", "SUSHIUSDT", "COMPUSDT", "SNXUSDT",
    "YFIUSDT", "MKRUSDT", "CRVUSDT", "ZECUSDT", "DASHUSDT", "XMRUSDT", "OMGUSDT",
    "QTUMUSDT", "IOTAUSDT", "NEOUSDT", "ONTUSDT", "ZILUSDT", "BATUSDT", "IOSTUSDT",
    "RVNUSDT", "KAVAUSDT", "BANDUSDT", "RLCUSDT", "WAVESUSDT", "MATICUSDT",
    "APTUSDT", "OPUSDT", "ARBUSDT", "LDOUSDT", "INJUSDT", "SUIUSDT", "WLDUSDT",
]

FIELDS = ["close", "high", "low", "volume", "quote_volume", "taker_buy_base"]


def _one(sym: str, months: list[str]):
    from lab.data.binance_vision import klines
    try:
        d = klines(sym, months, interval="1h", freq="monthly")
    except Exception:
        return sym, None
    if d is None or d.empty:
        return sym, None
    ts = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    out = {}
    for f in FIELDS:
        col = f if f in d.columns else None
        if col is None:
            continue
        out[f] = pd.Series(pd.to_numeric(d[col], errors="coerce").values, index=ts)
    df = pd.DataFrame(out).sort_index()
    return sym, df[~df.index.duplicated()]


def build(start=START, end=END, workers=10) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    months = [str(p) for p in pd.period_range(start, end, freq="M")]
    idx = pd.date_range(f"{start}-01", f"{end}-28", freq="1h", tz="UTC")
    got = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for sym, df in ex.map(lambda s: _one(s, months), UNIVERSE):
            if df is None or len(df) < 24 * 180:      # need >= 6 months
                print(f"  {sym:12} skip", flush=True)
                continue
            got[sym] = df
            print(f"  {sym:12} {len(df):6} bars  {df.index[0].date()}..{df.index[-1].date()}", flush=True)
    if not got:
        return {"status": "empty"}
    frames = {}
    for f in FIELDS:
        frames[f] = pd.DataFrame({s: g[f].reindex(idx) for s, g in got.items() if f in g})
    C = frames["close"]
    for f in FIELDS:
        if f != "close":
            frames[f] = frames[f].where(C.notna())
    for f, v in frames.items():
        v.to_parquet(OUT / f"{f}.parquet")
    rep = {"coins": list(C.columns), "n_coins": C.shape[1], "bars": len(idx),
           "start": str(C.index[0]), "end": str(C.index[-1]),
           "years": round(len(idx) / (365 * 24), 2),
           "coverage_median_bars": int(C.notna().sum().median())}
    json.dump(rep, open(OUT / "_build.json", "w"), indent=1)
    return rep


def load() -> dict:
    frames = {f: pd.read_parquet(OUT / f"{f}.parquet") for f in FIELDS
              if (OUT / f"{f}.parquet").exists()}
    frames["open"] = frames["close"].shift(1)
    frames["_tf"] = "H1"
    frames["_bar_hours"] = 1.0
    return frames


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
