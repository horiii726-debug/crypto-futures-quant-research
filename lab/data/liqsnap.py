"""Binance coin-M liquidationSnapshot -> hourly liquidation-pressure bars.

Files: data/raw/data/futures/cm/daily/liquidationSnapshot/{BTCUSD_PERP,ETHUSD_PERP}/
Row: time, side, order_type, time_in_force, original_quantity, price, average_price,
     order_status, last_fill_quantity, accumulated_fill_quantity

`side` is the side of the FORCED order:
  BUY  -> a short position was liquidated (forced buy-back)  -> upward pressure
  SELL -> a long  position was liquidated (forced sell)      -> downward pressure

Coin-M contracts are USD-denominated ($100 for BTCUSD_PERP, $10 for ETHUSD_PERP),
so notional_usd = last_fill_quantity * contract_usd.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LIQ = ROOT / "data" / "raw" / "data" / "futures" / "cm" / "daily" / "liquidationSnapshot"
CONTRACT_USD = {"BTCUSD_PERP": 100.0, "ETHUSD_PERP": 10.0}
OUT = ROOT / "data" / "processed" / "liqsnap"


def _day(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        raw = z.read(name)
    df = pd.read_csv(io.BytesIO(raw))
    # de-dup: the feed repeats each fill row twice
    df = df.drop_duplicates()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["qty"] = df["last_fill_quantity"].astype(float)
    df["px"] = df["average_price"].astype(float)
    return df[["ts", "side", "qty", "px"]]


def hourly_bars(symbol: str) -> pd.DataFrame:
    d = LIQ / symbol
    cusd = CONTRACT_USD[symbol]
    parts = []
    for f in sorted(d.glob(f"{symbol}-liquidationSnapshot-*.zip")):
        x = _day(f)
        if not x.empty:
            parts.append(x)
    if not parts:
        return pd.DataFrame()
    ev = pd.concat(parts, ignore_index=True).sort_values("ts")
    ev["notional"] = ev["qty"] * cusd
    ev["sell_notional"] = np.where(ev["side"] == "SELL", ev["notional"], 0.0)  # long liq (down)
    ev["buy_notional"] = np.where(ev["side"] == "BUY", ev["notional"], 0.0)     # short liq (up)
    ev = ev.set_index("ts")
    g = ev.resample("1h")
    bars = pd.DataFrame({
        "liq_sell_usd": g["sell_notional"].sum(),
        "liq_buy_usd": g["buy_notional"].sum(),
        "liq_n": g["qty"].count(),
        "liq_vwap": (g.apply(lambda s: (s["px"] * s["qty"]).sum() / s["qty"].sum()
                             if s["qty"].sum() else np.nan)),
    })
    bars["liq_total_usd"] = bars["liq_sell_usd"] + bars["liq_buy_usd"]
    bars["liq_imb"] = ((bars["liq_sell_usd"] - bars["liq_buy_usd"])
                       / bars["liq_total_usd"].replace(0, np.nan))
    return bars


def build(symbols=("BTCUSD_PERP", "ETHUSD_PERP")) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for s in symbols:
        b = hourly_bars(s)
        if b.empty:
            continue
        p = OUT / f"{s}.parquet"
        b.to_parquet(p)
        res[s] = {"path": str(p), "rows": len(b), "start": str(b.index[0]),
                  "end": str(b.index[-1]),
                  "total_liq_usd": float(b["liq_total_usd"].sum())}
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(build(), indent=1))
