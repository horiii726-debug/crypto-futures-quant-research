"""Multi-venue OI / price / funding ingest + normalisation + validation (spec J).

J2 normalisation:
  - one UTC hourly grid
  - OI in USD (Binance metrics `sum_open_interest_value` is already USD;
    Bybit `oi` is contracts -> * mark price)
  - funding annualised (Binance/Bybit 8h -> * 3 * 365)
  - symbol mapping explicit; delisted kept (no survivorship)

J3 validation (before use):
  - Bybit vs Binance OI correlation in the overlap period must be > 0.8
  - gap / outlier / zero-fill checks
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
BIN_OI = PROC / "binance_oi_hist"
BYB_OI = PROC / "bybit_oi"
OUT = PROC / "multi_venue"

# 2021-2022-era universe (coins that existed then). No survivorship filter.
HIST_UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
                 "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT", "BCHUSDT",
                 "MATICUSDT", "UNIUSDT", "AAVEUSDT", "FILUSDT", "XLMUSDT", "ATOMUSDT",
                 "ETCUSDT", "TRXUSDT", "NEARUSDT", "ICPUSDT", "APTUSDT"]


def _binance_price(sym: str, months: list[str]) -> pd.Series | None:
    from lab.data.binance_vision import klines
    try:
        d = klines(sym, months, interval="1h", freq="monthly")
    except Exception:
        return None
    if d is None or d.empty:
        return None
    ts = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    return pd.Series(d["close"].astype(float).values, index=ts).sort_index()


def _binance_qv(sym: str, months: list[str]) -> pd.Series | None:
    from lab.data.binance_vision import klines
    try:
        d = klines(sym, months, interval="1h", freq="monthly")
    except Exception:
        return None
    if d is None or d.empty:
        return None
    ts = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    return pd.Series(d["quote_volume"].astype(float).values, index=ts).sort_index()


def _binance_funding_ann(sym: str, months: list[str]) -> pd.Series | None:
    from lab.data.binance_vision import funding
    try:
        f = funding(sym, months)
    except Exception:
        return None
    if f is None or len(f) == 0 or "dt" not in f.columns:
        return None
    s = pd.Series(f["fundingRate"].astype(float).values, index=pd.to_datetime(f["dt"], utc=True))
    return (s * 3 * 365).sort_index()          # 8h funding -> annualised


def build(start="2021-01", end="2023-09") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    months = [str(p) for p in pd.period_range(start, end, freq="M")]
    idx = pd.date_range(f"{start}-01", f"{end}-28", freq="1h", tz="UTC")

    close, qv, oi_usd = {}, {}, {}
    for s in HIST_UNIVERSE:
        px = _binance_price(s, months)
        if px is None or len(px) < 500:
            continue
        close[s] = px.reindex(idx)
        q = _binance_qv(s, months)
        qv[s] = q.reindex(idx) if q is not None else np.nan
        p = BIN_OI / f"{s}.parquet"
        if p.exists():
            o = pd.read_parquet(p)["oi_usd"]
            o.index = pd.to_datetime(o.index, utc=True)
            oi_usd[s] = o.reindex(idx)
    C = pd.DataFrame(close).sort_index()
    QV = pd.DataFrame(qv).reindex(columns=C.columns)
    OI = pd.DataFrame(oi_usd).reindex(columns=C.columns)

    # zero/negative -> NaN ; forward-fill OI up to 6h ; mask where price missing
    OI = OI.where(OI > 0).ffill(limit=6).where(C.notna())
    QV = QV.where(QV > 0)

    C.to_parquet(OUT / "close_1h.parquet")
    QV.to_parquet(OUT / "quote_volume_1h.parquet")
    OI.to_parquet(OUT / "oi_usd_1h.parquet")

    report = {"start": str(C.index[0]), "end": str(C.index[-1]),
              "n_coins_price": int(C.notna().any().sum()),
              "n_coins_oi": int(OI.notna().any().sum()),
              "coins": list(C.columns)}
    json.dump(report, open(OUT / "_build.json", "w"), indent=1, default=str)
    return report


def validate_vs_bybit() -> dict:
    """J3: Bybit vs Binance OI correlation in the overlap."""
    bo = pd.read_parquet(OUT / "oi_usd_1h.parquet")
    out = {}
    for s in bo.columns:
        p = BYB_OI / f"{s}.parquet"
        if not p.exists():
            continue
        b = pd.read_parquet(p)
        col = "oi" if "oi" in b.columns else b.columns[0]
        b = b.set_index("dt")[col] if "dt" in b.columns else b[col]
        b.index = pd.to_datetime(b.index, utc=True)
        # Bybit oi is contracts for USDT-perp == coin units; * price = USD
        px = pd.read_parquet(OUT / "close_1h.parquet")[s]
        b_usd = (b.reindex(px.index).ffill(limit=6) * px).dropna()
        a = bo[s].reindex(b_usd.index).dropna()
        j = a.index.intersection(b_usd.index)
        if len(j) < 200:
            out[s] = {"status": "thin_overlap", "n": len(j)}
            continue
        r_lvl = float(np.corrcoef(a.loc[j], b_usd.loc[j])[0, 1])
        r_chg = float(np.corrcoef(a.loc[j].pct_change().dropna(),
                                  b_usd.loc[j].pct_change().reindex(a.loc[j].pct_change().dropna().index))[0, 1])
        out[s] = {"n_overlap": len(j), "corr_level": round(r_lvl, 3),
                  "corr_change": round(r_chg, 3),
                  "pass": bool(r_lvl > 0.8)}
    return out


if __name__ == "__main__":
    print("=== build multi-venue panel ===")
    print(json.dumps(build(), indent=1, default=str))
    print("\n=== J3: Bybit vs Binance OI validation ===")
    v = validate_vs_bybit()
    for s, r in v.items():
        print(f"  {s:10} {r}")
    npass = sum(1 for r in v.values() if r.get("pass"))
    print(f"\n{npass}/{len(v)} coins pass corr(OI) > 0.8")
    json.dump(v, open(OUT / "_validation.json", "w"), indent=1)
