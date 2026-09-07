"""Front-quarterly futures price for BTC/ETH -> continuous annualised basis vs perp.

F064 term-structure input. Binance vision delivery klines (BTCUSDT_YYMMDD).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "quarterly"
CONTRACTS = {
    "BTCUSDT": ["210326", "210625", "210924", "211231", "220325", "220624", "220930",
                "221230", "230331", "230630", "230929", "231229", "240329", "240628",
                "240927", "241227", "250328", "250627", "250926"],
    "ETHUSDT": ["210326", "210625", "210924", "211231", "220325", "220624", "220930",
                "221230", "230331", "230630", "230929", "231229", "240329", "240628",
                "240927", "241227", "250328", "250627", "250926"],
}


def _expiry(code: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime("20" + code, "%Y%m%d"), tz="UTC")


def _contract_1h(base: str, code: str) -> pd.Series | None:
    from lab.data.binance_vision import klines
    sym = f"{base}_{code}"
    exp = _expiry(code)
    months = [str(p) for p in pd.period_range(exp - pd.Timedelta(days=120), exp, freq="M")]
    try:
        d = klines(sym, months, interval="1h", freq="monthly")
    except Exception:
        return None
    if d is None or d.empty:
        return None
    ts = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    return pd.Series(d["close"].astype(float).values, index=ts).sort_index()


def build(base: str = "BTCUSDT") -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    from lab.data.binance_vision import klines
    # perp close 1h
    allm = [str(p) for p in pd.period_range("2021-01", "2025-06", freq="M")]
    perp = klines(base, allm, interval="1h", freq="monthly")
    perp = pd.Series(perp["close"].astype(float).values,
                     index=pd.to_datetime(perp["open_time"], unit="ms", utc=True)).sort_index()

    rows = []
    for code in CONTRACTS[base]:
        s = _contract_1h(base, code)
        if s is None:
            continue
        exp = _expiry(code)
        # use each contract only while it is the FRONT one: last 90 days before expiry
        s = s[(s.index >= exp - pd.Timedelta(days=90)) & (s.index < exp - pd.Timedelta(days=2))]
        for t, q in s.items():
            p = perp.reindex([t]).iloc[0] if t in perp.index else np.nan
            if not np.isfinite(p) or p <= 0:
                continue
            dte = (exp - t).total_seconds() / 86400.0
            basis = q / p - 1.0
            rows.append({"ts": t, "front_code": code, "quarterly": q, "perp": p,
                         "days_to_expiry": dte,
                         "basis": basis,
                         "basis_ann": basis * (365.0 / max(dte, 1.0))})
    df = pd.DataFrame(rows).drop_duplicates("ts").set_index("ts").sort_index()
    df.to_parquet(OUT / f"{base}.parquet")
    return df


if __name__ == "__main__":
    import sys
    for b in (sys.argv[1:] or ["BTCUSDT", "ETHUSDT"]):
        d = build(b)
        print(f"{b}: {len(d)} hours  {d.index[0].date()}..{d.index[-1].date()}  "
              f"basis_ann median {d['basis_ann'].median():.3f}  "
              f"range [{d['basis_ann'].quantile(.05):.2f}, {d['basis_ann'].quantile(.95):.2f}]")
