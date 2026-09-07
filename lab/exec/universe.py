"""Tradeable-universe filter — frozen by lab/reports/UNIVERSE_RULE.md (spec B).

In-sample only, rolling, no lookahead. Thresholds are constants here and are
never tuned against results.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
_CACHE = PROC / "universe_daily.parquet"

# ---- FROZEN thresholds (UNIVERSE_RULE.md) --------------------------------
MAX_HALF_SPREAD_BPS = 1.0
MIN_ADV_USD = 50_000_000.0
MIN_LISTED_DAYS = 90
WINDOW_DAYS = 30
TIER_HS = {1: 0.4, 2: 1.2, 3: 4.0}          # ADV-tier half-spread proxy (bps)
TIER_ADV = {1: 300e6, 2: 30e6}


@lru_cache(maxsize=1)
def _measured_half_spread() -> dict:
    cov: dict[str, float] = {}
    for f in ("aggtrades_micro.json", "spread_measured.json", "spread_extra.json"):
        p = PROC / f
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            if r.get("status") == "ok" and r.get("half_spread_bps") is not None:
                cov.setdefault(r["symbol"], float(r["half_spread_bps"]))
    return cov


def half_spread_bps_of(coin: str, adv_usd: float) -> float:
    m = _measured_half_spread()
    if coin in m:
        return m[coin]
    if not np.isfinite(adv_usd):
        return TIER_HS[2]
    tier = 1 if adv_usd >= TIER_ADV[1] else (2 if adv_usd >= TIER_ADV[2] else 3)
    return TIER_HS[tier]


def _roll_half_spread_bps(close: pd.Series) -> pd.Series:
    """Roll (1984) effective half-spread per UTC day, in bps of mid.
    1e4 * sqrt(max(0, -cov(dp_t, dp_{t-1}))) / mid."""
    df = pd.DataFrame({"p": close.values, "dp": close.diff().values}, index=close.index)
    df["dp_lag"] = df["dp"].shift(1)
    out = {}
    for d, g in df.groupby(df.index.floor("D")):
        g = g.dropna(subset=["dp", "dp_lag"])
        if len(g) < 30:
            continue
        c = np.cov(g["dp"].values, g["dp_lag"].values)[0, 1]
        mid = float(g["p"].median())
        out[d] = 1e4 * np.sqrt(max(0.0, -c)) / mid if mid > 0 else np.nan
    return pd.Series(out).sort_index()


def _build() -> pd.DataFrame:
    """per coin per day: half_spread_bps, adv_usd, days_listed."""
    if _CACHE.exists():
        return pd.read_parquet(_CACHE)
    rows = []
    for f in sorted((PROC / "klines_1m").glob("*.parquet")):
        sym = f.stem
        d = pd.read_parquet(f)
        tcol = "dt" if "dt" in d.columns else "open_time"
        idx = pd.to_datetime(d[tcol], utc=True, errors="coerce",
                             unit=(None if tcol == "dt" else "ms"))
        close = pd.Series(d["close"].astype(float).values, index=idx).dropna()
        qv = pd.Series((d["quote_volume"].astype(float).values
                        if "quote_volume" in d.columns
                        else (d["volume"].astype(float) * d["close"].astype(float)).values),
                       index=idx)
        if close.empty:
            continue
        adv_daily = qv.resample("1D").sum()
        listed0 = close.index.min()
        df = pd.DataFrame({"adv_usd": adv_daily})
        df["adv_30d"] = df["adv_usd"].rolling(WINDOW_DAYS, min_periods=15).median()
        df["hs_30d"] = df["adv_30d"].apply(lambda a: half_spread_bps_of(sym, a))
        df["roll1m_hs_bps"] = _roll_half_spread_bps(close).reindex(adv_daily.index)  # kept for reference
        df["days_listed"] = (df.index - listed0).days
        df["symbol"] = sym
        rows.append(df.reset_index(names="day"))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(_CACHE)
    return out


@lru_cache(maxsize=1)
def _tbl() -> pd.DataFrame:
    return _build().set_index(["symbol", "day"]).sort_index()


def eligible(coin: str, ts) -> bool:
    ts = pd.Timestamp(ts, tz="UTC") if pd.Timestamp(ts).tz is None else pd.Timestamp(ts)
    day = ts.floor("D")
    try:
        sub = _tbl().loc[coin]
    except KeyError:
        return False
    sub = sub[sub.index <= day]
    if sub.empty:
        return False
    r = sub.iloc[-1]
    return bool(r["hs_30d"] <= MAX_HALF_SPREAD_BPS
               and r["adv_30d"] >= MIN_ADV_USD
               and (r["days_listed"] - WINDOW_DAYS) >= MIN_LISTED_DAYS)


def tradeable_at(ts, candidates: list[str] | None = None) -> list[str]:
    tbl = _tbl()
    coins = candidates or sorted({c for c, _ in tbl.index})
    return [c for c in coins if eligible(c, ts)]


def summary(ts=None) -> dict:
    tbl = _tbl().reset_index()
    ts = pd.Timestamp(ts) if ts else tbl["day"].max()
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    coins = sorted(tbl["symbol"].unique())
    inc = tradeable_at(ts, coins)
    return {"as_of": str(ts.date()), "n_in": len(inc), "in": inc,
            "n_out": len(coins) - len(inc),
            "out": [c for c in coins if c not in inc]}


if __name__ == "__main__":
    import sys
    for d in ("2024-01-01", "2024-09-01", "2025-05-01"):
        s = summary(d)
        print(f"{s['as_of']}: {s['n_in']} in / {s['n_out']} out")
        print("   in :", " ".join(s["in"]))
    print(json.dumps(summary(), indent=1))
