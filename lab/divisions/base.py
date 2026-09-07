"""Division framework — shared panel loader + formula decorator.

A DIVISION is a family of ~20 formulas from the literature that share a data
requirement and an economic theme. Each formula is registered with its paper
reference, default parameters (author's, never tuned) and native horizon.

Every formula returns a (time x coin) DataFrame of cross-sectional scores where
HIGHER = expected out-performer. Point-in-time, no lookahead.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

# ---------------------------------------------------------------- registry --
DIVISIONS: dict[str, dict] = {}


def formula(division: str, fid: str, paper: str, params: dict, horizon: str,
            needs: tuple[str, ...] = ("close",)):
    """Register one formula in a division."""
    def deco(fn):
        DIVISIONS.setdefault(division, {})[fid] = {
            "fn": fn, "paper": paper, "params": params,
            "horizon": horizon, "needs": needs, "name": fn.__name__,
        }
        return fn
    return deco


# ------------------------------------------------------------------ maths --
def zx(df: pd.DataFrame) -> pd.DataFrame:
    """cross-sectional z-score (the standard signal normalisation)."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def zt(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """time-series z-score over a trailing window, clipped."""
    m = df.rolling(n, min_periods=max(5, n // 2)).mean()
    s = df.rolling(n, min_periods=max(5, n // 2)).std().replace(0, np.nan)
    return ((df - m) / s).clip(-4, 4)


def safe_log(df):
    return np.log(df.where(df > 0))


def rank_pct(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def winsor(df: pd.DataFrame, lo=0.01, hi=0.99) -> pd.DataFrame:
    ql = df.quantile(lo, axis=1)
    qh = df.quantile(hi, axis=1)
    return df.clip(lower=ql, upper=qh, axis=0)


# ------------------------------------------------------------------ panel --
_CACHE: dict = {}


def _stack(dirname: str, col: str, index, cols, mult=None) -> pd.DataFrame:
    d = {}
    for c in cols:
        p = PROC / dirname / f"{c}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        s = df.set_index("dt")[col] if "dt" in df.columns else df[col]
        s.index = pd.to_datetime(s.index, utc=True)
        d[c] = s
    out = pd.DataFrame(d).reindex(index).ffill(limit=8)
    return out.reindex(columns=cols)


def load_panel(tf: str = "H1") -> dict:
    """Full multi-source panel at a timeframe. Keys:
    open high low close volume quote_volume taker_buy_base
    oi_usd funding premium mark_close index_close
    (DOM/tape are separate, see load_micro)."""
    if tf in _CACHE:
        return _CACHE[tf]
    hours = {"H1": 1, "H4": 4, "D1": 24}[tf]

    frames = {}
    for name in ("close", "high", "low", "volume", "quote_volume", "taker_buy_base"):
        parts = []
        for part in ("train", "valid"):
            p = ROOT / "data" / part / "prio50_2y" / "1h" / f"{name}.parquet"
            if p.exists():
                parts.append(pd.read_parquet(p))
        if parts:
            frames[name] = pd.concat(parts).sort_index()
    C1 = frames["close"]
    # crypto perps trade 24/7 with no session gap, so open_t == close_{t-1}.
    # (Yang-Zhang's overnight term is then identically zero, which is correct
    #  for a continuous market — YZ degenerates to Rogers-Satchell + c2c.)
    frames["open"] = C1.shift(1)
    idx, cols = C1.index, list(C1.columns)

    frames["oi_usd"] = _stack("bybit_oi", "oi", idx, cols) * C1
    frames["funding"] = _stack("basis", "funding", idx, cols)
    frames["premium"] = _stack("basis", "premium", idx, cols)
    frames["mark_close"] = _stack("basis", "mark_close", idx, cols)
    frames["index_close"] = _stack("basis", "index_close", idx, cols)

    # survivorship: mask everything where price is missing
    for k in frames:
        if k != "close":
            frames[k] = frames[k].where(C1.notna())

    if hours > 1:
        agg = {"open": "first", "high": "max", "low": "min", "close": "last",
               "volume": "sum", "quote_volume": "sum", "taker_buy_base": "sum",
               "oi_usd": "last", "funding": "sum", "premium": "last",
               "mark_close": "last", "index_close": "last"}
        r = {}
        for k, v in frames.items():
            r[k] = v.resample(f"{hours}h", label="right", closed="right").agg(agg.get(k, "last"))
        frames = r
    frames["_tf"] = tf
    frames["_bar_hours"] = hours
    _CACHE[tf] = frames
    return frames


_MICRO_CACHE: dict = {}


def load_micro(bar: int = 15) -> dict:
    """DOM (bookDepth) + TAPE (aggTrades) bar panels, aligned to a common index.
    bar in {5, 15} minutes. Returns dict of (time x coin) frames."""
    key = f"micro{bar}"
    if key in _MICRO_CACHE:
        return _MICRO_CACHE[key]
    book_dir, tape_dir = PROC / "bookdepth_bars", PROC / "tape"
    bsyms = {p.stem.split("_")[0] for p in book_dir.glob(f"*_{bar}m.parquet")}
    tsyms = {p.stem.split("_")[0] for p in tape_dir.glob(f"*_{bar}m.parquet")}
    syms = sorted(bsyms | tsyms)

    book_cols = ["depth_imb_1pct", "depth_imb_5pct", "imb_vol", "book_slope",
                 "total_depth", "microprice_tilt", "depth_withdraw", "n_snap"]
    tape_cols = ["close", "open", "high", "low", "notional", "n_trades", "ofi_usd",
                 "buy_frac", "aggr_imb", "vwap", "kyle_lambda", "roll_spread_bps",
                 "trade_sign_ac1", "big_trade_imb", "realised_vol"]

    bk, tp = {}, {}
    for s in syms:
        pb = book_dir / f"{s}_{bar}m.parquet"
        if pb.exists():
            d = pd.read_parquet(pb)
            d.index = pd.to_datetime(d.index, utc=True)
            bk[s] = d
        pt = tape_dir / f"{s}_{bar}m.parquet"
        if pt.exists():
            d = pd.read_parquet(pt)
            d.index = pd.to_datetime(d.index, utc=True)
            tp[s] = d

    idx = None
    for d in list(bk.values()) + list(tp.values()):
        idx = d.index if idx is None else idx.union(d.index)
    if idx is None:
        return {}
    idx = idx.sort_values()

    out = {}
    for c in book_cols:
        out[c] = pd.DataFrame({s: bk[s][c].reindex(idx) for s in bk if c in bk[s].columns})
    for c in tape_cols:
        out[c] = pd.DataFrame({s: tp[s][c].reindex(idx) for s in tp if c in tp[s].columns})
    common = sorted(set().union(*[set(v.columns) for v in out.values() if not v.empty]))
    for k in out:
        out[k] = out[k].reindex(columns=common)
    out["_bar_min"] = bar
    out["_bar_hours"] = bar / 60.0
    _MICRO_CACHE[key] = out
    return out
