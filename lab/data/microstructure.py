"""bookDepth + bookTicker readers (data.binance.vision).

bookDepth: ~1-minute snapshots of cumulative depth at percentage bands
  (-20..-1, +1..+20) from mid. columns: timestamp, percentage, depth, notional.
  Available ~2023-01 .. 2024-05 for USDT-M perps. Small files.

bookTicker: every best-bid/ask update. Huge. We only ever pull a few DAILY
  files to calibrate the queue term of the fill model.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from lab.data.binance_vision import fetch_file, NotFound, _read_zip_csv

ROOT = Path(__file__).resolve().parents[2]

BOOKDEPTH_COLS = ["timestamp", "percentage", "depth", "notional"]
BOOKTICKER_COLS = ["update_id", "best_bid_price", "best_bid_qty",
                   "best_ask_price", "best_ask_qty", "transaction_time", "event_time"]


def book_depth(symbol: str, days: list[str]) -> pd.DataFrame:
    frames = []
    for d in days:
        key = f"data/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{d}.zip"
        try:
            blob = fetch_file(key, verify=False)
        except (NotFound, RuntimeError):
            continue
        df = _read_zip_csv(blob, BOOKDEPTH_COLS)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=BOOKDEPTH_COLS)
    df = pd.concat(frames, ignore_index=True)
    df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce")
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df["notional"] = pd.to_numeric(df["notional"], errors="coerce")
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.dropna(subset=["ts"]).sort_values(["ts", "percentage"]).reset_index(drop=True)


def depth_features(bd: pd.DataFrame) -> dict:
    """Aggregate book shape stats from a bookDepth sample.

    percentage is signed distance from mid in %, so |percentage| ~ price
    offset. depth is cumulative base qty out to that band; notional cumulative
    quote value. We use the +1% / -1% notional as 'near-touch liquidity' and
    the ratio of near vs far as book slope.
    """
    if bd.empty:
        return {"status": "no_data"}
    piv = bd.pivot_table(index="ts", columns="percentage", values="notional", aggfunc="last")
    cols = piv.columns
    near_bid = -1 if -1 in cols else min([c for c in cols if c < 0], default=np.nan)
    near_ask = 1 if 1 in cols else min([c for c in cols if c > 0], default=np.nan)
    far_bid = -5 if -5 in cols else min(cols)
    far_ask = 5 if 5 in cols else max(cols)
    bid1 = piv[near_bid] if near_bid in piv else pd.Series(dtype=float)
    ask1 = piv[near_ask] if near_ask in piv else pd.Series(dtype=float)
    imb = (bid1 - ask1) / (bid1 + ask1)
    slope_bid = (piv[near_bid] / piv[far_bid]).replace([np.inf, -np.inf], np.nan) if near_bid in piv and far_bid in piv else pd.Series(dtype=float)
    return {
        "status": "ok",
        "n_snapshots": int(piv.shape[0]),
        "near_touch_notional_bid_usd_median": float(np.nanmedian(bid1)) if len(bid1) else np.nan,
        "near_touch_notional_ask_usd_median": float(np.nanmedian(ask1)) if len(ask1) else np.nan,
        "depth_imbalance_std": float(np.nanstd(imb)) if len(imb) else np.nan,
        "book_slope_near_far_median": float(np.nanmedian(slope_bid)) if len(slope_bid) else np.nan,
        "pct_bands": sorted(float(c) for c in cols),
    }


def book_ticker_daily(symbol: str, days: list[str]) -> pd.DataFrame:
    frames = []
    for d in days:
        key = f"data/futures/um/daily/bookTicker/{symbol}/{symbol}-bookTicker-{d}.zip"
        try:
            blob = fetch_file(key, verify=False)
        except (NotFound, RuntimeError):
            continue
        frames.append(_read_zip_csv(blob, BOOKTICKER_COLS))
    if not frames:
        return pd.DataFrame(columns=BOOKTICKER_COLS)
    df = pd.concat(frames, ignore_index=True)
    for c in ["best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["transaction_time"] = pd.to_numeric(df["transaction_time"], errors="coerce").astype("int64")
    if (df["transaction_time"] > 2e15).any():
        df.loc[df["transaction_time"] > 2e15, "transaction_time"] //= 1000
    df["mid"] = (df["best_bid_price"] + df["best_ask_price"]) / 2
    df["spread_bps"] = (df["best_ask_price"] - df["best_bid_price"]) / df["mid"] * 1e4
    return df.sort_values("transaction_time").reset_index(drop=True)


def bbo_stats(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {"status": "no_data"}
    s = bt["spread_bps"].replace([np.inf, -np.inf], np.nan).dropna()
    s = s[(s > 0) & (s < 200)]
    dt = np.diff(bt["transaction_time"].values) / 1000.0
    dt = dt[(dt >= 0) & (dt < 60)]
    return {
        "status": "ok",
        "n_updates": int(len(bt)),
        "spread_bps_median": float(s.median()),
        "spread_bps_p25": float(s.quantile(0.25)),
        "spread_bps_p75": float(s.quantile(0.75)),
        "half_spread_bps_median": float(s.median() / 2),
        "quote_update_hz": float(1.0 / np.mean(dt)) if len(dt) else np.nan,
        "best_bid_qty_notional_median": float((bt["best_bid_qty"] * bt["best_bid_price"]).median()),
        "time_at_touch_p50_sec": float(np.median(dt)) if len(dt) else np.nan,
    }


if __name__ == "__main__":
    print("microstructure module ready")
