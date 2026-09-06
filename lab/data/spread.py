"""Real transaction-cost measurement.

Two independent routes, cross-checked:
  1. OHLC spread estimators (published, computed from data we already have):
       - Corwin & Schultz (2012) high-low estimator
       - Abdi & Ranaldo (2017) close-high-low estimator
  2. Realised half-spread from a SMALL aggTrades sample (ground truth):
       effective half-spread ~ mean( |p_trade - midpoint| ) proxied by the
       trade-price reversal, and quoted spread from consecutive opposite-sign
       trades.

slippage_bps for venue.yaml = validated half-spread + a linear impact term
sized from typical clip vs bar volume.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lab.data.binance_vision as bv  # noqa


def corwin_schultz(high: np.ndarray, low: np.ndarray) -> float:
    """CS (2012) 2-day high-low spread estimator. Returns proportional spread."""
    h = np.asarray(high, float)
    l = np.asarray(low, float)
    m = (h > 0) & (l > 0)
    h, l = h[m], l[m]
    if len(h) < 3:
        return np.nan
    beta = (np.log(h[1:] / l[1:]) ** 2 + np.log(h[:-1] / l[:-1]) ** 2)
    hi2 = np.maximum(h[1:], h[:-1])
    lo2 = np.minimum(l[1:], l[:-1])
    gamma = np.log(hi2 / lo2) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    s = s[np.isfinite(s)]
    s = s[s > 0]  # CS: negative estimates set to 0 / dropped
    return float(np.median(s)) if len(s) else 0.0


def abdi_ranaldo(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> float:
    """AR (2017) estimator: s^2 = 4 * E[(c_t - eta_t)(c_t - eta_{t+1})],
    eta = (high+low)/2 in logs. Returns proportional spread."""
    c = np.log(np.asarray(close, float))
    e = (np.log(np.asarray(high, float)) + np.log(np.asarray(low, float))) / 2
    if len(c) < 3:
        return np.nan
    s2 = 4 * (c[1:-1] - e[1:-1]) * (c[1:-1] - e[2:])
    s2 = s2[np.isfinite(s2)]
    val = np.mean(s2)
    return float(np.sqrt(val)) if val > 0 else 0.0


def realised_spread_aggtrades(symbol: str, days: list[str]) -> dict:
    df = bv.agg_trades(symbol, days)
    if df.empty:
        return {"symbol": symbol, "status": "no_data"}
    p = df["price"].to_numpy(float)
    side = np.where(df["is_buyer_maker"].to_numpy(), -1, 1)  # buyer_maker => sell aggressor
    # quoted spread proxy: mean absolute price change between a buy-aggressor and
    # the next sell-aggressor (bounded by the true quoted spread).
    dp = np.abs(np.diff(p)) / p[:-1]
    flip = np.diff(side) != 0
    quoted = float(np.median(dp[flip])) if flip.any() else np.nan
    # effective half-spread: Roll-style from trade-price autocovariance
    r = np.diff(np.log(p))
    cov = np.mean((r[1:] * r[:-1]))
    roll = float(2 * np.sqrt(-cov)) if cov < 0 else 0.0
    return {
        "symbol": symbol, "status": "ok", "n_trades": int(len(df)),
        "quoted_spread_prop": quoted,
        "roll_effective_spread_prop": roll,
        "half_spread_bps": float(np.nanmean([quoted, roll]) / 2 * 1e4),
    }


if __name__ == "__main__":
    print("spread module ready")
