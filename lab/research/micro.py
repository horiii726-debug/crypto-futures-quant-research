"""Causal microstructure signals built on the tape bar-feature panel
(lab/data/tape.py). Time-series, per asset. Signal at bar t uses bars <= t.

Each signal returns a pd.Series aligned to the bar index, in units of a target
position in [-1, 1] (already mapped). The micro study handles t+1 execution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _z(s: pd.Series, n: int) -> pd.Series:
    m = s.rolling(n, min_periods=max(5, n // 2)).mean()
    sd = s.rolling(n, min_periods=max(5, n // 2)).std().replace(0, np.nan)
    return ((s - m) / sd).clip(-4, 4)


def _tanh(s: pd.Series, k: float = 1.0) -> pd.Series:
    return np.tanh(k * s)


# ---- OFI / flow ----------------------------------------------------------

def ofi_momentum(df: pd.DataFrame, lb: int = 6, znorm: int = 200) -> pd.Series:
    x = df["ofi_usd"].rolling(lb, min_periods=1).sum()
    return _tanh(_z(x, znorm), 0.7)


def ofi_reversal(df: pd.DataFrame, znorm: int = 200) -> pd.Series:
    return -_tanh(_z(df["ofi_usd"], znorm), 0.7)


def aggr_imb_persist(df: pd.DataFrame, lb: int = 8) -> pd.Series:
    x = df["aggr_imb"].rolling(lb, min_periods=2).mean()
    return _tanh(3.0 * x)


def cum_delta_slope(df: pd.DataFrame, lb: int = 12) -> pd.Series:
    cd = df["cum_delta"]
    slope = (cd - cd.shift(lb)) / (df["notional"].rolling(lb, min_periods=2).sum() + 1e-9)
    return _tanh(2.0 * slope)


def big_trade_follow(df: pd.DataFrame, lb: int = 4) -> pd.Series:
    x = df["big_trade_imb"].rolling(lb, min_periods=1).mean()
    return _tanh(2.0 * x)


# ---- microprice / vwap tilt --------------------------------------------

def vwap_tilt(df: pd.DataFrame, znorm: int = 200) -> pd.Series:
    # vwap below close => buyers lifted late in the bar => short-term up drift
    tilt = (df["close"] - df["vwap"]) / df["close"]
    return _tanh(_z(tilt, znorm), 0.8)


def vwap_reversion(df: pd.DataFrame, znorm: int = 200) -> pd.Series:
    tilt = (df["close"] - df["vwap"]) / df["close"]
    return -_tanh(_z(tilt, znorm), 0.8)


# ---- liquidity / toxicity state (used as filter or fade) -----------------

def kyle_lambda_fade(df: pd.DataFrame, znorm: int = 300) -> pd.Series:
    # rising impact = deteriorating liquidity -> fade the last move
    lam_z = _z(df["kyle_lambda"], znorm)
    return -np.sign(df["ret"].rolling(3, min_periods=1).sum()) * (lam_z > 1).astype(float)


def spread_regime_fade(df: pd.DataFrame, znorm: int = 300) -> pd.Series:
    wide = _z(df["roll_spread_bps"], znorm) > 1.0
    return (-np.sign(df["ret"]) * wide.astype(float)).clip(-1, 1)


# ---- microstructure-trend (sign autocorrelation regime) -----------------

def signac_momentum(df: pd.DataFrame, lb: int = 6, ac_lb: int = 20) -> pd.Series:
    trending = df["trade_sign_ac1"].rolling(ac_lb, min_periods=5).mean() > 0.45
    mom = np.sign(df["ret"].rolling(lb, min_periods=1).sum())
    return (mom * trending.astype(float)).clip(-1, 1)


def intensity_breakout(df: pd.DataFrame, lb: int = 30) -> pd.Series:
    surge = df["intensity"] / df["intensity"].rolling(lb, min_periods=5).mean()
    direction = np.sign(df["ofi_usd"])
    return (_tanh(0.8 * (surge - 1.0)).clip(lower=0) * direction).clip(-1, 1)


# ---- composite (needs its own hypothesis / NEW_HYPOTHESIS) --------------

def ofi_plus_vwap(df: pd.DataFrame, lb: int = 6, znorm: int = 200) -> pd.Series:
    a = _z(df["ofi_usd"].rolling(lb, min_periods=1).sum(), znorm)
    b = _z((df["close"] - df["vwap"]) / df["close"], znorm)
    return _tanh(0.5 * (a + b))


REGISTRY = {
    "ofi_momentum": ofi_momentum,
    "ofi_reversal": ofi_reversal,
    "aggr_imb_persist": aggr_imb_persist,
    "cum_delta_slope": cum_delta_slope,
    "big_trade_follow": big_trade_follow,
    "vwap_tilt": vwap_tilt,
    "vwap_reversion": vwap_reversion,
    "signac_momentum": signac_momentum,
    "intensity_breakout": intensity_breakout,
    "ofi_plus_vwap": ofi_plus_vwap,
}
FAMILY = {
    "ofi_momentum": "F_MICRO_OFI", "ofi_reversal": "F_MICRO_OFI",
    "aggr_imb_persist": "F_MICRO_OFI", "cum_delta_slope": "F_MICRO_OFI",
    "big_trade_follow": "F_MICRO_OFI", "vwap_tilt": "F_MICRO_BOOK",
    "vwap_reversion": "F_MICRO_BOOK", "signac_momentum": "F_MICRO_OFI",
    "intensity_breakout": "F_MICRO_OFI", "ofi_plus_vwap": "F_MICRO_OFI",
}
