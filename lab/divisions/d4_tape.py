"""DIVISION 4 — TAPE / ORDER FLOW.  20 formulas.

Data: Binance aggTrades -> bar features (OFI, signed volume, VPIN, Kyle lambda,
Roll spread, trade-sign autocorrelation, big-trade imbalance, realised vol).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab.divisions.base import formula, zx, zt, safe_log

D = "D4_TAPE"
F = lambda fid, paper, params, horizon: formula(D, fid, paper, params, horizon,
                                                needs=("ofi_usd", "close"))


@F("O01_ofi", "Cont-Kukanov-Stoikov 2014 JFinEc", {"k": 3}, "M15")
def ofi(p, k=3):
    """order-flow imbalance, the canonical microstructure predictor."""
    return zx(p["ofi_usd"].rolling(k, min_periods=1).sum()
              / p["notional"].rolling(k, min_periods=1).sum().replace(0, np.nan))


@F("O02_ofi_reversal", "Bouchaud-Farmer-Lillo 2009", {"k": 1}, "M15")
def ofi_reversal(p, k=1):
    return -zx(p["ofi_usd"].rolling(k, min_periods=1).sum()
               / p["notional"].rolling(k, min_periods=1).sum().replace(0, np.nan))


@F("O03_cum_delta", "Easley-O'Hara 1987 JFE", {"k": 12}, "M15")
def cum_delta(p, k=12):
    d = (2 * p["buy_frac"] - 1.0) * p["notional"]
    return zx(d.rolling(k, min_periods=1).sum() / p["notional"].rolling(k, min_periods=1).sum().replace(0, np.nan))


@F("O04_aggr_imb", "Lee-Ready 1991 JF", {"k": 3}, "M15")
def aggr_imb(p, k=3):
    return zx(p["aggr_imb"].rolling(k, min_periods=1).mean())


@F("O05_vpin", "Easley-Lopez de Prado-O'Hara 2012 RFS", {"n": 48}, "M15")
def vpin(p, n=48):
    """flow toxicity; high VPIN -> adverse selection -> underperform."""
    v = (2 * p["buy_frac"] - 1.0).abs()
    return -zx(v.rolling(n, min_periods=n // 2).mean())


@F("O06_kyle_lambda", "Kyle 1985 ECMA", {"n": 48}, "M15")
def kyle_lambda(p, n=48):
    """low price-impact (liquid) names outperform risk-adjusted."""
    return -zx(safe_log(p["kyle_lambda"].abs()).rolling(n, min_periods=n // 2).mean())


@F("O07_roll_spread", "Roll 1984 JF", {"n": 48}, "M15")
def roll_spread(p, n=48):
    return -zx(p["roll_spread_bps"].rolling(n, min_periods=n // 2).mean())


@F("O08_sign_ac", "Lillo-Farmer 2004 SNDE", {"n": 48}, "M15")
def sign_ac(p, n=48):
    """long-memory of trade signs = persistent order splitting -> follow."""
    return zx(p["trade_sign_ac1"].rolling(n, min_periods=n // 2).mean()
              * np.sign(p["ofi_usd"].rolling(n, min_periods=1).sum()))


@F("O09_big_trade", "Barclay-Warner 1993 JFE", {"k": 6}, "M15")
def big_trade(p, k=6):
    """stealth-trading: mid-size prints carry the information."""
    return zx(p["big_trade_imb"].rolling(k, min_periods=1).mean())


@F("O10_big_trade_fade", "Chan-Lakonishok 1995 JF", {"k": 3}, "M15")
def big_trade_fade(p, k=3):
    return -zx(p["big_trade_imb"].rolling(k, min_periods=1).mean())


@F("O11_intensity", "Engle-Russell 1998 ECMA (ACD)", {"k": 12, "n": 96}, "M15")
def intensity(p, k=12, n=96):
    """trade-arrival intensity shock."""
    return -zx(zt(safe_log(p["n_trades"]), n).rolling(k, min_periods=1).mean())


@F("O12_hawkes_proxy", "Bacry-Muzy 2014 QF", {"halflife": 8}, "M15")
def hawkes_proxy(p, halflife=8):
    """exponentially-decayed signed flow = Hawkes intensity difference
    lambda^buy - lambda^sell in closed form."""
    s = p["ofi_usd"] / p["notional"].replace(0, np.nan)
    return zx(s.ewm(halflife=halflife).mean())


@F("O13_branching", "Filimonov-Sornette 2012 PRE", {"n": 96}, "M15")
def branching(p, n=96):
    """branching ratio n = alpha/beta -> endogeneity; high = reflexive/unstable."""
    return -zx(p["trade_sign_ac1"].rolling(n, min_periods=n // 2).mean())


@F("O14_propagator", "Bouchaud-Gefen-Potters-Wyart 2004 QF", {"lags": 12}, "M15")
def propagator(p, lags=12):
    """transient impact G(l) ~ l^-0.5 weighted signed flow."""
    s = p["ofi_usd"] / p["notional"].replace(0, np.nan)
    w = np.array([(i + 1) ** -0.5 for i in range(lags)])
    w = w / w.sum()
    acc = sum(w[i] * s.shift(i) for i in range(lags))
    return zx(acc)


@F("O15_vwap_tilt", "Berkowitz-Logue-Noser 1988 JF", {"k": 3}, "M15")
def vwap_tilt(p, k=3):
    return zx(((p["close"] - p["vwap"]) / p["close"]).rolling(k, min_periods=1).mean())


@F("O16_vwap_revert", "Madhavan-Smidt 1991 JFE", {"k": 6}, "M15")
def vwap_revert(p, k=6):
    return -zx(((p["close"] - p["vwap"]) / p["close"]).rolling(k, min_periods=1).mean())


@F("O17_amihud_bar", "Amihud 2002 JFM", {"n": 96}, "M15")
def amihud_bar(p, n=96):
    r = safe_log(p["close"]).diff().abs()
    return -zx((r / p["notional"].replace(0, np.nan)).rolling(n, min_periods=n // 2).mean())


@F("O18_realised_vol", "Andersen-Bollerslev-Diebold-Labys 2003 ECMA", {"n": 96}, "M15")
def realised_vol(p, n=96):
    return -zx(p["realised_vol"].rolling(n, min_periods=n // 2).mean())


@F("O19_ofi_x_vol", "Cont-Cucuringu-Zhang 2021 SSRN", {"k": 3, "n": 96}, "M15")
def ofi_x_vol(p, k=3, n=96):
    """OFI scaled by its own volatility = standardised flow shock."""
    s = p["ofi_usd"] / p["notional"].replace(0, np.nan)
    return zx(s.rolling(k, min_periods=1).mean()
              / s.rolling(n, min_periods=n // 2).std().replace(0, np.nan))


@F("O20_absorption", "Weber-Rosenow 2005 QF", {"k": 6}, "M15")
def absorption(p, k=6):
    """large flow that does NOT move price = absorption by a hidden bid."""
    flow = (p["ofi_usd"] / p["notional"].replace(0, np.nan)).rolling(k, min_periods=1).sum()
    move = safe_log(p["close"]).diff(k)
    return zx(flow.abs() * np.sign(flow) * (1.0 - move.abs() / move.abs().rolling(96).mean().replace(0, np.nan)).clip(-1, 1))
