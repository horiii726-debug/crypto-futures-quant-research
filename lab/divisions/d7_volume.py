"""DIVISION 7 — VOLUME.  20 formulas.
Abnormal volume, relative volume, volume-price relation, volume regime.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from lab.divisions.base import formula, zx, zt, safe_log

D = "D7_VOLUME"
F = lambda fid, paper, params, horizon: formula(D, fid, paper, params, horizon,
                                                needs=("close", "quote_volume", "volume", "taker_buy_base"))


def _lr(p): return safe_log(p["close"]).diff()


@F("U01_abnormal_vol", "Gervais-Kaniel-Mingelgrin 2001 JF", {"n": 168, "m": 720}, "H1")
def abnormal_vol(p, n=168, m=720):
    """'high-volume return premium': abnormally high volume predicts higher return."""
    v = p["quote_volume"].rolling(n, min_periods=n // 2).mean()
    return zx(safe_log(v / p["quote_volume"].rolling(m, min_periods=m // 2).mean().replace(0, np.nan)))


@F("U02_turnover_rev", "Lee-Swaminathan 2000 JF", {"n": 168}, "H1")
def turnover_rev(p, n=168):
    """high turnover = crowded = reverts."""
    return -zx(safe_log(p["quote_volume"].rolling(n, min_periods=n // 2).sum()))


@F("U03_vol_trend", "Campbell-Grossman-Wang 1993 QJE", {"fast": 48, "slow": 336}, "H1")
def vol_trend(p, fast=48, slow=336):
    v = p["quote_volume"]
    return zx(v.rolling(fast).mean() / v.rolling(slow).mean().replace(0, np.nan))


@F("U04_vpt", "Granville 1963 (OBV); Campbell-Grossman-Wang 1993", {"n": 168}, "H1")
def vpt(p, n=168):
    """volume-price trend: signed volume accumulation."""
    return zx((np.sign(_lr(p)) * p["quote_volume"]).rolling(n).sum()
              / p["quote_volume"].rolling(n).sum().replace(0, np.nan))


@F("U05_cgw_reversal", "Campbell-Grossman-Wang 1993 QJE", {"k": 24, "n": 336}, "H1")
def cgw_reversal(p, k=24, n=336):
    """returns on HIGH volume revert (liquidity provision), on low volume persist."""
    v = zt(safe_log(p["quote_volume"]), n)
    return -zx(safe_log(p["close"]).diff(k) * v.clip(lower=0))


@F("U06_taker_imb", "Lee-Ready 1991 JF", {"n": 168}, "H1")
def taker_imb(p, n=168):
    """aggressor imbalance from taker-buy base volume."""
    frac = p["taker_buy_base"] / p["volume"].replace(0, np.nan)
    return zx((frac - 0.5).rolling(n, min_periods=n // 2).mean())


@F("U07_taker_imb_fade", "Bouchaud-Farmer-Lillo 2009", {"n": 48}, "H1")
def taker_imb_fade(p, n=48):
    frac = p["taker_buy_base"] / p["volume"].replace(0, np.nan)
    return -zx((frac - 0.5).rolling(n, min_periods=n // 2).mean())


@F("U08_amihud", "Amihud 2002 JFM", {"n": 720}, "H1")
def amihud(p, n=720):
    """illiquidity premium: |r|/$vol."""
    return zx((_lr(p).abs() / p["quote_volume"].replace(0, np.nan)).rolling(n, min_periods=n // 2).mean())


@F("U09_vol_shock", "Gervais-Kaniel-Mingelgrin 2001", {"n": 336}, "H1")
def vol_shock(p, n=336):
    return zx(zt(safe_log(p["quote_volume"]), n))


@F("U10_vol_vol", "Chordia-Subrahmanyam-Anshuman 2001 JFE", {"n": 720}, "H1")
def vol_vol(p, n=720):
    """volatility OF volume is priced negatively."""
    return -zx(safe_log(p["quote_volume"]).rolling(n, min_periods=n // 2).std())


@F("U11_mfi", "Money flow (Chaikin); Lee-Ready sign", {"n": 168}, "H1")
def mfi(p, n=168):
    tp_ = (p["high"] + p["low"] + p["close"]) / 3
    mf = tp_ * p["volume"]
    pos = mf.where(tp_.diff() > 0, 0.0).rolling(n).sum()
    neg = mf.where(tp_.diff() < 0, 0.0).rolling(n).sum()
    return -zx(pos / (pos + neg).replace(0, np.nan))


@F("U12_vwap_dev", "Berkowitz-Logue-Noser 1988 JF", {"n": 168}, "H1")
def vwap_dev(p, n=168):
    vw = (p["close"] * p["quote_volume"]).rolling(n).sum() / p["quote_volume"].rolling(n).sum().replace(0, np.nan)
    return -zx(p["close"] / vw - 1.0)


@F("U13_illiq_shock", "Acharya-Pedersen 2005 JFE", {"n": 336}, "H1")
def illiq_shock(p, n=336):
    il = _lr(p).abs() / p["quote_volume"].replace(0, np.nan)
    return -zx(zt(safe_log(il), n))


@F("U14_vol_price_corr", "Karpoff 1987 JFQA", {"n": 336}, "H1")
def vol_price_corr(p, n=336):
    r = _lr(p).abs()
    v = safe_log(p["quote_volume"])
    return -zx(r.rolling(n, min_periods=n // 2).corr(v))


@F("U15_dollar_vol", "Brennan-Chordia-Subrahmanyam 1998 JFE", {"n": 720}, "H1")
def dollar_vol(p, n=720):
    return -zx(safe_log(p["quote_volume"].rolling(n, min_periods=n // 2).mean()))


@F("U16_vol_regime", "Ane-Geman 2000 JF", {"n": 168, "m": 1440}, "H1")
def vol_regime(p, n=168, m=1440):
    v = p["quote_volume"].rolling(n).sum()
    return zx(v / v.rolling(m, min_periods=m // 2).median().replace(0, np.nan))


@F("U17_signed_vol_mom", "Chordia-Roll-Subrahmanyam 2002 JFE", {"k": 72}, "H1")
def signed_vol_mom(p, k=72):
    frac = p["taker_buy_base"] / p["volume"].replace(0, np.nan)
    return zx(((2 * frac - 1) * p["quote_volume"]).rolling(k).sum()
              / p["quote_volume"].rolling(k).sum().replace(0, np.nan))


@F("U18_vol_breakout", "Gervais-Kaniel-Mingelgrin 2001", {"n": 168, "m": 720}, "H1")
def vol_breakout(p, n=168, m=720):
    v = p["quote_volume"].rolling(n).sum()
    hi = v.rolling(m, min_periods=m // 2).max()
    return zx((v / hi.replace(0, np.nan)) * np.sign(safe_log(p["close"]).diff(n)))


@F("U19_liquidity_beta", "Pastor-Stambaugh 2003 JPE", {"n": 720}, "H1")
def liquidity_beta(p, n=720):
    """sensitivity of a coin's return to market-wide liquidity innovations."""
    il = (_lr(p).abs() / p["quote_volume"].replace(0, np.nan))
    mkt_il = il.mean(axis=1)
    b = _lr(p).rolling(n, min_periods=n // 2).cov(mkt_il).div(
        mkt_il.rolling(n, min_periods=n // 2).var(), axis=0)
    return -zx(b)


@F("U20_kyle_from_vol", "Kyle 1985 ECMA", {"n": 336}, "H1")
def kyle_from_vol(p, n=336):
    """lambda = |r| / signed dollar volume, rolling regression slope proxy."""
    sv = (2 * p["taker_buy_base"] / p["volume"].replace(0, np.nan) - 1) * p["quote_volume"]
    lam = _lr(p).rolling(n, min_periods=n // 2).cov(sv).div(
        sv.rolling(n, min_periods=n // 2).var(), axis=0)
    return -zx(safe_log(lam.abs()))
