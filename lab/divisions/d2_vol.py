"""DIVISION 2 — VOLATILITY.  20 formulas.

As cross-sectional signals these express the low-vol anomaly, vol-change /
vol-risk-premium effects and jump asymmetry. (The same estimators are re-used
for SIZING in D8; here they are tested as directional tilts.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab.divisions.base import formula, zx, zt, safe_log

D = "D2_VOL"
F = lambda fid, paper, params, horizon: formula(D, fid, paper, params, horizon,
                                                needs=("close", "open", "high", "low", "volume"))


def _lr(p):
    return safe_log(p["close"]).diff()


@F("V01_lowvol", "Ang-Hodrick-Xing-Zhang 2006 JF", {"n": 720}, "H1")
def lowvol(p, n=720):
    """low realised vol out-performs risk-adjusted -> long low vol."""
    return -zx(_lr(p).rolling(n, min_periods=n // 2).std())


@F("V02_yang_zhang", "Yang-Zhang 2000 JB", {"n": 168}, "H1")
def yang_zhang(p, n=168):
    o, h, l, c = p["open"], p["high"], p["low"], p["close"]
    vo = np.log(o / c.shift(1)).rolling(n).var()
    vc = np.log(c / o).rolling(n).var()
    rs = (np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)).rolling(n).mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return -zx(np.sqrt((vo + k * vc + (1 - k) * rs).clip(lower=0)))


@F("V03_parkinson", "Parkinson 1980 JB", {"n": 168}, "H1")
def parkinson(p, n=168):
    hl = np.log(p["high"] / p["low"]) ** 2
    return -zx(np.sqrt(hl.rolling(n).mean() / (4 * np.log(2))))


@F("V04_garman_klass", "Garman-Klass 1980 JB", {"n": 168}, "H1")
def garman_klass(p, n=168):
    hl = 0.5 * np.log(p["high"] / p["low"]) ** 2
    co = (2 * np.log(2) - 1) * np.log(p["close"] / p["open"]) ** 2
    return -zx(np.sqrt((hl - co).rolling(n).mean().clip(lower=0)))


@F("V05_rogers_satchell", "Rogers-Satchell 1991 AAP", {"n": 168}, "H1")
def rogers_satchell(p, n=168):
    rs = (np.log(p["high"] / p["close"]) * np.log(p["high"] / p["open"])
          + np.log(p["low"] / p["close"]) * np.log(p["low"] / p["open"]))
    return -zx(np.sqrt(rs.rolling(n).mean().clip(lower=0)))


@F("V06_ewma", "JP Morgan RiskMetrics 1996", {"lam": 0.94}, "H1")
def ewma(p, lam=0.94):
    r2 = _lr(p) ** 2
    return -zx(np.sqrt(r2.ewm(alpha=1 - lam).mean()))


@F("V07_vol_change", "Bollerslev-Tauchen-Zhou 2009 RFS", {"fast": 48, "slow": 336}, "H1")
def vol_change(p, fast=48, slow=336):
    """vol-of-vol / variance-risk-premium proxy: recent vol vs its own baseline."""
    r = _lr(p)
    return -zx(r.rolling(fast).std() / r.rolling(slow).std().replace(0, np.nan))


@F("V08_har_rv", "Corsi 2009 JFEC", {"d": 24, "w": 120, "m": 528}, "H1")
def har_rv(p, d=24, w=120, m=528):
    """HAR-RV forecast; high forecast vol -> short (low-vol tilt)."""
    r2 = _lr(p) ** 2
    rv_d = r2.rolling(d).sum()
    rv_w = r2.rolling(w).sum() / (w / d)
    rv_m = r2.rolling(m).sum() / (m / d)
    return -zx(np.sqrt((0.4 * rv_d + 0.35 * rv_w + 0.25 * rv_m).clip(lower=0)))


@F("V09_semivar_skew", "Barndorff-Nielsen-Kinnebrock-Shephard 2010", {"n": 336}, "H1")
def semivar_skew(p, n=336):
    """signed jump variation RS+ - RS-; negative jump variation -> future underperf."""
    r = _lr(p)
    rsp = (r.clip(lower=0) ** 2).rolling(n).sum()
    rsm = (r.clip(upper=0) ** 2).rolling(n).sum()
    return zx((rsp - rsm) / (rsp + rsm).replace(0, np.nan))


@F("V10_bipower_jump", "Barndorff-Nielsen-Shephard 2004 JFEC", {"n": 336}, "H1")
def bipower_jump(p, n=336):
    r = _lr(p).abs()
    bv = (np.pi / 2) * (r * r.shift(1)).rolling(n).sum()
    rv = (_lr(p) ** 2).rolling(n).sum()
    return -zx((rv - bv).clip(lower=0) / rv.replace(0, np.nan))


@F("V11_vol_of_vol", "Baltussen-van Bekkum-van der Grient 2018 JFQA", {"n": 168, "m": 720}, "H1")
def vol_of_vol(p, n=168, m=720):
    v = _lr(p).rolling(n).std()
    return -zx(v.rolling(m, min_periods=m // 2).std())


@F("V12_atr_norm", "Wilder 1978", {"n": 48}, "H1")
def atr_norm(p, n=48):
    tr = pd.concat([(p["high"] - p["low"]),
                    (p["high"] - p["close"].shift()).abs(),
                    (p["low"] - p["close"].shift()).abs()]).groupby(level=0).max() \
        if False else (p["high"] - p["low"])
    return -zx(tr.rolling(n).mean() / p["close"])


@F("V13_bb_width", "Bollinger 1980", {"n": 96}, "H1")
def bb_width(p, n=96):
    c = p["close"]
    m = c.rolling(n).mean()
    s = c.rolling(n).std()
    return -zx(2 * s / m.replace(0, np.nan))


@F("V14_keltner_width", "Keltner 1960; Chester 1992", {"n": 96}, "H1")
def keltner_width(p, n=96):
    atr = (p["high"] - p["low"]).rolling(n).mean()
    return -zx(atr / p["close"].ewm(span=n).mean())


@F("V15_chaikin_vol", "Chaikin 1982", {"n": 48}, "H1")
def chaikin_vol(p, n=48):
    hl = (p["high"] - p["low"]).ewm(span=n).mean()
    return -zx(hl / hl.shift(n) - 1.0)


@F("V16_idio_vol", "Ang-Hodrick-Xing-Zhang 2006 JF", {"n": 504}, "H1")
def idio_vol(p, n=504):
    """residual vol after the market (equal-weight) factor."""
    r = _lr(p)
    mkt = r.mean(axis=1)
    beta = r.rolling(n, min_periods=n // 2).cov(mkt).div(
        mkt.rolling(n, min_periods=n // 2).var(), axis=0)
    resid = r.sub(beta.mul(mkt, axis=0))
    return -zx(resid.rolling(n, min_periods=n // 2).std())


@F("V17_downside_beta", "Ang-Chen-Xing 2006 RFS", {"n": 504}, "H1")
def downside_beta(p, n=504):
    r = _lr(p)
    mkt = r.mean(axis=1)
    dn = mkt < 0
    rd = r.where(dn); md = mkt.where(dn)
    b = rd.rolling(n, min_periods=n // 4).cov(md).div(md.rolling(n, min_periods=n // 4).var(), axis=0)
    return -zx(b)


@F("V18_max_lottery", "Bali-Cakici-Whitelaw 2011 JFE", {"n": 168}, "H1")
def max_lottery(p, n=168):
    """lottery demand: recent MAX return is overpriced -> short it."""
    return -zx(_lr(p).rolling(n).max())


@F("V19_ret_skew", "Harvey-Siddique 2000 JF", {"n": 504}, "H1")
def ret_skew(p, n=504):
    return -zx(_lr(p).rolling(n, min_periods=n // 2).skew())


@F("V20_ret_kurt", "Amaya-Christoffersen-Jacobs-Vasquez 2015 JFE", {"n": 504}, "H1")
def ret_kurt(p, n=504):
    return -zx(_lr(p).rolling(n, min_periods=n // 2).kurt())
