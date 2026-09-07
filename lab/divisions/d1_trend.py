"""DIVISION 1 — TREND / MOMENTUM.  20 formulas.

Cross-sectional: higher score = expected out-performer.
All parameters are the original authors' defaults. No tuning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab.divisions.base import formula, zx, zt, safe_log

D = "D1_TREND"
F = lambda fid, paper, params, horizon: formula(D, fid, paper, params, horizon,
                                                needs=("close", "high", "low", "volume"))


@F("T01_tsmom", "Moskowitz-Ooi-Pedersen 2012 JFE", {"k": 168}, "H1")
def tsmom(p, k=168):
    """r_{t-k:t} scaled by realised vol."""
    r = safe_log(p["close"]).diff(k)
    v = p["close"].pct_change().rolling(k, min_periods=k // 2).std()
    return zx(r / v.replace(0, np.nan))


@F("T02_xsmom", "Jegadeesh-Titman 1993 JF", {"k": 720, "skip": 24}, "H1")
def xsmom(p, k=720, skip=24):
    """classic 12-1: return over k, skipping the most recent `skip` bars."""
    c = safe_log(p["close"])
    return zx(c.shift(skip) - c.shift(k))


@F("T03_macd", "Appel 1979; Baz et al 2015 SSRN", {"fast": 12, "slow": 26, "sig": 9}, "H1")
def macd(p, fast=12, slow=26, sig=9):
    c = p["close"]
    m = c.ewm(span=fast).mean() - c.ewm(span=slow).mean()
    return zx((m - m.ewm(span=sig).mean()) / c)


@F("T04_ma_cross", "Brock-Lakonishok-LeBaron 1992 JF", {"fast": 24, "slow": 168}, "H1")
def ma_cross(p, fast=24, slow=168):
    c = p["close"]
    return zx(c.rolling(fast).mean() / c.rolling(slow).mean() - 1.0)


@F("T05_donchian", "Donchian 1960; Faber 2007 SSRN", {"n": 168}, "H1")
def donchian(p, n=168):
    h = p["high"].rolling(n).max()
    l = p["low"].rolling(n).min()
    return zx((p["close"] - l) / (h - l).replace(0, np.nan) - 0.5)


@F("T06_aroon", "Chande 1995", {"n": 100}, "H1")
def aroon(p, n=100):
    up = p["high"].rolling(n).apply(lambda x: float(np.argmax(x)), raw=True) / n
    dn = p["low"].rolling(n).apply(lambda x: float(np.argmin(x)), raw=True) / n
    return zx(up - dn)


@F("T07_adx_dmi", "Wilder 1978", {"n": 48}, "H1")
def adx_dmi(p, n=48):
    up = p["high"].diff()
    dn = -p["low"].diff()
    plus = up.where((up > dn) & (up > 0), 0.0)
    minus = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([(p["high"] - p["low"]).abs()], axis=0)
    atr = (p["high"] - p["low"]).rolling(n).mean().replace(0, np.nan)
    return zx((plus.rolling(n).mean() - minus.rolling(n).mean()) / atr)


@F("T08_vortex", "Botes-Siepman 2010 TASC", {"n": 48}, "H1")
def vortex(p, n=48):
    tr = (p["high"] - p["low"]).rolling(n).sum().replace(0, np.nan)
    vp = (p["high"] - p["low"].shift(1)).abs().rolling(n).sum()
    vm = (p["low"] - p["high"].shift(1)).abs().rolling(n).sum()
    return zx((vp - vm) / tr)


@F("T09_kama", "Kaufman 1995", {"n": 100, "fast": 2, "slow": 30}, "H1")
def kama(p, n=100, fast=2, slow=30):
    c = p["close"]
    change = (c - c.shift(n)).abs()
    vol = c.diff().abs().rolling(n).sum().replace(0, np.nan)
    er = change / vol                                     # efficiency ratio
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    return zx(sc * np.sign(c - c.shift(n)))


@F("T10_hurst_tilt", "Hurst 1951; Peters 1994", {"n": 240}, "H1")
def hurst_tilt(p, n=240):
    """R/S Hurst; H>0.5 persistent -> tilt with the trend, H<0.5 -> against."""
    r = safe_log(p["close"]).diff()

    def _h(x):
        x = x[np.isfinite(x)]
        if len(x) < 40:
            return np.nan
        y = np.cumsum(x - x.mean())
        R = y.max() - y.min()
        S = x.std()
        return np.log(R / S + 1e-12) / np.log(len(x)) if S > 0 else np.nan
    H = r.rolling(n, min_periods=n // 2).apply(_h, raw=True)
    mom = np.sign(safe_log(p["close"]).diff(n // 2))
    return zx((H - 0.5) * mom)


@F("T11_vr_tilt", "Lo-MacKinlay 1988 RFS", {"n": 240, "q": 8}, "H1")
def vr_tilt(p, n=240, q=8):
    r = safe_log(p["close"]).diff()
    v1 = r.rolling(n).var()
    vq = r.rolling(q).sum().rolling(n).var() / q
    vr = vq / v1.replace(0, np.nan)
    mom = np.sign(r.rolling(n // 4).sum())
    return zx((vr - 1.0) * mom)


@F("T12_tema", "Mulloy 1994 TASC", {"n": 48}, "H1")
def tema(p, n=48):
    c = p["close"]
    e1 = c.ewm(span=n).mean(); e2 = e1.ewm(span=n).mean(); e3 = e2.ewm(span=n).mean()
    t = 3 * e1 - 3 * e2 + e3
    return zx(t / c - 1.0)


@F("T13_ichimoku", "Hosoda 1969", {"conv": 9, "base": 26}, "H1")
def ichimoku(p, conv=9, base=26):
    ch = (p["high"].rolling(conv).max() + p["low"].rolling(conv).min()) / 2
    bl = (p["high"].rolling(base).max() + p["low"].rolling(base).min()) / 2
    return zx((ch - bl) / p["close"])


@F("T14_supertrend", "Olson 2004", {"n": 24, "mult": 3.0}, "H1")
def supertrend(p, n=24, mult=3.0):
    atr = (p["high"] - p["low"]).rolling(n).mean()
    mid = (p["high"] + p["low"]) / 2
    return zx((p["close"] - (mid - mult * atr)) / p["close"])


@F("T15_cmo", "Chande 1994", {"n": 48}, "H1")
def cmo(p, n=48):
    d = p["close"].diff()
    up = d.clip(lower=0).rolling(n).sum()
    dn = (-d).clip(lower=0).rolling(n).sum()
    return zx((up - dn) / (up + dn).replace(0, np.nan))


@F("T16_trix", "Hutson 1983 TASC", {"n": 30}, "H1")
def trix(p, n=30):
    c = safe_log(p["close"])
    e = c.ewm(span=n).mean().ewm(span=n).mean().ewm(span=n).mean()
    return zx(e.diff())


@F("T17_ppo", "Appel 1979", {"fast": 24, "slow": 96}, "H1")
def ppo(p, fast=24, slow=96):
    c = p["close"]
    return zx((c.ewm(span=fast).mean() - c.ewm(span=slow).mean()) / c.ewm(span=slow).mean())


@F("T18_elder_ray", "Elder 1993", {"n": 26}, "H1")
def elder_ray(p, n=26):
    e = p["close"].ewm(span=n).mean()
    return zx(((p["high"] - e) + (p["low"] - e)) / p["close"])


@F("T19_coppock", "Coppock 1962 FAJ", {"a": 294, "b": 168, "w": 60}, "H1")
def coppock(p, a=294, b=168, w=60):
    c = p["close"]
    roc = (c / c.shift(a) - 1) + (c / c.shift(b) - 1)
    return zx(roc.rolling(w).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(),
                                   raw=True))


@F("T20_reg_slope", "Ehlers 2001; Baz et al 2015", {"n": 120}, "H1")
def reg_slope(p, n=120):
    c = safe_log(p["close"])
    x = np.arange(n)
    xc = x - x.mean()
    den = (xc ** 2).sum()

    def _s(y):
        return float(np.dot(xc, y - y.mean()) / den)
    sl = c.rolling(n, min_periods=n).apply(_s, raw=True)
    resid_sd = c.diff().rolling(n).std().replace(0, np.nan)
    return zx(sl / resid_sd)
