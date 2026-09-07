"""DIVISION 5 — OPEN INTEREST / POSITIONING.  20 formulas.

Data: Bybit hourly OI (2y, 41 coins) + Binance metrics OI (2020-09+) x price/volume.
Derivatives-specific information not present in price alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab.divisions.base import formula, zx, zt, safe_log

D = "D5_OI"
F = lambda fid, paper, params, horizon: formula(D, fid, paper, params, horizon,
                                                needs=("oi_usd", "close", "quote_volume"))


@F("I01_leverage_ratio", "Project CANDIDATE-001", {"n": 168}, "H1")
def leverage_ratio(p, n=168):
    """OI notional / avg traded volume = crowding per unit liquidity."""
    lev = p["oi_usd"] / p["quote_volume"].rolling(n, min_periods=n // 2).mean().replace(0, np.nan)
    return -zx(safe_log(lev))


@F("I02_oi_delta_rev", "Hong-Yogo 2012 JFE", {"k": 24}, "H1")
def oi_delta_rev(p, k=24):
    """OI surged -> crowded -> fade."""
    return -zx(safe_log(p["oi_usd"]).diff(k))


@F("I03_oi_delta_mom", "Bessembinder-Chan 1992 JFE", {"k": 24}, "H1")
def oi_delta_mom(p, k=24):
    return zx(safe_log(p["oi_usd"]).diff(k))


@F("I04_oi_price_confirm", "Bessembinder-Seguin 1993 JFQA", {"k": 24}, "H1")
def oi_price_confirm(p, k=24):
    doi = safe_log(p["oi_usd"]).diff(k)
    dpx = safe_log(p["close"]).diff(k)
    return zx(np.sign(doi) * np.sign(dpx) * doi.abs() * np.sign(dpx))


@F("I05_short_cover", "Hong-Yogo 2012 JFE", {"k": 24}, "H1")
def short_cover(p, k=24):
    """price up + OI down = short covering = weak rally -> fade."""
    doi = p["oi_usd"].diff(k)
    dpx = safe_log(p["close"]).diff(k)
    sig = ((dpx > 0) & (doi < 0)).astype(float) - ((dpx < 0) & (doi < 0)).astype(float)
    return -zx(sig * dpx.abs())


@F("I06_oi_extreme", "Kang-Rouwenhorst-Tang 2020 JF", {"n": 720}, "H1")
def oi_extreme(p, n=720):
    return -zx(zt(safe_log(p["oi_usd"]), n))


@F("I07_oi_per_volume", "Project F_OI", {"k": 24}, "H1")
def oi_per_volume(p, k=24):
    build = p["oi_usd"].diff(k).abs() / p["quote_volume"].rolling(k, min_periods=k // 2).sum().replace(0, np.nan)
    return -zx(build)


@F("I08_oi_accel", "Project F_OI", {"k": 12}, "H1")
def oi_accel(p, k=12):
    return -zx(safe_log(p["oi_usd"]).diff(k).diff(k))


@F("I09_oi_vol_ratio", "Bessembinder-Seguin 1993", {"n": 168}, "H1")
def oi_vol_ratio(p, n=168):
    """OI relative to its own history, scaled by turnover."""
    r = p["oi_usd"] / p["quote_volume"].replace(0, np.nan)
    return -zx(zt(safe_log(r), n))


@F("I10_oi_persist", "Moskowitz-Ooi-Pedersen 2012", {"k": 72, "n": 336}, "H1")
def oi_persist(p, k=72, n=336):
    d = safe_log(p["oi_usd"]).diff(k)
    return zx(d.rolling(n, min_periods=n // 2).mean() / d.rolling(n, min_periods=n // 2).std().replace(0, np.nan))


@F("I11_oi_dispersion", "Kang-Rouwenhorst-Tang 2020", {"k": 24}, "H1")
def oi_dispersion(p, k=24):
    """coin's OI change relative to the cross-section's OI change."""
    d = safe_log(p["oi_usd"]).diff(k)
    return -zx(d.sub(d.mean(axis=1), axis=0))


@F("I12_oi_shock", "Cheng-Kirilenko-Xiong 2015 RF", {"n": 336}, "H1")
def oi_shock(p, n=336):
    d = safe_log(p["oi_usd"]).diff()
    return -zx(zt(d, n))


@F("I13_oi_x_vol", "Bessembinder-Seguin 1993", {"n": 336}, "H1")
def oi_x_vol(p, n=336):
    """OI growth in a high-vol regime = fragile positioning."""
    d = safe_log(p["oi_usd"]).diff(24)
    v = safe_log(p["close"]).diff().rolling(n, min_periods=n // 2).std()
    return -zx(d * zt(v, n))


@F("I14_oi_x_funding", "Hong-Yogo 2012; project F_OI", {"k": 24}, "H1")
def oi_x_funding(p, k=24):
    d = safe_log(p["oi_usd"]).diff(k)
    f = p["funding"].rolling(k, min_periods=1).mean()
    return -zx(d * np.sign(f))


@F("I15_oi_notional", "Kang-Rouwenhorst-Tang 2020", {"n": 168}, "H1")
def oi_notional(p, n=168):
    return -zx(safe_log(p["oi_usd"].rolling(n, min_periods=n // 2).mean()))


@F("I16_oi_turnover", "Lee-Swaminathan 2000 JF", {"n": 168}, "H1")
def oi_turnover(p, n=168):
    t = p["quote_volume"].rolling(n, min_periods=n // 2).sum() / p["oi_usd"].replace(0, np.nan)
    return zx(safe_log(t))


@F("I17_long_liq_bounce", "Project F_OI", {"k": 12}, "H1")
def long_liq_bounce(p, k=12):
    """OI down + price down = forced long liquidation = overshoot -> bounce."""
    doi = p["oi_usd"].diff(k)
    dpx = safe_log(p["close"]).diff(k)
    sig = ((dpx < 0) & (doi < 0)).astype(float)
    return zx(sig * dpx.abs())


@F("I18_squeeze_setup", "Project F_OI", {"k": 12}, "H1")
def squeeze_setup(p, k=12):
    """OI up + price down = fresh shorts = squeeze fuel -> long."""
    doi = p["oi_usd"].diff(k)
    dpx = safe_log(p["close"]).diff(k)
    sig = ((dpx < 0) & (doi > 0)).astype(float)
    return zx(sig * dpx.abs())


@F("I19_oi_breadth", "Kang-Rouwenhorst-Tang 2020", {"k": 24, "n": 336}, "H1")
def oi_breadth(p, k=24, n=336):
    d = safe_log(p["oi_usd"]).diff(k)
    breadth = (d > 0).sum(axis=1) / d.notna().sum(axis=1)
    return -zx(d.mul(breadth, axis=0))


@F("I20_oi_meanrev", "Hong-Yogo 2012", {"n": 720}, "H1")
def oi_meanrev(p, n=720):
    l = safe_log(p["oi_usd"])
    return -zx(l - l.rolling(n, min_periods=n // 2).mean())
