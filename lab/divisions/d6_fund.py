"""DIVISION 6 — FUNDING / BASIS / CARRY.  20 formulas.

Data: Binance perp funding (8h, posted hourly), premium index, mark & index close.
The economic content is the perpetual's cost-of-carry and the positioning it implies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab.divisions.base import formula, zx, zt, safe_log

D = "D6_FUND"
F = lambda fid, paper, params, horizon: formula(D, fid, paper, params, horizon,
                                                needs=("funding", "close"))


@F("C01_carry", "Koijen-Moskowitz-Pedersen-Vrugt 2018 JFE", {"k": 3}, "H4")
def carry(p, k=3):
    """short high-funding (crowded longs pay) — the canonical carry factor."""
    return -zx(p["funding"].rolling(k, min_periods=1).mean())


@F("C02_funding_mom", "Moskowitz-Ooi-Pedersen 2012", {"k": 24}, "H4")
def funding_mom(p, k=24):
    return zx(p["funding"].rolling(k, min_periods=1).mean())


@F("C03_funding_z", "Koijen et al 2018", {"n": 720}, "H4")
def funding_z(p, n=720):
    return -zx(zt(p["funding"], n))


@F("C04_funding_accel", "Project F_FUND", {"k": 8}, "H4")
def funding_accel(p, k=8):
    return -zx(p["funding"].diff(k))


@F("C05_funding_persist", "Project F_CN4 / Baz et al 2015", {"k": 24, "trail": 168}, "H4")
def funding_persist(p, k=24, trail=168):
    f = p["funding"]
    return zx((f - f.rolling(trail, min_periods=trail // 2).mean()).rolling(k, min_periods=1).mean())


@F("C06_premium", "Fama-French 1987 JB (cost of carry)", {"k": 8}, "H4")
def premium(p, k=8):
    return -zx(p["premium"].rolling(k, min_periods=1).mean())


@F("C07_mark_index", "Working 1949 AER (basis)", {"k": 8}, "H4")
def mark_index(p, k=8):
    b = p["mark_close"] / p["index_close"] - 1.0
    return -zx(b.rolling(k, min_periods=1).mean())


@F("C08_basis_mom", "Gorton-Hayashi-Rouwenhorst 2013 RF", {"k": 48}, "H4")
def basis_mom(p, k=48):
    b = p["mark_close"] / p["index_close"] - 1.0
    return zx(b.diff(k))


@F("C09_funding_vol", "Bakshi-Gao-Rossi 2019 MS", {"n": 336}, "H4")
def funding_vol(p, n=336):
    return -zx(p["funding"].rolling(n, min_periods=n // 2).std())


@F("C10_funding_skew", "Harvey-Siddique 2000", {"n": 720}, "H4")
def funding_skew(p, n=720):
    return -zx(p["funding"].rolling(n, min_periods=n // 2).skew())


@F("C11_funding_revert", "Koijen et al 2018", {"n": 336}, "H4")
def funding_revert(p, n=336):
    f = p["funding"]
    return -zx(f - f.rolling(n, min_periods=n // 2).mean())


@F("C12_funding_px_div", "Project F_FUND", {"k": 48}, "H4")
def funding_px_div(p, k=48):
    """funding rising while price falls = longs trapped -> short."""
    return -zx(p["funding"].rolling(k, min_periods=1).mean() * (-np.sign(safe_log(p["close"]).diff(k))))


@F("C13_carry_x_vol", "Koijen et al 2018 (risk-adjusted carry)", {"k": 3, "n": 336}, "H4")
def carry_x_vol(p, k=3, n=336):
    v = safe_log(p["close"]).diff().rolling(n, min_periods=n // 2).std().replace(0, np.nan)
    return -zx(p["funding"].rolling(k, min_periods=1).mean() / v)


@F("C14_funding_breadth", "Project F_FUND", {"k": 24}, "H4")
def funding_breadth(p, k=24):
    f = p["funding"].rolling(k, min_periods=1).mean()
    breadth = (f > 0).sum(axis=1) / f.notna().sum(axis=1)
    return -zx(f.mul(breadth, axis=0))


@F("C15_funding_disp", "Kang-Rouwenhorst-Tang 2020", {"k": 24}, "H4")
def funding_disp(p, k=24):
    f = p["funding"].rolling(k, min_periods=1).mean()
    return -zx(f.sub(f.mean(axis=1), axis=0))


@F("C16_cum_funding", "Koijen et al 2018", {"n": 168}, "H4")
def cum_funding(p, n=168):
    return -zx(p["funding"].rolling(n, min_periods=n // 2).sum())


@F("C17_premium_vol", "Bakshi-Gao-Rossi 2019", {"n": 336}, "H4")
def premium_vol(p, n=336):
    return -zx(p["premium"].rolling(n, min_periods=n // 2).std())


@F("C18_basis_extreme", "Gorton-Hayashi-Rouwenhorst 2013", {"n": 720}, "H4")
def basis_extreme(p, n=720):
    b = p["mark_close"] / p["index_close"] - 1.0
    return -zx(zt(b, n))


@F("C19_funding_regime", "Ang-Bekaert 2002 RFS", {"k": 24, "n": 720}, "H4")
def funding_regime(p, k=24, n=720):
    """carry works differently when the whole market's funding is extreme."""
    f = p["funding"].rolling(k, min_periods=1).mean()
    mkt = f.mean(axis=1)
    reg = zt(mkt.to_frame("m"), n)["m"]
    return -zx(f.mul(np.sign(reg), axis=0))


@F("C20_carry_x_mom", "Asness-Moskowitz-Pedersen 2013 JF", {"k": 3, "m": 336}, "H4")
def carry_x_mom(p, k=3, m=336):
    """value(carry) + momentum combined — the AMP 'everywhere' pair."""
    c = -zx(p["funding"].rolling(k, min_periods=1).mean())
    mo = zx(safe_log(p["close"]).diff(m))
    return zx(c + mo)
