"""DIVISION 3 — DOM / ORDER BOOK.  20 formulas.

Data: Binance bookDepth (±1%..±5% cumulative bands, 30s snapshots) resampled to
5m/15m bars. Native horizon is minutes; the F-1 feasibility bound says this is
COST_BOUND as a directional bet, so these are also the canonical inputs for a
market-making quote skew (D8).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab.divisions.base import formula, zx, zt, safe_log

D = "D3_DOM"
F = lambda fid, paper, params, horizon: formula(D, fid, paper, params, horizon,
                                                needs=("depth_imb_1pct", "close"))


@F("B01_qi_follow", "Cartea-Jaimungal-Penalva 2015", {"k": 3}, "M15")
def qi_follow(p, k=3):
    """queue imbalance QI = (Qb-Qa)/(Qb+Qa); follow the pressure."""
    return zx(p["depth_imb_1pct"].rolling(k, min_periods=1).mean())


@F("B02_qi_fade", "Gould-Bonart 2016 JFinEc", {"k": 1}, "M15")
def qi_fade(p, k=1):
    return -zx(p["depth_imb_1pct"].rolling(k, min_periods=1).mean())


@F("B03_qi_deep", "Cont-Kukanov-Stoikov 2014 JFinEc", {"k": 3}, "M15")
def qi_deep(p, k=3):
    """imbalance measured on the ±5% band = slower, less noisy."""
    return zx(p["depth_imb_5pct"].rolling(k, min_periods=1).mean())


@F("B04_microprice", "Stoikov 2018 QF", {"k": 1}, "M15")
def microprice(p, k=1):
    return zx(p["microprice_tilt"].rolling(k, min_periods=1).mean())


@F("B05_micro_revert", "Stoikov 2018; Lehalle-Mounjid 2017", {"k": 6}, "M15")
def micro_revert(p, k=6):
    m = p["microprice_tilt"]
    return -zx(m - m.rolling(k, min_periods=2).mean())


@F("B06_book_slope", "Kalay-Sade-Wohl 2004 JFE", {"k": 3}, "M15")
def book_slope(p, k=3):
    """steeper near-book = thinner top = easier to move -> fade."""
    return -zx(p["book_slope"].rolling(k, min_periods=1).mean())


@F("B07_slope_amplify", "Naes-Skjeltorp 2006 JFM", {"k": 3}, "M15")
def slope_amplify(p, k=3):
    return zx((-p["book_slope"].rolling(k, min_periods=2).mean())
              * np.sign(safe_log(p["close"]).diff(k)))


@F("B08_withdraw_fade", "Hasbrouck-Saar 2009 JFM", {"k": 3}, "M15")
def withdraw_fade(p, k=3):
    """liquidity withdrawal ahead of a move = informed; fade the move."""
    return -zx(p["depth_withdraw"].rolling(k, min_periods=1).mean()
               * np.sign(safe_log(p["close"]).diff(k)))


@F("B09_imbvol_fade", "Cont-Cucuringu-Zhang 2021 SSRN", {"k": 3}, "M15")
def imbvol_fade(p, k=3):
    """volatility of the imbalance x recent move — the strongest raw DOM signal
    found in this project (gross Sharpe 4.3, killed by turnover)."""
    return -zx(p["imb_vol"].rolling(k, min_periods=1).mean()
               * np.sign(safe_log(p["close"]).diff(k)))


@F("B10_depth_total", "Amihud 2002 JFM", {"k": 12}, "M15")
def depth_total(p, k=12):
    """deep book = liquid = lower expected return (illiquidity premium)."""
    return -zx(safe_log(p["total_depth"].rolling(k, min_periods=1).mean()))


@F("B11_depth_shock", "Foucault-Kadan-Kandel 2005 RFS", {"k": 6, "n": 96}, "M15")
def depth_shock(p, k=6, n=96):
    d = safe_log(p["total_depth"])
    return zx(-(d - d.rolling(n, min_periods=n // 2).mean()).rolling(k, min_periods=1).mean())


@F("B12_qi_momentum", "Cont-Kukanov-Stoikov 2014", {"k": 6}, "M15")
def qi_momentum(p, k=6):
    return zx(p["depth_imb_1pct"].diff(k))


@F("B13_qi_extreme", "Gould-Bonart 2016", {"n": 96}, "M15")
def qi_extreme(p, n=96):
    return -zx(zt(p["depth_imb_1pct"], n))


@F("B14_near_far", "Kalay-Sade-Wohl 2004", {"k": 3}, "M15")
def near_far(p, k=3):
    """near-touch vs deep imbalance divergence."""
    return zx((p["depth_imb_1pct"] - p["depth_imb_5pct"]).rolling(k, min_periods=1).mean())


@F("B15_book_pressure", "Cao-Hansch-Wang 2009 JFM", {"k": 3}, "M15")
def book_pressure(p, k=3):
    return zx((p["depth_imb_1pct"] * safe_log(p["total_depth"])).rolling(k, min_periods=1).mean())


@F("B16_resiliency", "Large 2007 JFM", {"k": 12}, "M15")
def resiliency(p, k=12):
    """how fast depth rebuilds after withdrawal -> resilient book = safe to follow."""
    reb = -p["depth_withdraw"].clip(upper=0).rolling(k, min_periods=1).mean()
    return zx(reb * np.sign(safe_log(p["close"]).diff(k)))


@F("B17_imb_vol_level", "Cont-Cucuringu-Zhang 2021", {"k": 12}, "M15")
def imb_vol_level(p, k=12):
    return -zx(p["imb_vol"].rolling(k, min_periods=1).mean())


@F("B18_snap_density", "Hasbrouck-Saar 2013 JFM", {"k": 12}, "M15")
def snap_density(p, k=12):
    """quote-update intensity = HFT activity proxy."""
    return -zx(p["n_snap"].rolling(k, min_periods=1).mean())


@F("B19_micro_x_qi", "Stoikov 2018 + Cont et al 2014", {"k": 3}, "M15")
def micro_x_qi(p, k=3):
    a = p["microprice_tilt"].rolling(k, min_periods=1).mean()
    b = p["depth_imb_1pct"].rolling(k, min_periods=1).mean()
    return zx(a * np.sign(b))


@F("B20_depth_asym_mom", "Cont-Cucuringu-Zhang 2021", {"k": 6, "n": 96}, "M15")
def depth_asym_mom(p, k=6, n=96):
    z = zt(p["depth_imb_5pct"], n)
    return zx(z.diff(k))
