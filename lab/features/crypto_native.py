"""M4 crypto-native formula library — F060-F070.

Each is point-in-time. Cross-sectional signals return a (time x coin) frame of
z-scores where higher = expected out-performer. `data` dict carries whatever a
formula needs: close, quote_volume, oi_usd, funding (annualised), funding_perp,
funding_quarterly, liq_long, liq_short, oi_total.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _zx(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def _z(df, n):
    return ((df - df.rolling(n, min_periods=max(5, n // 2)).mean())
            / df.rolling(n, min_periods=max(5, n // 2)).std().replace(0, np.nan)).clip(-4, 4)


# --------------------------------------------------------------------------- #
# F060 — leverage ratio (CANDIDATE-001 baseline)
def f060_leverage_ratio(d, n=168):
    lev = d["oi_usd"] * d["close"] / d["quote_volume"].rolling(n, min_periods=n // 2).mean().replace(0, np.nan)
    return -_zx(np.log(lev.replace(0, np.nan)))


# F061 — OI momentum / divergence (4-quadrant)
def f061_oi_divergence(d, k=24):
    doi = np.sign(d["oi_usd"].pct_change(k))
    dpx = np.sign(np.log(d["close"]).diff(k))
    mag = d["oi_usd"].pct_change(k).abs()
    # OI+ P+ = long build (continuation) ; OI+ P- = short build (squeeze up) ;
    # OI- P+ = short cover (weak, fade) ; OI- P- = long liq (bounce)
    quad = (doi * dpx)                       # +1 aligned (build), -1 opposed (unwind)
    return _zx(quad * mag * np.sign(dpx))    # build in the trend direction -> continue


# F062 — funding level (carry): short the high-funding names
def f062_funding_level(d, k=1):
    f = d["funding"].rolling(max(k, 1), min_periods=1).mean()
    return -_zx(f)


# F063 — funding momentum (persistence of the CHANGE, not the level)
def f063_funding_momentum(d, k=24, trail=72):
    f = d["funding"]
    trend = f.rolling(trail, min_periods=trail // 2).mean()
    chg = f - trend
    return _zx(chg.rolling(k, min_periods=1).mean())          # positive & rising funding persists


# F064 — funding basis / term structure  (perp vs quarterly)
def f064_term_structure(d, k=8):
    # slope = quarterly annualised basis - perp annualised funding
    if "funding_quarterly_ann" not in d or "funding_perp_ann" not in d:
        return None
    slope = d["funding_quarterly_ann"] - d["funding_perp_ann"]
    return -_zx(slope.rolling(k, min_periods=1).mean())       # steep contango -> perp cheap -> long perp


# F065 — cross-venue funding divergence
def f065_xvenue_funding(d, k=6):
    parts = [d[c] for c in ("funding_binance", "funding_bybit", "funding_okx") if c in d]
    if len(parts) < 2:
        return None
    spread = parts[0] - sum(parts[1:]) / (len(parts) - 1)
    return -_zx(spread.rolling(k, min_periods=1).mean())


# F066 — liquidation cascade aftermath (post-cascade reversal, 24-72h)
def f066_liq_cascade(d, n=336, hold_hint=48):
    if "liq_total_usd" not in d or "oi_usd" not in d:
        return None
    intensity = d["liq_total_usd"] / d["oi_usd"].replace(0, np.nan)
    spike = _z(np.log1p(intensity), n)
    dpx = np.sign(np.log(d["close"]).diff(6))
    return -_zx(spike.clip(lower=0) * dpx)                    # big liq + recent down -> bounce


# F067 — liquidation imbalance
def f067_liq_imbalance(d, k=12):
    if "liq_long_usd" not in d or "liq_short_usd" not in d:
        return None
    tot = (d["liq_long_usd"] + d["liq_short_usd"]).replace(0, np.nan)
    imb = (d["liq_long_usd"] - d["liq_short_usd"]) / tot       # >0 => longs forced out => oversold
    return _zx(imb.rolling(k, min_periods=1).mean())


# F068 — estimated liquidation magnet (OI-weighted distance to a leverage cluster)
def f068_liq_magnet(d, lev=20.0, k=24):
    # crude: coins whose price is close to a 1/lev move from a recent OI-build anchor
    anchor = d["close"].rolling(k, min_periods=k // 2).mean()
    dist = (d["close"] / anchor - 1.0).abs()
    near = (1.0 / lev - dist).clip(lower=0)                    # within the liq band
    build = d["oi_usd"].pct_change(k).clip(lower=0)
    return _zx(near * build * np.sign(np.log(d["close"]).diff(k)))  # magnet pull in trend dir


# F070 — basis carry (spot-perp), annualised
def f070_basis_carry(d, k=3):
    f_ann = d["funding"] * 365 * 3 if d["funding"].abs().median() < 0.01 else d["funding"]
    return -_zx(f_ann.rolling(k, min_periods=1).mean())


REGISTRY = {
    "F060_leverage_ratio":   (f060_leverage_ratio, dict(n=168), "H1", ["oi_usd", "close", "quote_volume"]),
    "F061_oi_divergence":    (f061_oi_divergence, dict(k=24), "H4", ["oi_usd", "close"]),
    "F062_funding_level":    (f062_funding_level, dict(k=1), "H4", ["funding"]),
    "F063_funding_momentum": (f063_funding_momentum, dict(k=24, trail=72), "H4", ["funding"]),
    "F064_term_structure":   (f064_term_structure, dict(k=8), "D1", ["funding_quarterly_ann", "funding_perp_ann"]),
    "F065_xvenue_funding":   (f065_xvenue_funding, dict(k=6), "H4", ["funding_binance", "funding_bybit"]),
    "F066_liq_cascade":      (f066_liq_cascade, dict(n=336), "H4", ["liq_total_usd", "oi_usd", "close"]),
    "F067_liq_imbalance":    (f067_liq_imbalance, dict(k=12), "H4", ["liq_long_usd", "liq_short_usd"]),
    "F068_liq_magnet":       (f068_liq_magnet, dict(lev=20.0, k=24), "H4", ["close", "oi_usd"]),
    "F070_basis_carry":      (f070_basis_carry, dict(k=3), "D1", ["funding"]),
}
