"""DIVISION 8 — MARKET REGIME + CROSS-SECTIONAL + CORRELATION.  20 formulas.

Covers user categories 9 (regime), 10 (cross-sectional), 11 (correlation /
intermarket). These are mostly CONDITIONING signals — they say *when* and *which
name*, and they are the natural gates for the ensemble.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from lab.divisions.base import formula, zx, zt, safe_log

D = "D8_REGIME"
F = lambda fid, paper, params, horizon: formula(D, fid, paper, params, horizon,
                                                needs=("close", "quote_volume"))


def _lr(p): return safe_log(p["close"]).diff()


# ---------------------------------------------------------------- regime ---
@F("R01_trend_regime", "Hamilton 1989 ECMA (2-state)", {"n": 720}, "H1")
def trend_regime(p, n=720):
    """market-wide trend state x coin beta -> ride the regime."""
    r = _lr(p); mkt = r.mean(axis=1)
    state = np.sign(mkt.rolling(n, min_periods=n // 2).mean())
    beta = r.rolling(n, min_periods=n // 2).cov(mkt).div(mkt.rolling(n, min_periods=n // 2).var(), axis=0)
    return zx(beta.mul(state, axis=0))


@F("R02_vol_regime_switch", "Ang-Bekaert 2002 RFS", {"n": 168, "m": 1440}, "H1")
def vol_regime_switch(p, n=168, m=1440):
    """momentum in calm regimes, reversal in stressed regimes."""
    r = _lr(p)
    v = r.rolling(n).std()
    hi = (zt(v, m) > 0).astype(float) * 2 - 1
    return zx(-hi * r.rolling(n).sum())


@F("R03_risk_on_off", "Bekaert-Hoerova 2014 JoE", {"n": 336}, "H1")
def risk_on_off(p, n=336):
    """cross-sectional dispersion falling = risk-on = high-beta leads."""
    r = _lr(p)
    disp = r.std(axis=1).rolling(n, min_periods=n // 2).mean()
    on = -np.sign(disp.diff(n))
    beta = r.rolling(n, min_periods=n // 2).cov(r.mean(axis=1)).div(
        r.mean(axis=1).rolling(n, min_periods=n // 2).var(), axis=0)
    return zx(beta.mul(on, axis=0))


@F("R04_cusum", "Page 1954 Biometrika", {"n": 336, "k": 0.5}, "H1")
def cusum(p, n=336, k=0.5):
    """CUSUM change-point statistic on standardised returns."""
    r = _lr(p)
    z = (r - r.rolling(n, min_periods=n // 2).mean()) / r.rolling(n, min_periods=n // 2).std().replace(0, np.nan)
    return zx((z - k).clip(lower=0).rolling(n // 4).sum())


@F("R05_range_regime", "Alexander 1961; Lo-MacKinlay 1988", {"n": 336, "q": 8}, "H1")
def range_regime(p, n=336, q=8):
    """variance ratio < 1 => ranging => fade the last move."""
    r = _lr(p)
    vr = (r.rolling(q).sum().rolling(n).var() / q) / r.rolling(n).var().replace(0, np.nan)
    return zx(-(1.0 - vr).clip(lower=0) * r.rolling(n // 8).sum())


@F("R06_hurst_regime", "Hurst 1951; Peters 1994", {"n": 480}, "H1")
def hurst_regime(p, n=480):
    r = _lr(p)

    def _h(x):
        x = x[np.isfinite(x)]
        if len(x) < 60: return np.nan
        y = np.cumsum(x - x.mean()); s = x.std()
        return np.log((y.max() - y.min()) / s + 1e-12) / np.log(len(x)) if s > 0 else np.nan
    H = r.rolling(n, min_periods=n // 2).apply(_h, raw=True)
    return zx((H - 0.5) * r.rolling(n // 4).sum())


@F("R07_drawdown_state", "Grossman-Zhou 1993 MF", {"n": 720}, "H1")
def drawdown_state(p, n=720):
    """distance below the running max — recovery tendency."""
    c = p["close"]
    dd = c / c.rolling(n, min_periods=n // 2).max() - 1.0
    return zx(dd)


@F("R08_vol_term", "Bollerslev-Tauchen-Zhou 2009 RFS", {"s": 168, "l": 1440}, "H1")
def vol_term(p, s=168, l=1440):
    """short vs long vol = variance term structure slope."""
    r = _lr(p)
    return -zx(r.rolling(s).std() / r.rolling(l, min_periods=l // 2).std().replace(0, np.nan))


# ------------------------------------------------------- cross-sectional ---
@F("R09_rel_strength", "Jegadeesh-Titman 1993 JF", {"k": 336}, "H1")
def rel_strength(p, k=336):
    r = safe_log(p["close"]).diff(k)
    return zx(r.sub(r.mean(axis=1), axis=0))


@F("R10_btc_beta", "Sharpe 1964; Frazzini-Pedersen 2014 JFE", {"n": 720}, "H1")
def btc_beta(p, n=720):
    """betting-against-beta: low-beta names outperform risk-adjusted."""
    r = _lr(p)
    if "BTCUSDT" not in r.columns: return zx(r * np.nan)
    b = r.rolling(n, min_periods=n // 2).cov(r["BTCUSDT"]).div(
        r["BTCUSDT"].rolling(n, min_periods=n // 2).var(), axis=0)
    return -zx(b)


@F("R11_btc_leadlag", "Lo-MacKinlay 1990 RFS", {"lag": 1, "n": 336}, "H1")
def btc_leadlag(p, lag=1, n=336):
    """BTC leads alts: use BTC's lagged move x the coin's beta."""
    r = _lr(p)
    if "BTCUSDT" not in r.columns: return zx(r * np.nan)
    b = r.rolling(n, min_periods=n // 2).cov(r["BTCUSDT"]).div(
        r["BTCUSDT"].rolling(n, min_periods=n // 2).var(), axis=0)
    return zx(b.mul(r["BTCUSDT"].shift(lag), axis=0))


@F("R12_pca_residual", "Avellaneda-Lee 2010 QF", {"n": 504, "k": 3}, "H1")
def pca_residual(p, n=504, k=3):
    """s-score on the residual after removing k principal components."""
    r = _lr(p).fillna(0.0)
    R = r.values; T, N = R.shape
    out = np.full((T, N), np.nan)
    for t0 in range(n, T, 168):
        W = R[t0 - n:t0]
        Wc = W - W.mean(0, keepdims=True)
        try:
            _, _, Vt = np.linalg.svd(Wc, full_matrices=False)
        except Exception:
            continue
        Pm = Vt[:k].T @ Vt[:k]
        blk = slice(t0, min(t0 + 168, T))
        Rc = R[blk] - R[blk].mean(axis=1, keepdims=True)
        out[blk] = Rc - Rc @ Pm.T
    res = pd.DataFrame(out, index=r.index, columns=r.columns)
    cum = res.cumsum()
    return -zx(cum - cum.rolling(n // 2, min_periods=n // 4).mean())


@F("R13_peer_momentum", "Hou 2007 RFS", {"k": 168, "n": 504}, "H1")
def peer_momentum(p, k=168, n=504):
    """correlation-weighted peer return spills over."""
    r = _lr(p)
    rk = r.rolling(k).sum()
    mkt = rk.mean(axis=1)
    return zx(rk.sub(mkt, axis=0).shift(k // 4))


@F("R14_dispersion", "Stivers-Sun 2010 JFQA", {"n": 336}, "H1")
def dispersion(p, n=336):
    """momentum works in high-dispersion regimes, reversal in low."""
    r = _lr(p)
    d = r.std(axis=1).rolling(n, min_periods=n // 2).mean()
    dz = (d - d.rolling(n * 2, min_periods=n).mean()) / d.rolling(n * 2, min_periods=n).std()
    return zx(r.rolling(n // 2).sum().mul(np.sign(dz), axis=0))


@F("R15_corr_regime", "Pollet-Wilson 2010 JFE", {"n": 336}, "H1")
def corr_regime(p, n=336):
    """average pairwise correlation rising = systemic stress = de-risk high beta."""
    r = _lr(p)
    mkt = r.mean(axis=1)
    rho = r.rolling(n, min_periods=n // 2).corr(mkt)
    avg = rho.mean(axis=1)
    return zx(-rho.mul(np.sign(avg.diff(n)), axis=0))


@F("R16_idio_mom", "Blitz-Huij-Martens 2011 JEF", {"k": 336, "n": 504}, "H1")
def idio_mom(p, k=336, n=504):
    """momentum in market-model residuals — lower turnover, cleaner signal."""
    r = _lr(p); mkt = r.mean(axis=1)
    b = r.rolling(n, min_periods=n // 2).cov(mkt).div(mkt.rolling(n, min_periods=n // 2).var(), axis=0)
    resid = r.sub(b.mul(mkt, axis=0))
    return zx(resid.rolling(k).sum() / resid.rolling(n, min_periods=n // 2).std().replace(0, np.nan))


@F("R17_beta_dispersion", "Frazzini-Pedersen 2014 JFE", {"n": 720}, "H1")
def beta_dispersion(p, n=720):
    r = _lr(p); mkt = r.mean(axis=1)
    b = r.rolling(n, min_periods=n // 2).cov(mkt).div(mkt.rolling(n, min_periods=n // 2).var(), axis=0)
    return -zx(b.sub(b.mean(axis=1), axis=0))


@F("R18_corr_to_btc", "Pollet-Wilson 2010", {"n": 504}, "H1")
def corr_to_btc(p, n=504):
    r = _lr(p)
    if "BTCUSDT" not in r.columns: return zx(r * np.nan)
    return -zx(r.rolling(n, min_periods=n // 2).corr(r["BTCUSDT"]))


@F("R19_eigen_exposure", "Avellaneda-Lee 2010", {"n": 504}, "H1")
def eigen_exposure(p, n=504):
    """loading on the first eigen-portfolio; high loading = systematic = fade."""
    r = _lr(p).fillna(0.0)
    R = r.values; T, N = R.shape
    out = np.full((T, N), np.nan)
    for t0 in range(n, T, 168):
        W = R[t0 - n:t0] - R[t0 - n:t0].mean(0, keepdims=True)
        try:
            _, _, Vt = np.linalg.svd(W, full_matrices=False)
        except Exception:
            continue
        out[t0:min(t0 + 168, T)] = Vt[0]
    return -zx(pd.DataFrame(out, index=r.index, columns=r.columns))


@F("R20_lead_lag_net", "Billio-Getmansky-Lo-Pelizzon 2012 JFE", {"n": 336}, "H1")
def lead_lag_net(p, n=336):
    """Granger-style: coins that lag the market catch up."""
    r = _lr(p); mkt = r.mean(axis=1)
    lag_corr = r.rolling(n, min_periods=n // 2).corr(mkt.shift(1))
    now_corr = r.rolling(n, min_periods=n // 2).corr(mkt)
    return zx((lag_corr - now_corr).mul(np.sign(mkt.rolling(n // 8).sum()), axis=0))
