"""Volatility estimators — for SIZING and timing, never direction (spec D).

All causal / in-sample. Default parameters; no grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_H_PER_YEAR = 365 * 24


# --------------------------------------------------------------------------- #
#  D1 — Yang-Zhang (most efficient OHLC estimator)                            #
# --------------------------------------------------------------------------- #
def yang_zhang(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
               n: int = 30) -> pd.Series:
    """Rolling Yang-Zhang variance (per bar). n = window in bars."""
    lo_c1 = np.log(o / c.shift(1))          # overnight
    lo_co = np.log(c / o)                    # open-to-close
    rs = (np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o))   # Rogers-Satchell
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    v_o = lo_c1.rolling(n).var(ddof=1)
    v_c = lo_co.rolling(n).var(ddof=1)
    v_rs = rs.rolling(n).mean()
    return (v_o + k * v_c + (1 - k) * v_rs).clip(lower=0)


def yang_zhang_sigma_ann(o, h, l, c, n: int = 30, bars_per_year: float = TRADING_H_PER_YEAR):
    return np.sqrt(yang_zhang(o, h, l, c, n) * bars_per_year)


# --------------------------------------------------------------------------- #
#  D2 — EWMA (fast, real-time)                                                #
# --------------------------------------------------------------------------- #
def ewma_var(returns: pd.Series, lam: float = 0.94, warmup: int = 50) -> pd.Series:
    r2 = returns.fillna(0.0) ** 2
    out = np.empty(len(r2))
    s = float(r2.iloc[:warmup].mean()) if len(r2) >= warmup else float(r2.mean() or 1e-8)
    for i, x in enumerate(r2.values):
        s = lam * s + (1 - lam) * x
        out[i] = s
    return pd.Series(out, index=returns.index)


def ewma_sigma_ann(returns, lam: float = 0.94, bars_per_year: float = TRADING_H_PER_YEAR):
    return np.sqrt(ewma_var(returns, lam) * bars_per_year)


# --------------------------------------------------------------------------- #
#  D3 — GJR-GARCH(1,1)  (asymmetry: down-moves raise vol more)                #
# --------------------------------------------------------------------------- #
_GJR_START = dict(omega=1e-6, alpha=0.05, gamma=0.05, beta=0.90)


def gjr_garch_sigma(returns: np.ndarray, *, refit_every: int = 168, min_obs: int = 500,
                    bars_per_year: float = TRADING_H_PER_YEAR) -> np.ndarray:
    """1-step-ahead conditional sigma (per bar, not annualised) from GJR-GARCH(1,1).
    Re-fit by MLE every `refit_every` bars, recursion rolled forward between fits.
    Falls back to a 500-bar rolling std when `arch` is unavailable or a fit fails."""
    r = np.asarray(returns, float)
    r = np.nan_to_num(r) * 100.0                       # arch likes % returns
    n = r.size
    sig = np.full(n, np.nan)
    try:
        from arch import arch_model
        have_arch = True
    except Exception:
        have_arch = False
    params, last = None, None
    for t in range(min_obs, n):
        if have_arch and (last is None or (t - last) >= refit_every):
            try:
                am = arch_model(r[:t], vol="GARCH", p=1, o=1, q=1, dist="normal",
                                rescale=False)
                fit = am.fit(disp="off", show_warning=False,
                             starting_values=None)
                params = fit.params
                last = t
                sig[t] = np.sqrt(fit.forecast(horizon=1, reindex=False).variance.values[-1, 0]) / 100.0
                continue
            except Exception:
                params = None
        if params is not None and {"omega", "alpha[1]", "beta[1]"}.issubset(params.index):
            om, al = params["omega"], params["alpha[1]"]
            be, ga = params["beta[1]"], params.get("gamma[1]", 0.0)
            prev = (sig[t - 1] * 100.0) ** 2 if np.isfinite(sig[t - 1]) else np.nanmean(r[:t] ** 2)
            shock = r[t - 1] ** 2
            ind = 1.0 if r[t - 1] < 0 else 0.0
            sig[t] = np.sqrt(max(om + (al + ga * ind) * shock + be * prev, 1e-10)) / 100.0
        else:
            sig[t] = np.nanstd(r[max(0, t - 500):t]) / 100.0
    return sig


# --------------------------------------------------------------------------- #
#  D4 — vol targeting (sizing)                                                #
# --------------------------------------------------------------------------- #
def vol_target_weight(sigma_ann: pd.Series | np.ndarray, *, target_vol_ann: float = 0.10,
                      max_leverage: float = 3.0) -> pd.Series | np.ndarray:
    s = np.asarray(sigma_ann, float)
    w = np.clip(target_vol_ann / np.where(s > 0, s, np.nan), 0.0, max_leverage)
    w = np.nan_to_num(w, nan=0.0)
    if isinstance(sigma_ann, pd.Series):
        return pd.Series(w, index=sigma_ann.index)
    return w


def apply_vol_target(pnl: pd.Series, sigma_ann: pd.Series, *, target_vol_ann: float = 0.10,
                     max_leverage: float = 3.0, turnover_cap: float | None = None) -> pd.Series:
    """scale a strategy's per-bar pnl by a 1-bar-lagged vol-target weight."""
    w = vol_target_weight(sigma_ann, target_vol_ann=target_vol_ann, max_leverage=max_leverage)
    w = w.reindex(pnl.index).shift(1).fillna(0.0)
    if turnover_cap is not None:
        dw = w.diff().abs()
        w = w.where(dw <= turnover_cap, w.shift(1) + np.sign(w.diff()) * turnover_cap)
    return pnl * w


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 3000
    r = rng.normal(0, 0.01, n)
    r[1500:1600] *= 4                       # a vol spike
    s = pd.Series(r)
    print("EWMA sigma_ann  spike vs calm:",
          round(float(ewma_sigma_ann(s).iloc[1550]), 3), "vs",
          round(float(ewma_sigma_ann(s).iloc[500]), 3))
    g = gjr_garch_sigma(r, refit_every=500, min_obs=400)
    print("GJR sigma  spike vs calm:", round(float(np.nanmean(g[1520:1580])), 4),
          "vs", round(float(np.nanmean(g[450:500])), 4))
    w = vol_target_weight(ewma_sigma_ann(s), target_vol_ann=0.1)
    print("vol-target weight  calm vs spike:", round(float(w.iloc[500]), 2),
          "vs", round(float(w.iloc[1550]), 2), " (should shrink in the spike)")
