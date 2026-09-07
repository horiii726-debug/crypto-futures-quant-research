"""Kalman filters — dynamic hedge ratio, signal smoothing, realtime z-score (spec E).

Q/R is ONE parameter. Default Q/R = 1e-5. Never grid.
All strictly causal (filter, not smoother).
"""
from __future__ import annotations

import numpy as np
from numba import njit


# --------------------------------------------------------------------------- #
#  E1 — dynamic hedge ratio  y_t = beta_t * x_t + v_t ,  beta_t random walk    #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _kf_hedge(y, x, q_over_r, r_var):
    n = y.size
    beta = 0.0
    P = 1.0
    q = q_over_r * r_var
    b_out = np.empty(n)
    resid = np.empty(n)
    S_out = np.empty(n)
    for t in range(n):
        P += q                                   # predict
        xt = x[t]
        e = y[t] - beta * xt                      # innovation
        S = xt * xt * P + r_var
        if S <= 0.0:
            b_out[t] = beta; resid[t] = e; S_out[t] = r_var
            continue
        K = P * xt / S
        beta += K * e
        P = (1.0 - K * xt) * P
        b_out[t] = beta
        resid[t] = e
        S_out[t] = S
    return b_out, resid, S_out


def hedge_ratio(y, x, q_over_r: float = 1e-5, r_var: float | None = None) -> dict:
    """Return the filtered beta path, the innovation (= spread), its variance,
    and the normalised innovation z-score (E3)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if r_var is None:
        r_var = float(np.nanvar(np.diff(y))) or 1.0
    beta, resid, S = _kf_hedge(np.nan_to_num(y), np.nan_to_num(x), q_over_r, r_var)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = resid / np.sqrt(S)
    return {"beta": beta, "spread": resid, "innov_var": S, "z": z}


# --------------------------------------------------------------------------- #
#  E2 — local-level smoother  (Kalman replacement for a moving average)        #
#      x_t = x_{t-1} + w ,  y_t = x_t + v                                      #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _kf_local_level(y, q_over_r, r_var):
    n = y.size
    xhat = y[0]
    P = 1.0
    q = q_over_r * r_var
    out = np.empty(n)
    resid = np.empty(n)
    S_out = np.empty(n)
    for t in range(n):
        P += q
        e = y[t] - xhat
        S = P + r_var
        K = P / S
        xhat += K * e
        P = (1.0 - K) * P
        out[t] = xhat
        resid[t] = e
        S_out[t] = S
    return out, resid, S_out


def local_level(y, q_over_r: float = 1e-5, r_var: float | None = None) -> dict:
    """Adaptive-lag smoother. Bigger Q/R -> follows faster (less lag, more noise)."""
    y = np.asarray(y, float)
    if r_var is None:
        r_var = float(np.nanvar(np.diff(y))) or 1.0
    lvl, resid, S = _kf_local_level(np.nan_to_num(y, nan=float(np.nanmean(y))), q_over_r, r_var)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = resid / np.sqrt(S)
    return {"level": lvl, "resid": resid, "z": z}


# --------------------------------------------------------------------------- #
#  E3 — normalised-innovation z-score for gating (NOT direction)              #
# --------------------------------------------------------------------------- #
def innovation_z(y, q_over_r: float = 1e-5) -> np.ndarray:
    """z_t = e_t / sqrt(S_t) from the local-level model — a z-score that already
    accounts for state uncertainty. Use to GATE execution, not to pick a side."""
    return local_level(y, q_over_r=q_over_r)["z"]


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n = 4000
    x = np.cumsum(rng.normal(0, 0.01, n)) + 5.0
    beta_true = 1.0 + 0.3 * np.sin(np.arange(n) / 800.0)     # slowly-varying
    y = beta_true * x + rng.normal(0, 0.02, n)
    r = hedge_ratio(y, x, q_over_r=1e-5)
    err = np.abs(r["beta"][500:] - beta_true[500:]).mean()
    print(f"E1 dynamic hedge: mean |beta_hat - beta_true| after warmup = {err:.3f}")
    # local level vs MA
    sig = np.cumsum(rng.normal(0, 0.05, n))
    noisy = sig + rng.normal(0, 0.3, n)
    ll = local_level(noisy, q_over_r=1e-3)["level"]
    import pandas as pd
    ma = pd.Series(noisy).rolling(20).mean().values
    print(f"E2 tracking error  Kalman {np.nanmean((ll-sig)**2):.4f}  vs  MA20 {np.nanmean((ma-sig)**2):.4f}")
    z = innovation_z(noisy, q_over_r=1e-3)
    print(f"E3 innovation z: mean {np.nanmean(z):.3f}  std {np.nanstd(z):.3f}  (std ~1 = calibrated)")
