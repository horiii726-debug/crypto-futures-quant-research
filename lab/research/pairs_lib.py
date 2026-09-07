"""RESEARCH ROUND 2 · P3 — numerical toolkit for F_PAIRS (cointegration stat-arb).

All estimation is in-sample / trailing-window only. Nothing here looks forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

# --------------------------------------------------------------------------- #
#  Johansen trace test (statsmodels)                                          #
# --------------------------------------------------------------------------- #
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def johansen_rank1(y: np.ndarray, det_order: int = 0, k_ar_diff: int = 1) -> dict | None:
    """y : (T, 2) price levels (logs). Return the r=1 cointegrating vector iff
    the trace test rejects r=0 at 95% AND fails to reject r=1 at 95%."""
    if y.shape[0] < 120 or not np.isfinite(y).all():
        return None
    try:
        res = coint_johansen(y, det_order, k_ar_diff)
    except Exception:
        return None
    tr = res.lr1                      # trace statistics [r=0, r=1]
    cv = res.cvt[:, 1]                # 95% critical values
    if not (tr[0] > cv[0] and tr[1] <= cv[1]):
        return None
    vec = res.evec[:, 0]
    if abs(vec[0]) < 1e-9:
        return None
    beta = -vec[1] / vec[0]           # y0 = beta * y1 + resid  (normalise y0 coef to 1)
    return {"beta": float(beta), "trace_r0": float(tr[0]), "cv_r0": float(cv[0]),
            "trace_r1": float(tr[1]), "cv_r1": float(cv[1])}


# --------------------------------------------------------------------------- #
#  OU fit on a spread series                                                  #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _ar1(S):
    n = S.size
    x = S[:-1]
    y = S[1:]
    mx = x.mean()
    my = y.mean()
    sxx = 0.0
    sxy = 0.0
    for i in range(n - 1):
        dx = x[i] - mx
        sxx += dx * dx
        sxy += dx * (y[i] - my)
    if sxx <= 0.0:
        return 0.0, 0.0, 0.0
    b = sxy / sxx
    a = my - b * mx
    resid_var = 0.0
    for i in range(n - 1):
        e = y[i] - (a + b * x[i])
        resid_var += e * e
    resid_var /= (n - 2)
    return a, b, resid_var


def ou_fit(S: np.ndarray, dt: float = 1.0) -> dict | None:
    """S_t = a + b S_{t-1} + e  ->  theta = -ln(b)/dt, mu = a/(1-b),
    sigma_eq = std of the stationary distribution."""
    S = np.asarray(S, float)
    S = S[np.isfinite(S)]
    if S.size < 80:
        return None
    a, b, rv = _ar1(S)
    if not (0.0 < b < 1.0) or rv <= 0:
        return None
    theta = -np.log(b) / dt
    mu = a / (1.0 - b)
    sigma_ou = np.sqrt(rv * 2.0 * theta / (1.0 - b * b))
    sigma_eq = sigma_ou / np.sqrt(2.0 * theta)
    half_life = np.log(2.0) / theta
    return {"theta": float(theta), "mu": float(mu), "sigma_eq": float(sigma_eq),
            "half_life_h": float(half_life / dt * dt), "half_life": float(half_life),
            "ar1_b": float(b)}


# --------------------------------------------------------------------------- #
#  Kalman dynamic hedge ratio  (state = [beta, intercept])                    #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def kalman_hedge(y, x, q_over_r, r_var):
    """y_t = beta_t x_t + alpha_t + v ;  [beta,alpha]_t = [beta,alpha]_{t-1} + w
    Q = q_over_r * r_var * I2 ;  returns spread_t = y_t - (beta_t x_t + alpha_t)."""
    n = y.size
    beta = 0.0
    alpha = 0.0
    p11 = 1.0
    p22 = 1.0
    p12 = 0.0
    q = q_over_r * r_var
    spread = np.empty(n)
    betas = np.empty(n)
    for t in range(n):
        # predict: P += Q
        p11 += q
        p22 += q
        # observation h = [x_t, 1]
        xt = x[t]
        yhat = beta * xt + alpha
        resid = y[t] - yhat
        spread[t] = resid
        betas[t] = beta
        # S = h P h' + R
        s = p11 * xt * xt + 2.0 * p12 * xt + p22 + r_var
        if s <= 0.0:
            continue
        k1 = (p11 * xt + p12) / s
        k2 = (p12 * xt + p22) / s
        beta += k1 * resid
        alpha += k2 * resid
        np11 = p11 - k1 * (p11 * xt + p12)
        np12 = p12 - k1 * (p12 * xt + p22)
        np22 = p22 - k2 * (p12 * xt + p22)
        p11, p12, p22 = np11, np12, np22
    return spread, betas


# --------------------------------------------------------------------------- #
#  Avellaneda-Lee PCA residual                                               #
# --------------------------------------------------------------------------- #
def pca_residual_returns(ret: pd.DataFrame, n_factors: int = 3, win: int = 504,
                         refit: int = 168) -> pd.DataFrame:
    """project each coin's return off the top-n_factors PCs of the trailing
    window; keep the residual. Loadings re-estimated every `refit` bars (they
    are stable over hours), then applied to the whole block at once. Causal:
    the projection used from bar t0..t0+refit is estimated only on data < t0."""
    R = ret.fillna(0.0).values
    T, N = R.shape
    out = np.full((T, N), np.nan)
    Rc = R - R.mean(axis=1, keepdims=True)
    for t0 in range(win, T, refit):
        W = R[t0 - win:t0]
        Wc = W - W.mean(0, keepdims=True)
        try:
            _, _, Vt = np.linalg.svd(Wc, full_matrices=False)
        except Exception:
            continue
        F = Vt[:n_factors]
        proj = F.T @ F                                  # (N, N)
        blk = slice(t0, min(t0 + refit, T))
        out[blk] = Rc[blk] - Rc[blk] @ proj.T
    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


# --------------------------------------------------------------------------- #
#  GJR-GARCH(1,1) conditional vol  (for SIZING only, never direction)         #
# --------------------------------------------------------------------------- #
def gjr_sigma(returns: np.ndarray, refit_every: int = 168, min_obs: int = 500) -> np.ndarray:
    """1-step-ahead conditional sigma from GJR-GARCH(1,1), re-fit periodically,
    filtered forward between fits. Causal."""
    from arch import arch_model
    r = np.asarray(returns, float) * 100.0     # arch likes % returns
    n = r.size
    sig = np.full(n, np.nan)
    last = None
    for t in range(min_obs, n):
        if last is None or (t - last) >= refit_every:
            try:
                am = arch_model(r[:t], vol="GARCH", p=1, o=1, q=1, dist="normal", rescale=False)
                fit = am.fit(disp="off", show_warning=False)
                last = t
                fc = fit.forecast(horizon=1, reindex=False)
                sig[t] = np.sqrt(fc.variance.values[-1, 0]) / 100.0
                params = fit.params
            except Exception:
                sig[t] = np.nanstd(r[max(0, t - 500):t]) / 100.0
                params = None
        else:
            # roll the recursion one step with the last fit's params
            if params is not None and {"omega", "alpha[1]", "beta[1]"}.issubset(set(params.index)):
                om = params["omega"]; al = params["alpha[1]"]; be = params["beta[1]"]
                ga = params.get("gamma[1]", 0.0)
                prev_var = (sig[t - 1] * 100.0) ** 2
                shock = r[t - 1] ** 2
                ind = 1.0 if r[t - 1] < 0 else 0.0
                var = om + (al + ga * ind) * shock + be * prev_var
                sig[t] = np.sqrt(max(var, 1e-8)) / 100.0
            else:
                sig[t] = np.nanstd(r[max(0, t - 500):t]) / 100.0
    return sig
