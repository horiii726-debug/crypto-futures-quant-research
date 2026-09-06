"""Statistical power for OOS Sharpe / IC tests, with overlap and
autocorrelation corrections.

R5  UNDERPOWERED is a legal verdict. If effective_n < required_n the test
    cannot reject the null and the result is UNDERPOWERED, not a rejection.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def required_n_for_sharpe(sr_target: float, *, alpha: float = 0.05,
                          power: float = 0.80, two_sided: bool = True) -> int:
    """Approx observations needed to detect per-period Sharpe sr_target."""
    if sr_target <= 0:
        return np.iinfo(np.int64).max
    z_a = norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_b = norm.ppf(power)
    n = ((z_a + z_b) / sr_target) ** 2
    return int(np.ceil(n))


def required_n_for_ic(ic_target: float, *, alpha: float = 0.05,
                      power: float = 0.80, two_sided: bool = True) -> int:
    """Observations to detect an information coefficient (corr) of ic_target,
    via Fisher z."""
    r = abs(ic_target)
    if r <= 0 or r >= 1:
        return np.iinfo(np.int64).max
    z_a = norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_b = norm.ppf(power)
    zr = np.arctanh(r)
    n = ((z_a + z_b) / zr) ** 2 + 3
    return int(np.ceil(n))


def autocorr_inflation(x: np.ndarray, max_lag: int | None = None) -> float:
    """Variance inflation factor from serial correlation (Newey-West style).
    effective_n = n / VIF."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 10:
        return 1.0
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return 1.0
    L = max_lag or int(np.ceil(4 * (n / 100.0) ** (2 / 9)))
    L = min(L, n - 2)
    vif = 1.0
    for k in range(1, L + 1):
        rk = np.sum(x[k:] * x[:-k]) / denom
        vif += 2.0 * (1.0 - k / (L + 1.0)) * rk
    return float(max(vif, 1e-6))


def overlap_deflation(label_horizon: int, step: int = 1) -> float:
    """Overlapping h-bar labels sampled every `step` bars share information.
    Independent-sample fraction ~ step / h (clipped to (0,1])."""
    if label_horizon <= 1:
        return 1.0
    return float(min(1.0, step / label_horizon))


def effective_n(returns: np.ndarray, *, label_horizon: int = 1, step: int = 1,
                uniqueness: np.ndarray | None = None) -> dict:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = r.size
    vif = autocorr_inflation(r)
    ov = overlap_deflation(label_horizon, step)
    if uniqueness is not None:
        u = np.asarray(uniqueness, float)
        u = u[np.isfinite(u)]
        ov = float(np.clip(u.mean(), 1e-6, 1.0)) if u.size else ov
    eff = n * ov / vif
    return {
        "n_raw": int(n),
        "vif_autocorr": vif,
        "overlap_fraction": ov,
        "effective_n": float(max(eff, 0.0)),
    }


def power_verdict(returns: np.ndarray, *, sr_target: float, label_horizon: int = 1,
                  step: int = 1, alpha: float = 0.05, power: float = 0.80,
                  uniqueness: np.ndarray | None = None) -> dict:
    eff = effective_n(returns, label_horizon=label_horizon, step=step,
                      uniqueness=uniqueness)
    req = required_n_for_sharpe(sr_target, alpha=alpha, power=power)
    ok = eff["effective_n"] >= req
    return {
        **eff,
        "required_n": int(req) if req < np.iinfo(np.int64).max else None,
        "sr_target": sr_target,
        "powered": bool(ok),
        "verdict": "POWERED" if ok else "UNDERPOWERED",
    }


def _selftest():
    rng = np.random.default_rng(0)
    # strongly autocorrelated series -> VIF > 1, effective_n < n
    e = rng.normal(size=2000)
    ar = np.zeros_like(e)
    for t in range(1, len(e)):
        ar[t] = 0.6 * ar[t - 1] + e[t]
    d = effective_n(ar, label_horizon=10, step=1)
    print(d)
    assert d["vif_autocorr"] > 1.5
    assert d["effective_n"] < d["n_raw"]
    assert required_n_for_sharpe(0.05) > required_n_for_sharpe(0.15)
    v = power_verdict(rng.normal(size=50), sr_target=0.1)
    assert v["verdict"] == "UNDERPOWERED"
    print("power selftest OK")


if __name__ == "__main__":
    _selftest()
