"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

R3  The trial count is READ FROM THE LEDGER, not passed as an argument by an
    agent. `deflated_sharpe(returns, family=...)` calls Ledger.trial_count().
    An explicit `n_trials=` override raises unless `allow_manual_trials=True`
    is set (used only by unit selftests on synthetic inputs).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

SQRT2 = np.sqrt(2.0)
EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray, periods_per_year: float | None = None) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    if periods_per_year:
        sr *= np.sqrt(periods_per_year)
    return float(sr)


def _skew_kurt(r: np.ndarray):
    r = r[np.isfinite(r)]
    n = r.size
    m = r.mean()
    s = r.std(ddof=0)
    if s == 0 or n < 3:
        return 0.0, 3.0
    g1 = np.mean(((r - m) / s) ** 3)
    g2 = np.mean(((r - m) / s) ** 4)
    return float(g1), float(g2)


def expected_max_sharpe(n_trials: int, var_trial_sr: float) -> float:
    """E[max SR] across n_trials iid trials with variance var_trial_sr (SR^2).

    Bailey-LdP approximation using the expected value of the max of n standard
    normals.
    """
    n = max(int(n_trials), 1)
    if n == 1:
        return 0.0
    sigma = np.sqrt(max(var_trial_sr, 1e-18))
    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sigma * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def probabilistic_sharpe_ratio(sr_hat: float, n: int, skew: float, kurt: float,
                               sr_benchmark: float = 0.0) -> float:
    """PSR: P(true SR > benchmark) given the estimate and its standard error."""
    if n < 2:
        return float("nan")
    denom = np.sqrt(max(1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2, 1e-12))
    z = (sr_hat - sr_benchmark) * np.sqrt(n - 1.0) / denom
    return float(norm.cdf(z))


def deflated_sharpe(returns: np.ndarray, *, family: str | None = None,
                    hyp_id: str | None = None, ledger=None,
                    var_trial_sr: float | None = None,
                    n_trials: int | None = None,
                    allow_manual_trials: bool = False,
                    periods_per_year: float | None = None) -> dict:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size

    # ---- R3: trial count comes from the ledger --------------------------
    if n_trials is not None and not allow_manual_trials:
        raise ValueError(
            "R3 violation: n_trials must come from the ledger, not an argument. "
            "Pass family=/hyp_id= (and optionally ledger=) instead. "
            "allow_manual_trials=True is for synthetic selftests only.")
    if n_trials is None:
        if ledger is None:
            from lab.ledger import Ledger
            ledger = Ledger()
        n_trials = ledger.trial_count(family=family, hyp_id=hyp_id)
        n_trials = max(n_trials, 1)

    sr_hat = sharpe_ratio(r)  # per-period, non-annualised for the deflation
    skew, kurt = _skew_kurt(r)

    if var_trial_sr is None:
        # default: variance of a single trial's SR estimate under the null,
        # ~ 1/(n-1) for per-period SR of iid returns.
        var_trial_sr = 1.0 / max(n - 1, 1)

    sr0 = expected_max_sharpe(n_trials, var_trial_sr)  # deflation benchmark
    dsr = probabilistic_sharpe_ratio(sr_hat, n, skew, kurt, sr_benchmark=sr0)
    psr0 = probabilistic_sharpe_ratio(sr_hat, n, skew, kurt, sr_benchmark=0.0)

    return {
        "sharpe_hat": sr_hat,
        "sharpe_annualised": sharpe_ratio(r, periods_per_year) if periods_per_year else None,
        "n_obs": int(n),
        "n_trials": int(n_trials),
        "trials_source": "ledger" if not allow_manual_trials else "manual(selftest)",
        "skew": skew,
        "kurtosis": kurt,
        "var_trial_sr": float(var_trial_sr),
        "sr_deflation_benchmark": sr0,
        "psr_vs_zero": psr0,
        "dsr": dsr,
    }


def _selftest():
    rng = np.random.default_rng(0)
    # pure noise, many trials -> DSR should be low
    noise = rng.normal(0, 1, 2000)
    d0 = deflated_sharpe(noise, n_trials=500, allow_manual_trials=True)
    # real edge, few trials -> DSR high
    edge = rng.normal(0.08, 1, 2000)
    d1 = deflated_sharpe(edge, n_trials=5, allow_manual_trials=True)
    print("noise  DSR=%.3f (want low) " % d0["dsr"], d0["sharpe_hat"])
    print("edge   DSR=%.3f (want high)" % d1["dsr"], d1["sharpe_hat"])
    assert d0["dsr"] < 0.5, d0
    assert d1["dsr"] > 0.9, d1
    print("dsr selftest OK")


if __name__ == "__main__":
    _selftest()
