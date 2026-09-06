"""Model Confidence Set (Hansen, Lunde, Nason 2011) + SPA (Hansen 2005)
+ Diebold-Mariano (1995).

Selftest contract (config/gates.yaml G5, spec in the S0 brief):
  200 models drawn from the SAME distribution -> the MCS must retain MANY of
  them, not crown a single winner. If it returns a singleton on exchangeable
  noise, the implementation is wrong and the selftest fails.
"""
from __future__ import annotations

import numpy as np


# ---- Diebold-Mariano -------------------------------------------------------

def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> dict:
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        return {"dm_stat": float("nan"), "p_value": float("nan"), "n": n}
    dbar = d.mean()
    # Newey-West long-run variance
    gamma0 = np.mean((d - dbar) ** 2)
    lrv = gamma0
    for k in range(1, h):
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        lrv += 2 * (1 - k / h) * cov
    var = lrv / n
    if var <= 0:
        return {"dm_stat": 0.0, "p_value": 1.0, "n": n}
    stat = dbar / np.sqrt(var)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(stat)))
    return {"dm_stat": float(stat), "p_value": float(p), "n": int(n), "mean_diff": float(dbar)}


# ---- block bootstrap helper ---------------------------------------------

def _stationary_bootstrap_idx(n: int, block: float, rng) -> np.ndarray:
    idx = np.empty(n, dtype=int)
    p = 1.0 / max(block, 1.0)
    i = rng.integers(0, n)
    for t in range(n):
        idx[t] = i
        if rng.random() < p:
            i = rng.integers(0, n)
        else:
            i = (i + 1) % n
    return idx


# ---- SPA (Hansen 2005) -------------------------------------------------------

def spa_test(bench_loss: np.ndarray, model_losses: np.ndarray, *,
             n_boot: int = 1000, block: float = 10.0, seed: int = 0) -> dict:
    """H0: no model beats the benchmark (in expected loss).
    model_losses: (T, K). Returns consistent SPA p-value."""
    rng = np.random.default_rng(seed)
    b = np.asarray(bench_loss, float)
    L = np.asarray(model_losses, float)
    T, K = L.shape
    d = b[:, None] - L                     # >0 means model better than bench
    dbar = d.mean(axis=0)
    omega = d.std(axis=0, ddof=1) / np.sqrt(T)
    omega = np.where(omega == 0, np.inf, omega)
    t_stat = np.sqrt(T) * dbar / (omega * np.sqrt(T))
    obs = max(0.0, float(np.max(t_stat)))
    # recentering threshold (consistent variant)
    thresh = -np.sqrt(2 * np.log(np.log(max(T, 3)))) * omega * np.sqrt(T) / np.sqrt(T)
    keep = dbar >= thresh
    count = 0
    for _ in range(n_boot):
        bi = _stationary_bootstrap_idx(T, block, rng)
        db = d[bi]
        zb = np.sqrt(T) * (db.mean(axis=0) - np.where(keep, dbar, 0.0)) / (omega * np.sqrt(T))
        tb = float(np.max(np.maximum(zb, 0.0)))
        if tb >= obs:
            count += 1
    return {"spa_p_value": count / n_boot, "max_t": obs, "n_models": int(K)}


# ---- Model Confidence Set ----------------------------------------------------

def model_confidence_set(losses: np.ndarray, *, alpha: float = 0.10,
                         n_boot: int = 1000, block: float = 10.0,
                         seed: int = 0) -> dict:
    """losses: (T, K) per-period loss for K models (lower = better).
    Returns the set of model indices not rejected at level alpha, using the
    range statistic T_R with a stationary-bootstrap null distribution.
    """
    rng = np.random.default_rng(seed)
    L = np.asarray(losses, float)
    T, K = L.shape
    alive = list(range(K))
    boot_idx = [_stationary_bootstrap_idx(T, block, rng) for _ in range(n_boot)]
    pvals = {}

    while len(alive) > 1:
        sub = L[:, alive]
        Lbar = sub.mean(axis=0)
        # pairwise mean loss differences
        dij = Lbar[:, None] - Lbar[None, :]
        # bootstrap variance of dij
        var = np.zeros((len(alive), len(alive)))
        boot_d = np.empty((n_boot, len(alive), len(alive)))
        for bnum, bi in enumerate(boot_idx):
            sb = sub[bi].mean(axis=0)
            bd = (sb[:, None] - sb[None, :]) - dij
            boot_d[bnum] = bd
        var = boot_d.var(axis=0, ddof=1)
        var[var == 0] = np.inf
        tstat = np.abs(dij) / np.sqrt(var)
        TR = float(np.nanmax(tstat))
        # null distribution of the range statistic
        boot_TR = np.nanmax(np.abs(boot_d) / np.sqrt(var)[None, :, :], axis=(1, 2))
        p = float(np.mean(boot_TR >= TR))
        # identify the worst model (largest average excess loss vs set)
        elim = alive[int(np.argmax(Lbar - Lbar.mean()))]
        pvals[elim] = p
        if p > alpha:
            # cannot reject equal predictive ability -> whole set is the MCS
            break
        alive.remove(elim)

    mcs_pvals = {}
    running = 0.0
    for m, p in pvals.items():
        running = max(running, p)
        mcs_pvals[m] = running
    for m in alive:
        mcs_pvals[m] = 1.0

    return {
        "mcs": sorted(alive),
        "mcs_size": len(alive),
        "n_models": int(K),
        "alpha": alpha,
        "elimination_pvalues": pvals,
        "mcs_pvalues": mcs_pvals,
        "singleton": len(alive) == 1,
    }


def _selftest():
    rng = np.random.default_rng(42)
    T, K = 500, 200
    # 200 exchangeable models: same distribution, only sampling noise differs
    losses = rng.normal(0.0, 1.0, (T, K)) ** 2  # squared-error style losses
    res = model_confidence_set(losses, alpha=0.10, n_boot=400, seed=1)
    print(f"MCS on 200 iid-noise models: kept {res['mcs_size']} / {K}, "
          f"singleton={res['singleton']}")
    assert not res["singleton"], "MCS collapsed to one winner on pure noise - BUG"
    assert res["mcs_size"] >= K // 4, f"MCS kept too few ({res['mcs_size']})"

    # SPA on noise: benchmark vs 20 noise models -> high p-value
    spa = spa_test(losses[:, 0], losses[:, 1:21], n_boot=400, seed=2)
    print(f"SPA p-value on noise = {spa['spa_p_value']:.3f} (want high)")
    assert spa["spa_p_value"] > 0.10

    # DM: identical losses -> p ~ 1
    dm = diebold_mariano(losses[:, 0], losses[:, 0])
    print(f"DM identical series p={dm['p_value']:.3f}")
    print("mcs selftest OK")


if __name__ == "__main__":
    _selftest()
