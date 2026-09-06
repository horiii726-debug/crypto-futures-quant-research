"""Probability of Backtest Overfitting via CSCV.

Bailey, Borwein, Lopez de Prado, Zhu (2017), "The Probability of Backtest
Overfitting". Combinatorially Symmetric Cross-Validation.

Input: matrix M of shape (T, N) - T time observations of per-period performance
for N candidate configurations. Returns PBO = P(the in-sample best config has
below-median out-of-sample rank).
"""
from __future__ import annotations

import itertools

import numpy as np


def _perf(block: np.ndarray) -> np.ndarray:
    """Per-config performance on a time block: Sharpe-like mean/std."""
    mu = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    sd = np.where(sd == 0, np.nan, sd)
    return mu / sd


def cscv_pbo(M: np.ndarray, s: int = 10, seed: int = 0) -> dict:
    M = np.asarray(M, dtype=float)
    T, N = M.shape
    if N < 2:
        return {"pbo": float("nan"), "n_configs": N, "note": "need >=2 configs"}
    s = s if s % 2 == 0 else s - 1
    s = max(2, min(s, T))
    # split time into s equal blocks
    idx = np.array_split(np.arange(T), s)
    blocks = [b for b in idx if len(b) > 1]
    s = len(blocks)
    if s < 2:
        return {"pbo": float("nan"), "n_configs": N, "note": "series too short"}

    logits = []
    combos = list(itertools.combinations(range(s), s // 2))
    for is_sel in combos:
        oos_sel = tuple(j for j in range(s) if j not in is_sel)
        is_rows = np.concatenate([blocks[j] for j in is_sel])
        oos_rows = np.concatenate([blocks[j] for j in oos_sel])
        is_perf = _perf(M[is_rows])
        oos_perf = _perf(M[oos_rows])
        if np.all(np.isnan(is_perf)):
            continue
        n_star = int(np.nanargmax(is_perf))
        # rank of the IS-best config OOS (1 = worst ... N = best)
        order = np.argsort(np.argsort(np.nan_to_num(oos_perf, nan=-np.inf)))
        rank = (order[n_star] + 1) / (N + 1)  # in (0,1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))

    logits = np.array(logits)
    pbo = float(np.mean(logits <= 0)) if logits.size else float("nan")
    return {
        "pbo": pbo,
        "n_configs": int(N),
        "n_splits": int(s),
        "n_combos": int(logits.size),
        "logit_mean": float(np.mean(logits)) if logits.size else float("nan"),
    }


def _selftest():
    T, N = 800, 24
    # all configs pure noise -> IS winner is random -> PBO ~ 0.5.
    # A single CSCV draw is high-variance, so average over several seeds.
    noise_pbos, edge_pbos = [], []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        noise_pbos.append(cscv_pbo(rng.normal(0, 1, (T, N)), s=12)["pbo"])
        M_edge = rng.normal(0, 1, (T, N))
        M_edge[:, 0] += 0.15
        edge_pbos.append(cscv_pbo(M_edge, s=12)["pbo"])
    p_noise = float(np.nanmean(noise_pbos))
    p_edge = float(np.nanmean(edge_pbos))
    print("noise PBO mean=%.2f over 12 seeds (want ~0.5)" % p_noise)
    print("edge  PBO mean=%.2f over 12 seeds (want low) " % p_edge)
    assert 0.35 <= p_noise <= 0.65, noise_pbos
    assert p_edge < 0.15, edge_pbos
    print("pbo selftest OK")


if __name__ == "__main__":
    _selftest()
