"""Purged K-Fold, embargo, and Combinatorial Purged CV (CPCV).

Lopez de Prado, "Advances in Financial ML", ch. 7 & 12.

Splits must not leak: any training observation whose label window [t0, t1]
overlaps a test observation's window is PURGED, and a fraction of bars after
each test block is EMBARGOED from training.
"""
from __future__ import annotations

import itertools

import numpy as np


def _overlaps(a0, a1, b0, b1) -> bool:
    return a0 <= b1 and b0 <= a1


def purged_kfold(t0: np.ndarray, t1: np.ndarray, n_splits: int = 5,
                 embargo_pct: float = 0.01):
    """Yield (train_idx, test_idx) with purge + embargo.
    t0, t1 are per-observation label start/end bar indices."""
    t0 = np.asarray(t0, int)
    t1 = np.asarray(t1, int)
    n = len(t0)
    idx = np.arange(n)
    order = np.argsort(t0)
    folds = np.array_split(order, n_splits)
    max_bar = int(t1.max())
    embargo = int(round(embargo_pct * max_bar))

    for f in folds:
        test_idx = np.sort(f)
        test_t0, test_t1 = t0[test_idx].min(), t1[test_idx].max()
        emb_hi = test_t1 + embargo
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        for i in idx:
            if not train_mask[i]:
                continue
            # purge: training label window overlaps any test label window
            if _overlaps(t0[i], t1[i], test_t0, test_t1):
                train_mask[i] = False
                continue
            # embargo: training obs starts within the embargo band after test
            if test_t1 < t0[i] <= emb_hi:
                train_mask[i] = False
        yield idx[train_mask], test_idx


def cpcv(t0: np.ndarray, t1: np.ndarray, n_groups: int = 6, k_test: int = 2,
         embargo_pct: float = 0.01):
    """Combinatorial Purged CV: all C(n_groups, k_test) test combinations.
    Yields (train_idx, test_idx, test_groups)."""
    t0 = np.asarray(t0, int)
    t1 = np.asarray(t1, int)
    n = len(t0)
    order = np.argsort(t0)
    groups = np.array_split(order, n_groups)
    max_bar = int(t1.max())
    embargo = int(round(embargo_pct * max_bar))

    for combo in itertools.combinations(range(n_groups), k_test):
        test_idx = np.sort(np.concatenate([groups[g] for g in combo]))
        test_t0, test_t1 = t0[test_idx].min(), t1[test_idx].max()
        emb_hi = test_t1 + embargo
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        for i in range(n):
            if not train_mask[i]:
                continue
            if _overlaps(t0[i], t1[i], test_t0, test_t1):
                train_mask[i] = False
            elif test_t1 < t0[i] <= emb_hi:
                train_mask[i] = False
        yield np.arange(n)[train_mask], test_idx, combo


def leakage_check(t0, t1, train_idx, test_idx) -> dict:
    """Assert no train/test label-window overlap remains."""
    t0 = np.asarray(t0, int)
    t1 = np.asarray(t1, int)
    tt0, tt1 = t0[test_idx].min(), t1[test_idx].max()
    bad = [int(i) for i in train_idx if _overlaps(t0[i], t1[i], tt0, tt1)]
    return {"leaked_train_obs": bad, "clean": len(bad) == 0}


def walk_forward(n_obs: int, train_span: int, test_span: int, step: int | None = None):
    """Anchored-free rolling walk-forward index windows."""
    step = step or test_span
    start = 0
    while start + train_span + test_span <= n_obs:
        tr = np.arange(start, start + train_span)
        te = np.arange(start + train_span, start + train_span + test_span)
        yield tr, te
        start += step


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    t0 = np.sort(rng.integers(0, 9000, n))
    t1 = t0 + rng.integers(5, 40, n)
    total_leak = 0
    for tr, te in purged_kfold(t0, t1, n_splits=5, embargo_pct=0.01):
        chk = leakage_check(t0, t1, tr, te)
        total_leak += len(chk["leaked_train_obs"])
    print(f"purged_kfold: {total_leak} leaked train obs across folds (want 0)")
    combos = list(cpcv(t0, t1, n_groups=6, k_test=2))
    print(f"cpcv: {len(combos)} train/test combinations (want 15)")
    assert total_leak == 0 and len(combos) == 15
    print("cv selftest OK")
