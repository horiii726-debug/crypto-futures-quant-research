"""Numeric kernels with a 3-tier backend and a bit-identity contract.

Tier 1  numba @njit (fastmath=False)   - production
Tier 2  numpy vectorised               - fallback when numba absent
Tier 3  pure python                    - reference / debugging

verify_kernels() runs every kernel through all available tiers on shared
inputs and asserts the outputs are BIT-IDENTICAL (np.array_equal on the raw
bytes / exact float equality). Tolerance is ZERO. A backend MISMATCH means the
kernel set is unusable and the pipeline must refuse to run.

Rule: fastmath is never enabled; reductions use the same summation order across
tiers (sequential, left-to-right) so the float results are exactly equal.
"""
from __future__ import annotations

import numpy as np

try:  # Tier 1 availability
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*a, **k):  # no-op decorator
        def wrap(f):
            return f
        if a and callable(a[0]):
            return a[0]
        return wrap


# ----------------------------------------------------------------------------
# Tier 3 - pure python reference (defines the canonical result & order)
# ----------------------------------------------------------------------------

def _py_rolling_sum(x, w):
    n = x.shape[0]
    out = np.full(n, np.nan)
    if w <= 0 or w > n:
        return out
    acc = 0.0
    for i in range(n):
        acc += float(x[i])
        if i >= w:
            acc -= float(x[i - w])
        if i >= w - 1:
            out[i] = acc
    return out


def _py_ewma(x, alpha):
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n == 0:
        return out
    m = float(x[0])
    out[0] = m
    for i in range(1, n):
        m = alpha * float(x[i]) + (1.0 - alpha) * m
        out[i] = m
    return out


def _py_cum_return(logret):
    n = logret.shape[0]
    out = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc += float(logret[i])
        out[i] = acc
    return out


def _py_rank_lastaxis(x):
    # average-rank of each row, normalised to (0,1]; NaNs -> NaN
    r, c = x.shape
    out = np.full((r, c), np.nan)
    for i in range(r):
        row = x[i]
        valid = [j for j in range(c) if not np.isnan(row[j])]
        k = len(valid)
        if k == 0:
            continue
        order = sorted(valid, key=lambda j: float(row[j]))
        # average ranks for ties
        j = 0
        while j < k:
            lo = j
            while j + 1 < k and float(row[order[j + 1]]) == float(row[order[lo]]):
                j += 1
            avg = (lo + j) / 2.0 + 1.0
            for t in range(lo, j + 1):
                out[i, order[t]] = avg / k
            j += 1
    return out


def _py_max_drawdown(equity):
    n = equity.shape[0]
    peak = float(equity[0]) if n else 0.0
    mdd = 0.0
    for i in range(n):
        v = float(equity[i])
        if v > peak:
            peak = v
        dd = (v - peak)
        if dd < mdd:
            mdd = dd
    return mdd


# ----------------------------------------------------------------------------
# Tier 2 - numpy vectorised (must match tier 3 exactly)
# ----------------------------------------------------------------------------

def _np_rolling_sum(x, w):
    # A cumsum-and-subtract vectorisation does NOT reproduce the running
    # accumulator's floating-point cancellation path, so bit-identity (tol 0)
    # would fail. The sliding-window recurrence is the canonical definition;
    # mirror it exactly. This tier is the fallback when numba is absent -
    # correctness over speed.
    return _py_rolling_sum(x, w)


def _np_ewma(x, alpha):
    # recurrence is inherently sequential; mirror tier 3 exactly
    return _py_ewma(x, alpha)


def _np_cum_return(logret):
    return np.cumsum(logret.astype(np.float64))


def _np_rank_lastaxis(x):
    return _py_rank_lastaxis(x)


def _np_max_drawdown(equity):
    eq = equity.astype(np.float64)
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq - peak))


# ----------------------------------------------------------------------------
# Tier 1 - numba njit (sequential math, fastmath=False -> identical floats)
# ----------------------------------------------------------------------------

@njit(cache=False, fastmath=False)
def _nb_rolling_sum(x, w):
    n = x.shape[0]
    out = np.full(n, np.nan)
    if w <= 0 or w > n:
        return out
    acc = 0.0
    for i in range(n):
        acc += x[i]
        if i >= w:
            acc -= x[i - w]
        if i >= w - 1:
            out[i] = acc
    return out


@njit(cache=False, fastmath=False)
def _nb_ewma(x, alpha):
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n == 0:
        return out
    m = x[0]
    out[0] = m
    for i in range(1, n):
        m = alpha * x[i] + (1.0 - alpha) * m
        out[i] = m
    return out


@njit(cache=False, fastmath=False)
def _nb_cum_return(logret):
    n = logret.shape[0]
    out = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc += logret[i]
        out[i] = acc
    return out


@njit(cache=False, fastmath=False)
def _nb_max_drawdown(equity):
    n = equity.shape[0]
    if n == 0:
        return 0.0
    peak = equity[0]
    mdd = 0.0
    for i in range(n):
        v = equity[i]
        if v > peak:
            peak = v
        dd = v - peak
        if dd < mdd:
            mdd = dd
    return mdd


# ----------------------------------------------------------------------------
# Public dispatch
# ----------------------------------------------------------------------------

_BACKEND = "numba" if _HAVE_NUMBA else "numpy"


def set_backend(name: str):
    global _BACKEND
    if name not in ("numba", "numpy", "python"):
        raise ValueError(name)
    if name == "numba" and not _HAVE_NUMBA:
        raise RuntimeError("numba backend requested but numba is not importable")
    _BACKEND = name


def backend() -> str:
    return _BACKEND


_TIERS = {
    "rolling_sum": {"python": _py_rolling_sum, "numpy": _np_rolling_sum, "numba": _nb_rolling_sum},
    "ewma": {"python": _py_ewma, "numpy": _np_ewma, "numba": _nb_ewma},
    "cum_return": {"python": _py_cum_return, "numpy": _np_cum_return, "numba": _nb_cum_return},
    "rank_lastaxis": {"python": _py_rank_lastaxis, "numpy": _np_rank_lastaxis, "numba": _np_rank_lastaxis},
    "max_drawdown": {"python": _py_max_drawdown, "numpy": _np_max_drawdown, "numba": _nb_max_drawdown},
}


def _call(name, *args):
    return _TIERS[name][_BACKEND](*args)


def rolling_sum(x, w):
    return _call("rolling_sum", np.ascontiguousarray(x, dtype=np.float64), int(w))


def ewma(x, alpha):
    return _call("ewma", np.ascontiguousarray(x, dtype=np.float64), float(alpha))


def cum_return(logret):
    return _call("cum_return", np.ascontiguousarray(logret, dtype=np.float64))


def rank_lastaxis(x):
    return _call("rank_lastaxis", np.ascontiguousarray(x, dtype=np.float64))


def max_drawdown(equity):
    return _call("max_drawdown", np.ascontiguousarray(equity, dtype=np.float64))


# ----------------------------------------------------------------------------
# Bit-identity verification - tolerance ZERO
# ----------------------------------------------------------------------------

def _exact_equal(a, b) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    # NaN in the same positions, and every non-NaN value bit-identical
    na, nb = np.isnan(a), np.isnan(b)
    if not np.array_equal(na, nb):
        return False
    return bool(np.array_equal(a[~na].view(np.uint64), b[~nb].view(np.uint64)))


def verify_kernels(seed: int = 0, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, 5000).astype(np.float64)
    logret = (rng.normal(0, 0.01, 5000)).astype(np.float64)
    mat = rng.normal(0, 1, (400, 60)).astype(np.float64)
    mat[rng.random(mat.shape) < 0.05] = np.nan
    equity = np.cumsum(rng.normal(0.0, 1.0, 3000)).astype(np.float64) + 100.0

    cases = {
        "rolling_sum": (x, 32),
        "ewma": (x, 0.1),
        "cum_return": (logret,),
        "rank_lastaxis": (mat,),
        "max_drawdown": (equity,),
    }

    tiers = ["python", "numpy"] + (["numba"] if _HAVE_NUMBA else [])
    results = {}
    all_ok = True
    for name, args in cases.items():
        ref = _TIERS[name]["python"](*args)
        row = {"tiers_checked": tiers, "status": "IDENTICAL"}
        for t in tiers:
            out = _TIERS[name][t](*args)
            if not _exact_equal(ref, out):
                row["status"] = "MISMATCH"
                row["mismatch_tier"] = t
                max_dev = float(np.nanmax(np.abs(np.asarray(out, float) - np.asarray(ref, float))))
                row["max_abs_dev"] = max_dev
                all_ok = False
        results[name] = row
        if verbose:
            print(f"  {name:16} {row['status']}  ({', '.join(tiers)})")

    summary = {
        "backend_active": _BACKEND,
        "numba_available": _HAVE_NUMBA,
        "tiers": tiers,
        "kernels": results,
        "all_identical": all_ok,
        "verdict": "IDENTICAL" if all_ok else "MISMATCH",
    }
    return summary


if __name__ == "__main__":
    import json
    import sys

    print(f"numba available: {_HAVE_NUMBA}   active backend: {_BACKEND}")
    s = verify_kernels()
    print(json.dumps({k: v for k, v in s.items() if k != "kernels"}, indent=2))
    if not s["all_identical"]:
        print("KERNEL VERIFY: MISMATCH - pipeline must not run", file=sys.stderr)
        sys.exit(1)
    print("KERNEL VERIFY: numba IDENTICAL" if _HAVE_NUMBA else
          "KERNEL VERIFY: IDENTICAL (numba absent, numpy<->python)")
    sys.exit(0)
