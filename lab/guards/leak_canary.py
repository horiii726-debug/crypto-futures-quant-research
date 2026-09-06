#!/usr/bin/env python3
"""PostToolUse(Write|Edit) hook + importable checker: look-ahead leak detector.

R7  Live-parity: a feature that cannot be computed in real time is invalid.

Two independent tests, tolerance 1e-10:

  future_perturbation:  perturb the input series at indices > t and recompute
                        the feature at t. If feature[t] moves, it peeked ahead.

  prefix_stability:     compute the feature on x[:k] and on x[:k+m]. The first
                        k values must be bit-stable regardless of how much
                        future data is appended.

As a hook it inspects newly written feature modules under lab/features/ and
lab/engine/signals.py for syntactic red flags (negative shifts, .shift(-n),
iloc[i+ ...], center=True rolling) and warns (exit 0, non-blocking) so
red-team still runs the numeric test. It never blocks on heuristics alone.
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np

TOL = 1e-10

RED_FLAGS = [
    re.compile(r"\.shift\(\s*-\d+"),
    re.compile(r"\.shift\(\s*-\s*\w+"),
    re.compile(r"rolling\([^)]*center\s*=\s*True"),
    re.compile(r"iloc\[\s*\w+\s*\+\s*\d+\s*\]"),
    re.compile(r"\[\s*i\s*\+\s*[1-9]"),
    re.compile(r"np\.roll\([^)]*,\s*-\d+"),
    re.compile(r"future|lookahead|look_ahead|peek", re.I),
]


# ---- numeric checks (called by tests + red-team) -----------------------

def future_perturbation(feature_fn, x: np.ndarray, *, n_probe: int = 25,
                        seed: int = 0, tol: float = TOL) -> dict:
    """feature_fn: ndarray -> ndarray (same length, feature per bar)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    base = np.asarray(feature_fn(x.copy()), dtype=float)
    n = len(x)
    max_dev = 0.0
    worst_t = -1
    probes = rng.integers(low=n // 4, high=n - 2, size=min(n_probe, max(1, n - 4)))
    for t in probes:
        xp = x.copy()
        xp[t + 1:] += rng.normal(0, np.std(x) + 1e-9, size=n - t - 1)
        fp = np.asarray(feature_fn(xp), dtype=float)
        dev = np.nanmax(np.abs(fp[: t + 1] - base[: t + 1])) if t + 1 > 0 else 0.0
        if dev > max_dev:
            max_dev, worst_t = float(dev), int(t)
    return {"max_dev": max_dev, "worst_t": worst_t, "passed": max_dev <= tol}


def prefix_stability(feature_fn, x: np.ndarray, *, cuts: int = 8,
                     tol: float = TOL) -> dict:
    x = np.asarray(x, dtype=float)
    n = len(x)
    max_dev = 0.0
    ks = np.linspace(n // 3, n - 5, cuts, dtype=int)
    full = np.asarray(feature_fn(x.copy()), dtype=float)
    for k in ks:
        part = np.asarray(feature_fn(x[:k].copy()), dtype=float)
        m = min(k, len(part))
        dev = np.nanmax(np.abs(part[:m] - full[:m]))
        max_dev = max(max_dev, float(dev))
    return {"max_dev": max_dev, "passed": max_dev <= tol}


def assert_causal(feature_fn, x: np.ndarray, *, tol: float = TOL, seed: int = 0):
    fp = future_perturbation(feature_fn, x, seed=seed, tol=tol)
    ps = prefix_stability(feature_fn, x, tol=tol)
    ok = fp["passed"] and ps["passed"]
    return ok, {"future_perturbation": fp, "prefix_stability": ps}


# ---- hook mode (heuristic, non-blocking) -------------------------------

def scan_source(path: str, text: str):
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("#"):
            continue
        for rx in RED_FLAGS:
            if rx.search(line):
                hits.append((i, line.strip(), rx.pattern))
    return hits


def _is_watched(path: str) -> bool:
    p = (path or "").replace("\\", "/")
    return ("lab/features/" in p or p.endswith("lab/engine/signals.py"))


def main() -> int:
    raw = sys.stdin.read()
    try:
        event = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return 0
    ti = event.get("tool_input") or {}
    path = ti.get("file_path", "")
    if not _is_watched(path):
        return 0
    text = ti.get("content") or ti.get("new_string") or ""
    hits = scan_source(path, text)
    if hits:
        sys.stderr.write(
            f"leak_canary WARNING on {path} - look-ahead red flags "
            f"(non-blocking, run the numeric causality test in red-team):\n"
        )
        for ln, txt, pat in hits[:15]:
            sys.stderr.write(f"  L{ln}: {txt}   <~ /{pat}/\n")
    return 0  # heuristics never block; only numeric assert_causal blocks in CI


if __name__ == "__main__":
    sys.exit(main())
