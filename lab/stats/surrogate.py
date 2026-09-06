"""Surrogate / label-shuffle noise thresholds via block bootstrap.

red-team uses this: shuffle the labels in blocks (preserving the feature
autocorrelation and the label marginal), re-run the same evaluation many
times, and read off the noise ceiling. A live strategy metric that does not
clear this ceiling is indistinguishable from luck.
"""
from __future__ import annotations

import numpy as np


def circular_block_shuffle(y: np.ndarray, block: int, rng) -> np.ndarray:
    n = len(y)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    out = np.concatenate([np.take(y, range(s, s + block), mode="wrap") for s in starts])
    return out[:n]


def block_permutation(y: np.ndarray, block: int, rng) -> np.ndarray:
    n = len(y)
    idx = np.arange(n)
    blocks = [idx[i:i + block] for i in range(0, n, block)]
    rng.shuffle(blocks)
    return y[np.concatenate(blocks)][:n]


def surrogate_threshold(metric_fn, feature, label, *, n_surr: int = 500,
                        block: int = 24, seed: int = 0, mode: str = "permute",
                        quantiles=(0.95, 0.99)) -> dict:
    """metric_fn(feature, label) -> float (higher = better, e.g. net Sharpe or IC).
    Returns the observed metric and the surrogate null distribution summary."""
    rng = np.random.default_rng(seed)
    feature = np.asarray(feature, float)
    label = np.asarray(label, float)
    observed = float(metric_fn(feature, label))
    null = np.empty(n_surr)
    for i in range(n_surr):
        if mode == "circular":
            ys = circular_block_shuffle(label, block, rng)
        else:
            ys = block_permutation(label, block, rng)
        null[i] = metric_fn(feature, ys)
    null = null[np.isfinite(null)]
    qs = {f"q{int(q*100)}": float(np.quantile(null, q)) for q in quantiles}
    p_value = float((np.sum(null >= observed) + 1) / (null.size + 1))
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        **qs,
        "noise_ceiling_q99": qs.get("q99"),
        "p_value": p_value,
        "clears_noise": observed > qs.get("q99", np.inf),
        "n_surrogates": int(null.size),
    }


def _selftest():
    rng = np.random.default_rng(0)
    n = 4000
    f = rng.normal(size=n)

    def ic(feat, lab):
        feat = feat[np.isfinite(feat) & np.isfinite(lab)]
        lab = lab[np.isfinite(lab)][: feat.size]
        if feat.size < 3 or feat.std() == 0 or lab.std() == 0:
            return 0.0
        return float(np.corrcoef(feat[: lab.size], lab)[0, 1])

    # no relationship -> observed should NOT clear noise ceiling
    y_noise = rng.normal(size=n)
    r0 = surrogate_threshold(ic, f, y_noise, n_surr=300, block=24, seed=1)
    # real relationship -> observed clears ceiling
    y_edge = 0.3 * f + rng.normal(size=n)
    r1 = surrogate_threshold(ic, f, y_edge, n_surr=300, block=24, seed=2)
    print(f"noise: obs={r0['observed']:.3f} q99={r0['q99']:.3f} clears={r0['clears_noise']}")
    print(f"edge : obs={r1['observed']:.3f} q99={r1['q99']:.3f} clears={r1['clears_noise']}")
    assert not r0["clears_noise"]
    assert r1["clears_noise"]
    assert 0.0 < r0["q99"] < 0.2, "noise ceiling should be small but positive"
    print("surrogate selftest OK")


if __name__ == "__main__":
    _selftest()
