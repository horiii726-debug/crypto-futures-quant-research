"""Triple-barrier labelling with average uniqueness and overlap factor.

Lopez de Prado, "Advances in Financial Machine Learning", ch. 3-4.

- triple_barrier: for each event t0, place an upper (profit) and lower (stop)
  horizontal barrier scaled by local volatility, plus a vertical barrier
  (max holding). Label = sign of the barrier touched first; 0 if only the
  vertical barrier is hit.
- average_uniqueness: how much each label's holding window overlaps with the
  others. Overlapping labels are not independent observations - power.py and
  cv.py consume this.
"""
from __future__ import annotations

import numpy as np


def daily_vol(close: np.ndarray, span: int = 100) -> np.ndarray:
    close = np.asarray(close, float)
    ret = np.zeros_like(close)
    ret[1:] = close[1:] / close[:-1] - 1.0
    # EWMA std of returns
    alpha = 2.0 / (span + 1.0)
    var = np.zeros_like(ret)
    var[0] = ret[0] ** 2
    mean = ret[0]
    for i in range(1, len(ret)):
        mean = alpha * ret[i] + (1 - alpha) * mean
        var[i] = alpha * (ret[i] - mean) ** 2 + (1 - alpha) * var[i - 1]
    return np.sqrt(np.maximum(var, 0.0))


def triple_barrier(close: np.ndarray, events_idx: np.ndarray, *,
                   pt_mult: float = 2.0, sl_mult: float = 2.0,
                   max_hold: int = 20, vol: np.ndarray | None = None,
                   vol_span: int = 100, side: np.ndarray | None = None) -> dict:
    close = np.asarray(close, float)
    n = len(close)
    events_idx = np.asarray(events_idx, int)
    if vol is None:
        vol = daily_vol(close, vol_span)
    vol = np.asarray(vol, float)
    if side is None:
        side = np.ones(len(events_idx))
    side = np.asarray(side, float)

    labels = np.zeros(len(events_idx), dtype=np.int8)
    t1 = np.zeros(len(events_idx), dtype=np.int64)     # exit bar index
    ret = np.zeros(len(events_idx), dtype=float)
    touch = np.array([""] * len(events_idx), dtype=object)

    for k, t0 in enumerate(events_idx):
        if t0 >= n - 1:
            t1[k] = t0
            touch[k] = "none"
            continue
        v = vol[t0] if vol[t0] > 0 else np.nanmedian(vol[vol > 0])
        up = pt_mult * v
        dn = -sl_mult * v
        vbar = min(t0 + max_hold, n - 1)
        path = close[t0 + 1: vbar + 1] / close[t0] - 1.0
        path = side[k] * path  # respect trade side for meta-labelling
        hit_up = np.where(path >= up)[0]
        hit_dn = np.where(path <= dn)[0]
        first_up = hit_up[0] if hit_up.size else np.inf
        first_dn = hit_dn[0] if hit_dn.size else np.inf
        if first_up < first_dn:
            j = int(first_up)
            labels[k] = 1
            touch[k] = "pt"
        elif first_dn < first_up:
            j = int(first_dn)
            labels[k] = -1
            touch[k] = "sl"
        else:
            j = len(path) - 1
            labels[k] = 0
            touch[k] = "vertical"
        t1[k] = t0 + 1 + j
        ret[k] = float(path[j])

    return {
        "events_idx": events_idx,
        "t1": t1,
        "label": labels,
        "ret": ret,
        "touch": touch,
        "params": {"pt_mult": pt_mult, "sl_mult": sl_mult, "max_hold": max_hold},
    }


def _indicator_matrix(t0: np.ndarray, t1: np.ndarray, n_bars: int) -> np.ndarray:
    m = np.zeros((n_bars, len(t0)), dtype=np.float64)
    for k in range(len(t0)):
        m[int(t0[k]): int(t1[k]) + 1, k] = 1.0
    return m


def average_uniqueness(t0: np.ndarray, t1: np.ndarray,
                       n_bars: int | None = None) -> dict:
    t0 = np.asarray(t0, int)
    t1 = np.asarray(t1, int)
    if n_bars is None:
        n_bars = int(t1.max()) + 1
    ind = _indicator_matrix(t0, t1, n_bars)
    concurrency = ind.sum(axis=1)
    concurrency[concurrency == 0] = 1.0
    u = np.zeros(len(t0))
    for k in range(len(t0)):
        span = slice(int(t0[k]), int(t1[k]) + 1)
        contrib = ind[span, k] / concurrency[span]
        u[k] = contrib.mean() if contrib.size else 0.0
    avg_u = float(np.mean(u)) if len(u) else 0.0
    return {
        "uniqueness": u,
        "avg_uniqueness": avg_u,
        "overlap_factor": float(1.0 / avg_u) if avg_u > 0 else np.inf,
        "mean_concurrency": float(np.mean(concurrency)),
    }


def barrier_payoff_distribution(res: dict) -> dict:
    r = np.asarray(res["ret"], float)
    lab = np.asarray(res["label"], int)
    return {
        "mean": float(r.mean()),
        "std": float(r.std(ddof=1)) if r.size > 1 else 0.0,
        "skew": float(((r - r.mean()) ** 3).mean() / (r.std() ** 3 + 1e-18)),
        "p_up": float(np.mean(lab == 1)),
        "p_dn": float(np.mean(lab == -1)),
        "p_vertical": float(np.mean(lab == 0)),
        "q05": float(np.quantile(r, 0.05)),
        "q95": float(np.quantile(r, 0.95)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000)))
    ev = np.arange(50, 2900, 5)
    tight = triple_barrier(px, ev, pt_mult=1.0, sl_mult=1.0, max_hold=20)
    wide = triple_barrier(px, ev, pt_mult=4.0, sl_mult=4.0, max_hold=20)
    print("tight barriers payoff:", barrier_payoff_distribution(tight))
    print("wide  barriers payoff:", barrier_payoff_distribution(wide))
    au = average_uniqueness(tight["events_idx"], tight["t1"], len(px))
    print("avg uniqueness=%.3f overlap=%.2f" % (au["avg_uniqueness"], au["overlap_factor"]))
