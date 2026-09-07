"""RESEARCH FACTORY pipeline — N3 cheap screen + N4-N9 gates + O taxonomy.

One formula = one trial at author defaults (no grid). Every TF result is logged.
DSR is per-family (pre-registered), never pooled.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
from lab.engine.cv import purged_kfold                          # noqa: E402
from lab.exec.cost_model import cost_frac_by_coin, roundtrip_bps  # noqa: E402
from lab.stats.dsr import deflated_sharpe                        # noqa: E402

TF_HOURS = {"M3": 0.05, "M5": 0.0833, "M15": 0.25, "H1": 1.0, "H4": 4.0, "D1": 24.0}
ANN_FROM_TF = {k: np.sqrt(365 * 24 / v) for k, v in TF_HOURS.items()}

ROOT_CAUSES = ("COST_BOUND", "NO_EDGE", "OVERFIT", "REGIME_DEPENDENT",
               "DATA_LIMITED", "IMPLEMENTATION", "METRIC_MISMATCH")


def _sr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1)) if x.size > 5 and x.std(ddof=1) > 0 else 0.0


def rank_ic(sig: pd.DataFrame, fwd: pd.DataFrame) -> float:
    a, b = sig.rank(axis=1), fwd.rank(axis=1)
    return float(a.corrwith(b, axis=1).mean(skipna=True))


def signal_autocorr_turnover(sig: pd.DataFrame) -> float:
    """turnover estimate from lag-1 autocorrelation of the cross-sectional signal."""
    s = sig.rank(axis=1, pct=True).sub(0.5)
    a = s.shift(1)
    m = s.notna() & a.notna()
    if m.values.sum() < 100:
        return 1.0
    rho = float(np.corrcoef(s.values[m.values], a.values[m.values])[0, 1])
    return float(np.clip(2.0 * (1.0 - max(rho, 0.0)), 0.01, 2.0))


# --------------------------------------------------------------------------- #
#  N3 — cheap screen across all timeframes                                    #
# --------------------------------------------------------------------------- #
def cheap_screen(feature_by_tf: dict[str, pd.DataFrame],
                 close_by_tf: dict[str, pd.DataFrame],
                 *, universe_cost_bps: float | None = None) -> dict:
    """feature_by_tf / close_by_tf : {TF: DataFrame (time x coin)}.
    Returns per-TF screen metrics + the best ratio."""
    out = {}
    for tf, feat in feature_by_tf.items():
        C = close_by_tf.get(tf)
        if C is None or feat is None or feat.dropna(how="all").empty:
            continue
        fwd = C.shift(-1) / C - 1.0
        feat, fwd = feat.align(fwd, join="inner")
        ic = rank_ic(feat, fwd)
        breadth = float(feat.notna().sum(axis=1).median())
        sig_fwd_bps = float(np.nanmedian(fwd.std(axis=1))) * 1e4
        # per-bar edge of a rank-weighted cross-sectional book ~ IC * cross-sectional
        # dispersion of forward returns (fundamental law: IR = IC*sqrt(breadth),
        # but the PER-BAR RETURN is IC*sigma_spread, not scaled by breadth).
        edge_bps = abs(ic) * sig_fwd_bps
        turn = signal_autocorr_turnover(feat)
        cost = universe_cost_bps if universe_cost_bps is not None else \
            float(np.nanmean([roundtrip_bps(c) for c in list(C.columns)[:12]]))
        # scale cost to the holding implied by turnover (fewer turns -> less cost drag per bar)
        cost_per_bar = cost * turn
        ratio = edge_bps / cost_per_bar if cost_per_bar > 0 else 0.0
        out[tf] = {"ic_raw": round(ic, 5), "breadth": round(breadth, 1),
                   "sigma_fwd_bps": round(sig_fwd_bps, 1),
                   "edge_bps_est": round(edge_bps, 2), "turnover_est": round(turn, 3),
                   "cost_bps": round(cost, 2), "cost_per_bar_bps": round(cost_per_bar, 3),
                   "ratio": round(ratio, 2)}
    best_tf = max(out, key=lambda t: out[t]["ratio"], default=None)
    return {"by_tf": out, "best_tf": best_tf,
            "best_ratio": out[best_tf]["ratio"] if best_tf else 0.0,
            "passes_screen": bool(best_tf and out[best_tf]["ratio"] >= 3.0),
            "short_tf_dead": bool(out and all(out[t]["ratio"] < 3.0
                                              for t in ("M3", "M5", "M15") if t in out)
                             and any(out[t]["ratio"] >= 3.0 for t in ("H1", "H4", "D1") if t in out))}


# --------------------------------------------------------------------------- #
#  N4-N9 — full gate                                                          #
# --------------------------------------------------------------------------- #
def full_gate(sig: pd.DataFrame, close: pd.DataFrame, *, family: str, ledger,
              tf: str, hold_bars: int = 1, q: float = 0.25, exec_lag: int = 1,
              n_placebo: int = 300, n_surrogate: int = 300, seed: int = 0,
              cost_mode: str = "hybrid") -> dict:
    rng = np.random.default_rng(seed)
    ann = ANN_FROM_TF[tf]
    sig, close = sig.align(close, join="inner", axis=0)
    sig = sig.reindex(columns=close.columns)
    cols = list(close.columns)
    cvec = cost_frac_by_coin(cols, mode=cost_mode).reindex(cols)
    cvec = cvec.fillna(cvec.median())

    r = sig.rank(axis=1, pct=True)
    lo = (r >= 1 - q).astype(float); sh = (r <= q).astype(float)
    nl = lo.sum(axis=1).replace(0, np.nan); ns = sh.sum(axis=1).replace(0, np.nan)
    pos = (lo.div(nl, axis=0) - sh.div(ns, axis=0)).fillna(0.0) / 2.0
    if hold_bars > 1:
        pos = pos.rolling(hold_bars, min_periods=1).mean()
    bwd = close / close.shift(1) - 1.0
    held = pos.shift(exec_lag + 1)
    gross = (held * bwd).sum(axis=1)
    turn_coin = held.diff().abs().fillna(held.abs())
    tc = turn_coin.mul(cvec, axis=1).sum(axis=1)
    net = (gross - tc).dropna()
    gross = gross.reindex(net.index)
    turnover = float(turn_coin.sum(axis=1).mean())

    fwd1 = close.shift(-1) / close - 1.0
    ic = rank_ic(sig, fwd1)

    # purged CV, embargo = 1.5 * max(hold, horizon)
    T = len(net)
    idx = net.index
    t0 = np.arange(T)
    emb = int(1.5 * max(hold_bars, 1))
    t1 = np.minimum(t0 + emb, T - 1)
    fold = []
    for tr, te in purged_kfold(t0, t1, n_splits=6, embargo_pct=0.02):
        seg = net.iloc[np.sort(te)]
        if len(seg) > 20 and seg.std() > 0:
            fold.append(_sr(seg) * ann)
    cv_pos = sum(1 for f in fold if f > 0)

    # placebo — shuffle the cross-sectional ranking
    obs = _sr(net) * ann
    base = sig.notna()
    plc = np.empty(n_placebo)
    for i in range(n_placebo):
        nz = pd.DataFrame(rng.normal(size=sig.shape), index=sig.index, columns=sig.columns).where(base)
        rr = nz.rank(axis=1, pct=True)
        pl = ((rr >= 1 - q).astype(float).div((rr >= 1 - q).sum(axis=1).replace(0, np.nan), axis=0)
              - (rr <= q).astype(float).div((rr <= q).sum(axis=1).replace(0, np.nan), axis=0)).fillna(0) / 2
        if hold_bars > 1:
            pl = pl.rolling(hold_bars, min_periods=1).mean()
        h = pl.shift(exec_lag + 1)
        g = (h * bwd).sum(axis=1)
        t = h.diff().abs().fillna(h.abs()).mul(cvec, axis=1).sum(axis=1)
        plc[i] = _sr((g - t).dropna()) * ann
    placebo_p = float((np.sum(plc >= obs) + 1) / (n_placebo + 1))

    # surrogate — block-bootstrap forward returns, recompute |IC|
    arr = fwd1.values
    L = len(arr); blk = max(hold_bars * 3, 24)
    obs_ic = abs(ic)
    sr_ = sig.rank(axis=1).values
    src = sr_ - np.nanmean(sr_, axis=1, keepdims=True)
    ss = np.nansum(src ** 2, axis=1)
    fr_all = fwd1.rank(axis=1).values
    sur = np.empty(n_surrogate)
    for i in range(n_surrogate):
        st = rng.integers(0, L, int(np.ceil(L / blk)))
        order = np.concatenate([np.arange(s, s + blk) % L for s in st])[:L]
        fr = fr_all[order]; frc = fr - np.nanmean(fr, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = np.nansum(src * frc, axis=1) / np.sqrt(ss * np.nansum(frc ** 2, axis=1))
        sur[i] = abs(float(np.nanmean(rr)))
    surrogate_p = float((np.sum(sur >= obs_ic) + 1) / (n_surrogate + 1))

    # PBO proxy — 8 blocks, IS/OOS sign persistence
    sub = np.array([_sr(net.iloc[k * (T // 8):(k + 1) * (T // 8)]) * ann for k in range(8)])
    pbo = float(np.mean([(sub[k] > 0 and sub[k + 1] <= 0) for k in range(7)])) if T >= 80 else 1.0

    dsr = deflated_sharpe(net.values, ledger=ledger, family=family)

    n4 = T // 4
    wf = [round(_sr(net.iloc[k * n4:(k + 1) * n4]) * ann, 2) for k in range(4)]

    return {"tf": tf, "ic": ic, "gross_sr_ann": _sr(gross) * ann,
            "net_sr_ann": obs, "turnover": turnover,
            "cv_folds_positive": cv_pos, "cv_n": len(fold),
            "cv_sr_mean": float(np.nanmean(fold)) if fold else float("nan"),
            "placebo_p": placebo_p, "surrogate_p": surrogate_p, "pbo": pbo,
            "dsr_family": dsr["dsr"], "dsr_n_trials": dsr["n_trials"],
            "walk_forward": wf,
            "max_drawdown": float(((1 + net).cumprod() / (1 + net).cumprod().cummax() - 1).min())}


# --------------------------------------------------------------------------- #
#  O — failure taxonomy                                                       #
# --------------------------------------------------------------------------- #
def diagnose(screen: dict, gate: dict | None) -> tuple[str, str]:
    if gate is None:
        if screen.get("short_tf_dead"):
            return "COST_BOUND", "gross edge exists at H1+ but < 3x cost at M3-M15"
        return "NO_EDGE", f"screen ratio {screen.get('best_ratio', 0):.2f} < 3 at every TF"
    g = gate
    if g["surrogate_p"] >= 0.05 or g["placebo_p"] >= 0.05:
        return "NO_EDGE", f"placebo p={g['placebo_p']:.3f}, surrogate p={g['surrogate_p']:.3f}"
    if g["gross_sr_ann"] > 0.5 and g["net_sr_ann"] <= 0:
        return "COST_BOUND", f"gross Sharpe {g['gross_sr_ann']:.2f} but net {g['net_sr_ann']:.2f}"
    if g["pbo"] >= 0.5:
        return "OVERFIT", f"PBO {g['pbo']:.2f}; WF {g['walk_forward']}"
    if abs(g["ic"]) < 0.010 and g["net_sr_ann"] > 0.5:
        return "METRIC_MISMATCH", f"net Sharpe {g['net_sr_ann']:.2f} but |rank-IC| {abs(g['ic']):.4f} < 0.010"
    if g["dsr_family"] < 0.95:
        return "NO_EDGE", f"DSR {g['dsr_family']:.2f} < 0.95 vs {g['dsr_n_trials']} family trials"
    if any(x < -0.5 for x in g["walk_forward"]):
        return "REGIME_DEPENDENT", f"walk-forward {g['walk_forward']}"
    return "NO_EDGE", "cleared gate individually — check cross-venue before PASS"


def verdict_of(gate: dict | None) -> str:
    if gate is None:
        return "SCREEN_FAIL"
    ok = (gate["surrogate_p"] < 0.05 and gate["placebo_p"] < 0.05
          and gate["cv_folds_positive"] >= 4 and gate["cv_sr_mean"] > 0
          and gate["net_sr_ann"] >= 1.0 and gate["turnover"] <= 0.05
          and gate["pbo"] < 0.5 and gate["dsr_family"] >= 0.95
          and not any(x < -0.5 for x in gate["walk_forward"]))
    return "PASS" if ok else "FAIL"
