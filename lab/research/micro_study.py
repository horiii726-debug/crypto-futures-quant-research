"""Time-series microstructure study (fast, numpy). One signal, one bar size.

Full gate ladder: purged CV + embargo, placebo (per-coin sign flip), surrogate
(block-shuffle bar returns), family-scoped DSR, taker cost cross-check.

R9: signal at bar t -> position earns the close-to-close return realised at
bar t+2 (decide at close t, fill during t+1, hold to t+2). `hold` bars smears
the position forward to cut turnover.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lab.engine.cv import purged_kfold
from lab.ledger import Ledger
from lab.research.micro import REGISTRY, FAMILY
from lab.stats.dsr import deflated_sharpe

ROOT = Path(__file__).resolve().parents[2]
_EXEC = json.loads((ROOT / "data" / "processed" / "exec_summary.json").read_text())
MAKER_ONEWAY = _EXEC["hybrid_roundtrip_bps_mean"] / 2.0 * 1e-4
TAKER_ONEWAY = (0.0005 + 0.85e-4)
BARS_PER_YEAR = {3: 365 * 24 * 20, 5: 365 * 24 * 12, 15: 365 * 24 * 4}


def _stack(tape_panel: dict, feat_fn, params: dict, hold: int, thr: float):
    P, R = [], []
    idx = None
    for s, df in tape_panel.items():
        if len(df) < 500:
            continue
        sig = feat_fn(df, **params).astype(float)
        pos = sig.where(sig.abs() >= thr, 0.0)
        if hold > 1:
            pos = pos.rolling(hold, min_periods=1).mean()
        pos = pos.clip(-1, 1)
        ret = np.log(df["close"]).diff()
        P.append(pos.rename(s))
        R.append(ret.rename(s))
    Pdf = pd.concat(P, axis=1).sort_index()
    Rdf = pd.concat(R, axis=1).sort_index()
    return Pdf, Rdf


def _sr(x):
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1)) if x.size > 5 and x.std(ddof=1) > 0 else 0.0


def _pooled_ic(P: np.ndarray, Rf: np.ndarray) -> float:
    a, b = P.ravel(), Rf.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100 or a[m].std() == 0 or b[m].std() == 0:
        return 0.0
    return float(np.corrcoef(a[m], b[m])[0, 1])


def run_micro_study(name, feature, params, *, bar_min, hold, thr, tape_panel,
                    ledger: Ledger, exp_id, hyp_id, n_placebo=150, n_surrogate=150,
                    seed=0, n_folds=6, embargo=0.02) -> dict:
    fn = REGISTRY[feature]
    fam = FAMILY[feature]
    rng = np.random.default_rng(seed)
    Pdf, Rdf = _stack(tape_panel, fn, params, hold, thr)
    idx = Pdf.index
    Pv = np.nan_to_num(Pdf.values)                 # (T,N)
    Rv = np.nan_to_num(Rdf.values)
    T, N = Pv.shape
    held = np.zeros_like(Pv)
    held[2:] = Pv[:-2]                             # R9: decide t -> earn t+2
    gross_bar = np.nanmean(held * Rv, axis=1)
    dheld = np.zeros_like(held)
    dheld[1:] = np.abs(held[1:] - held[:-1])
    dheld[0] = np.abs(held[0])
    turn_bar = np.nanmean(dheld, axis=1)
    ann = np.sqrt(BARS_PER_YEAR[bar_min])

    def score(oneway):
        net = gross_bar - turn_bar * oneway
        net = net[np.isfinite(net)]
        return _sr(net), net

    maker_sr, maker_net = score(MAKER_ONEWAY)
    taker_sr, _ = score(TAKER_ONEWAY)
    gross_sr = _sr(gross_bar[np.isfinite(gross_bar)])

    Rf = np.full_like(Rv, np.nan)
    Rf[:-2] = Rv[2:]
    ic = _pooled_ic(Pv, Rf)

    # purged CV on the bar index
    valid = np.isfinite(gross_bar)
    t0 = np.arange(T)
    t1 = np.minimum(t0 + hold + 2, T - 1)
    net_maker_full = gross_bar - turn_bar * MAKER_ONEWAY
    fold_sr = []
    for tr, te in purged_kfold(t0[valid], t1[valid], n_splits=n_folds, embargo_pct=embargo):
        if len(te) < 50:
            continue
        seg = net_maker_full[np.sort(te)]
        seg = seg[np.isfinite(seg)]
        if seg.size > 5 and seg.std() > 0:
            fold_sr.append(_sr(seg))

    # placebo: per-coin sign flip (keeps timing, randomises direction)
    absP = np.abs(Pv)
    plc = np.empty(n_placebo)
    for k in range(n_placebo):
        fl = rng.choice(np.array([-1.0, 1.0]), size=N)
        h = np.zeros_like(absP)
        h[2:] = (absP * fl[None, :])[:-2]
        gb = np.nanmean(h * Rv, axis=1)
        db = np.zeros_like(h); db[1:] = np.abs(h[1:] - h[:-1]); db[0] = np.abs(h[0])
        tb = np.nanmean(db, axis=1)
        nn = (gb - tb * MAKER_ONEWAY)
        plc[k] = _sr(nn[np.isfinite(nn)])
    placebo_p = float((np.sum(plc >= maker_sr) + 1) / (n_placebo + 1))

    # surrogate: block-shuffle the return rows, recompute pooled IC
    surr = np.empty(n_surrogate)
    blk = 20
    for k in range(n_surrogate):
        st = rng.integers(0, T, int(np.ceil(T / blk)))
        order = np.concatenate([np.arange(s, s + blk) % T for s in st])[:T]
        Rsh = Rv[order]
        Rfsh = np.full_like(Rsh, np.nan)
        Rfsh[:-2] = Rsh[2:]
        surr[k] = abs(_pooled_ic(Pv, Rfsh))
    surr_p = float((np.sum(surr >= abs(ic)) + 1) / (n_surrogate + 1))

    dsr = deflated_sharpe(maker_net, ledger=ledger, family=fam)
    dsr_g = deflated_sharpe(maker_net, ledger=ledger)

    out = {
        "name": name, "feature": feature, "family": fam, "params": params,
        "bar_min": bar_min, "hold": hold, "thr": thr,
        "ic_fwd2": ic, "maker_sr_net_ann": maker_sr * ann, "maker_sr_gross_ann": gross_sr * ann,
        "taker_sr_net_ann": taker_sr * ann,
        "turnover_per_bar": float(np.nanmean(turn_bar)),
        "cost_drag_per_bar_bps": float(np.nanmean(turn_bar) * MAKER_ONEWAY * 1e4),
        "total_net_maker": float(np.nansum(maker_net)),
        "cv_sr_mean": float(np.nanmean(fold_sr)) if fold_sr else np.nan, "cv_folds": len(fold_sr),
        "placebo_p": placebo_p, "surrogate_p": surr_p,
        "surrogate_q99": float(np.nanquantile(surr, 0.99)),
        "dsr_family": dsr["dsr"], "dsr_family_n": dsr["n_trials"],
        "dsr_portfolio": dsr_g["dsr"], "dsr_portfolio_n": dsr_g["n_trials"],
        "n_bars": int(valid.sum()), "n_symbols": int(N),
    }
    for k in ["ic_fwd2", "maker_sr_net_ann", "maker_sr_gross_ann", "taker_sr_net_ann",
              "cv_sr_mean", "placebo_p", "surrogate_p", "dsr_family", "dsr_portfolio",
              "turnover_per_bar", "cost_drag_per_bar_bps"]:
        if out[k] == out[k]:
            ledger.log_metric(exp_id, k, float(out[k]), context={"study": name})
    return out
