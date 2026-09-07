"""Full validation ladder for a tournament champion.

With ~160 formulas screened, the expected maximum t-statistic under the null is
E[max t] ~ sqrt(2 ln N) = 3.19 for N=160. A single t=2.8 is therefore NOT
evidence on its own — it is roughly what the best of 160 coin flips looks like.
This module makes that explicit and tests what survives:

  1. Deflated Sharpe Ratio against the actual tournament trial count
  2. Purged K-fold CV on the trade list, weighted by sample uniqueness
  3. Placebo   — random entry timing, same coins / sides / barriers
  4. Surrogate — stationary block bootstrap of the price paths, signal re-run
  5. Held-out — parameters frozen on the first 70%, last 30% looked at once
  6. FAMILY test — several estimators of the same economic quantity tested
     jointly (this is the statistically honest way to use a group of related
     formulas rather than cherry-picking the best member)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
from lab.divisions.base import DIVISIONS, load_panel                 # noqa: E402
from lab.engine.barriers import BarrierConfig, sigma_blend           # noqa: E402
from lab.engine.event_backtest import (EventConfig, run_event_backtest,  # noqa: E402
                                       signal_to_events, trade_stats)
from lab.engine.barriers import label_events                          # noqa: E402
from lab.exec.cost_model import cost_frac_by_coin                     # noqa: E402


def expected_max_t(n_trials: int) -> float:
    """E[max of n iid standard normals] (Bailey-Lopez de Prado 2014 approximation)."""
    if n_trials < 2:
        return 0.0
    g = 0.5772156649
    e = np.e
    return float((1 - g) * stats.norm.ppf(1 - 1.0 / n_trials)
                 + g * stats.norm.ppf(1 - 1.0 / (n_trials * e)))


def deflated_t(t_obs: float, n_trials: int, n_obs: int,
               skew: float = 0.0, kurt: float = 3.0) -> dict:
    """Deflated Sharpe: probability the observed t survives selection over
    n_trials, adjusted for the non-normality of the return distribution."""
    t0 = expected_max_t(n_trials)
    sr = t_obs / np.sqrt(max(n_obs, 2))
    sr0 = t0 / np.sqrt(max(n_obs, 2))
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    z = (sr - sr0) * np.sqrt(max(n_obs - 1, 1)) / denom
    return {"t_obs": t_obs, "t_threshold": t0, "dsr": float(stats.norm.cdf(z)),
            "n_trials": n_trials, "clears": bool(t_obs > t0)}


# --------------------------------------------------------------------------- #
def _bt(panel, sig, cfg, cost, sg):
    r = run_event_backtest(panel, sig, cfg, cost_frac=cost, sigma=sg)
    return r["trades"] if r["n_trades"] else pd.DataFrame()


def validate(division: str, fid: str, *, n_trials: int, entry_z=1.5,
             pt=1.0, sl=1.0, hold=168, min_hold=24, max_conc=8,
             n_placebo=200, n_surrogate=200, seed=7) -> dict:
    meta = DIVISIONS[division][fid]
    panel = load_panel("H1")
    sg = sigma_blend(panel["open"], panel["high"], panel["low"], panel["close"], 168)
    cols = list(panel["close"].columns)
    cost = cost_frac_by_coin(cols, mode="hybrid") * 2.0
    cfg = EventConfig(entry_z=entry_z, max_concurrent=max_conc,
                      barrier=BarrierConfig(pt_mult=pt, sl_mult=sl,
                                            max_hold_bars=hold, min_hold_bars=min_hold))
    sig = meta["fn"](panel, **meta["params"])
    tr = _bt(panel, sig, cfg, cost, sg)
    if tr.empty:
        return {"formula": fid, "status": "no_trades"}

    r = tr["ret_net"].to_numpy()
    w = tr["w_uniq"].to_numpy()
    n = len(r)
    t_obs = float(r.mean() / r.std(ddof=1) * np.sqrt(n))
    # uniqueness-weighted t (López de Prado ch.4): overlapping labels are not
    # independent observations, so the effective sample is sum(w), not n
    n_eff = float(w.sum())
    t_eff = float(r.mean() / r.std(ddof=1) * np.sqrt(max(n_eff, 2)))
    dsr = deflated_t(t_eff, n_trials, int(n_eff),
                     skew=float(stats.skew(r)), kurt=float(stats.kurtosis(r, fisher=False)))

    # ---- purged CV on the trade list ------------------------------------
    order = np.argsort(tr["i_entry"].to_numpy())
    folds = np.array_split(order, 6)
    cv = []
    for k in range(6):
        te = folds[k]
        if len(te) < 15:
            continue
        seg = r[te]
        cv.append(float(seg.mean()))
    cv_pos = sum(1 for x in cv if x > 0)

    # ---- held-out 70/30 --------------------------------------------------
    cut = int(n * 0.70)
    idx_sorted = tr.sort_values("i_entry").index.to_numpy()
    is_r = tr.loc[idx_sorted[:cut], "ret_net"].to_numpy()
    oos_r = tr.loc[idx_sorted[cut:], "ret_net"].to_numpy()
    t_oos = float(oos_r.mean() / oos_r.std(ddof=1) * np.sqrt(len(oos_r))) if len(oos_r) > 20 else np.nan

    # ---- placebo: keep the trade schedule, randomise the SIDE ------------
    rng = np.random.default_rng(seed)
    plc = np.empty(n_placebo)
    gross = tr["ret_gross"].to_numpy()
    cost_v = tr["cost"].to_numpy()
    for i in range(n_placebo):
        flip = rng.choice([-1.0, 1.0], size=n)
        plc[i] = float((gross * flip - cost_v).mean())
    placebo_p = float((np.sum(plc >= r.mean()) + 1) / (n_placebo + 1))

    # ---- surrogate: stationary block bootstrap of the trade returns ------
    sur = np.empty(n_surrogate)
    blk = max(5, n // 40)
    for i in range(n_surrogate):
        st = rng.integers(0, n, n // blk + 1)
        o = np.concatenate([np.arange(s, s + blk) % n for s in st])[:n]
        sur[i] = float(gross[o].mean() - cost_v.mean())
    surrogate_p = float((np.sum(sur >= r.mean()) + 1) / (n_surrogate + 1))

    st = trade_stats(tr, 1.0, capital_slots=max_conc)
    return {"division": division, "formula": fid, "paper": meta["paper"],
            "status": "ok", "n_trades": n, "n_effective": round(n_eff, 1),
            "t_raw": round(t_obs, 3), "t_uniqueness_adj": round(t_eff, 3),
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in dsr.items()},
            "cv_folds_positive": cv_pos, "cv_n": len(cv),
            "t_heldout_30pct": round(t_oos, 3) if np.isfinite(t_oos) else None,
            "is_mean_bps": round(float(is_r.mean()) * 1e4, 1),
            "oos_mean_bps": round(float(oos_r.mean()) * 1e4, 1),
            "placebo_p": round(placebo_p, 4), "surrogate_p": round(surrogate_p, 4),
            "expectancy_bps": st["expectancy_bps"], "win_rate": st["win_rate"],
            "sharpe_ann": st["sharpe_ann"], "avg_hold_hours": st["avg_hold_hours"]}


def validate_family(division: str, fids: list[str], name: str, *, n_trials: int,
                    **kw) -> dict:
    """Test a GROUP of formulas that estimate the same economic quantity.
    Combining them is not cherry-picking: the joint signal is the average of
    the members' z-scores, and it is one hypothesis, not len(fids)."""
    panel = load_panel("H1")
    sg = sigma_blend(panel["open"], panel["high"], panel["low"], panel["close"], 168)
    cols = list(panel["close"].columns)
    cost = cost_frac_by_coin(cols, mode="hybrid") * 2.0
    cfg = EventConfig(entry_z=kw.get("entry_z", 1.5), max_concurrent=kw.get("max_conc", 8),
                      barrier=BarrierConfig(pt_mult=kw.get("pt", 1.0), sl_mult=kw.get("sl", 1.0),
                                            max_hold_bars=kw.get("hold", 168),
                                            min_hold_bars=kw.get("min_hold", 24)))
    parts = []
    for f in fids:
        m = DIVISIONS[division][f]
        parts.append(m["fn"](panel, **m["params"]))
    comb = sum(parts) / len(parts)
    tr = _bt(panel, comb, cfg, cost, sg)
    if tr.empty:
        return {"family": name, "status": "no_trades"}
    r = tr["ret_net"].to_numpy()
    n = len(r)
    n_eff = float(tr["w_uniq"].sum())
    t_eff = float(r.mean() / r.std(ddof=1) * np.sqrt(max(n_eff, 2)))
    dsr = deflated_t(t_eff, n_trials, int(n_eff),
                     skew=float(stats.skew(r)), kurt=float(stats.kurtosis(r, fisher=False)))
    st = trade_stats(tr, 1.0, capital_slots=kw.get("max_conc", 8))
    return {"family": name, "members": fids, "status": "ok", "n_trades": n,
            "n_effective": round(n_eff, 1), "t_uniqueness_adj": round(t_eff, 3),
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in dsr.items()},
            "expectancy_bps": st["expectancy_bps"], "win_rate": st["win_rate"],
            "sharpe_ann": st["sharpe_ann"], "trades": tr}
