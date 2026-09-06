"""S3-S6 study engine for a single cross-sectional hypothesis.

Pipeline (all mandatory, CLAUDE.md + user brief):
  build causal signal
  -> cross-sectional long/short positions, dollar-neutral, gross = 1
  -> t+1 execution (position earns from t+2 bar; matches lab/engine off-by-one)
  -> real costs from venue.yaml + measured slippage
  -> purged K-fold + embargo on TRAIN: per-fold OOS IC & net Sharpe
  -> placebo: cross-sectionally shuffled ranking (random long/short)
  -> surrogate: block-shuffled forward returns -> noise ceiling on |IC|
  -> DSR: net pnl, trial count READ FROM LEDGER (cumulative, never reset)
  -> PBO across the pre-registered config grid
  -> gate verdicts written to the ledger
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lab.engine.cv import purged_kfold
from lab.ledger import Ledger
from lab.research.panel import Panel, REGISTRY
from lab.stats.dsr import deflated_sharpe
from lab.stats.pbo import cscv_pbo

ROOT = Path(__file__).resolve().parents[2]
VENUE = yaml.safe_load((ROOT / "config" / "venue.yaml").read_text())
GATES = yaml.safe_load((ROOT / "config" / "gates.yaml").read_text())


def cost_per_unit_turnover() -> float:
    f = VENUE["fees"]["taker_fee"]
    slip = VENUE.get("slippage_bps")
    if slip is None:
        raise RuntimeError("venue.yaml slippage_bps is null - measure it first (S1)")
    return float(f) + float(slip) * 1e-4


# ---- positions + backtest ------------------------------------------------

def signal_to_positions(sig: pd.DataFrame, q: float = 0.2, mode: str = "decile") -> pd.DataFrame:
    s = sig.copy()
    if mode == "zscore":
        w = s.sub(s.mean(axis=1), axis=0)
        w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0)
        return w.fillna(0.0)
    r = s.rank(axis=1, pct=True)
    long = (r >= 1 - q).astype(float)
    short = (r <= q).astype(float)
    nl = long.sum(axis=1).replace(0, np.nan)
    ns = short.sum(axis=1).replace(0, np.nan)
    w = long.div(nl, axis=0) - short.div(ns, axis=0)          # +1 gross long, -1 gross short
    return (w / 2.0).fillna(0.0)                               # sum|w| = 1


def bt(close: pd.DataFrame, pos: pd.DataFrame, cost: float,
       funding: pd.DataFrame | None = None, execution_lag: int = 1) -> dict:
    bwd = close / close.shift(1) - 1.0
    held = pos.shift(execution_lag + 1)                        # signal t -> earn from t+2
    gross = (held * bwd).sum(axis=1)
    dpos = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    tc = dpos * cost
    fund = pd.Series(0.0, index=close.index)
    if funding is not None:
        fund = -(held * funding.reindex_like(held)).sum(axis=1)
    net = gross - tc + fund
    net = net.dropna()
    gross = gross.reindex_like(net)

    def sr(x):
        x = x[np.isfinite(x)]
        return float(x.mean() / x.std(ddof=1)) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0

    ann = np.sqrt(365 * 24 / VENUE_BAR_HOURS)
    return {"net": net, "gross_series": gross,
            "sharpe_net": sr(net), "sharpe_gross": sr(gross),
            "sharpe_net_ann": sr(net) * ann, "sharpe_gross_ann": sr(gross) * ann,
            "turnover_mean": float(dpos.mean()), "cost_drag": float(tc.mean()),
            "total_net": float(net.sum()), "total_gross": float(gross.sum())}


VENUE_BAR_HOURS = 1.0  # set by run_study per horizon


def rank_ic(sig: pd.DataFrame, fwd: pd.DataFrame) -> float:
    a = sig.rank(axis=1)
    b = fwd.rank(axis=1)
    ic = a.corrwith(b, axis=1)
    return float(ic.mean(skipna=True))


# ---- the study ---------------------------------------------------------

def run_study(name: str, feature: str, params: dict, horizon_bars: int,
              panel_train: Panel, *, family: str, hyp_id: str, exp_id: str,
              ledger: Ledger, q: float = 0.2, mode: str = "decile",
              n_folds: int = 6, embargo: float = 0.02, n_placebo: int = 200,
              n_surrogate: int = 200, seed: int = 0) -> dict:
    global VENUE_BAR_HOURS
    VENUE_BAR_HOURS = panel_train.bar_hours
    fn, fam = REGISTRY[feature]
    assert fam == family, (feature, fam, family)
    cost = cost_per_unit_turnover()
    rng = np.random.default_rng(seed)

    sig = fn(panel_train, **params)
    fwd = panel_train.fwd_ret(horizon_bars)
    valid = panel_train.mask_valid()
    sig = sig.where(valid, np.nan)

    pos = signal_to_positions(sig, q=q, mode=mode)
    res = bt(panel_train.close, pos, cost, panel_train.funding)
    ic_full = rank_ic(sig, panel_train.fwd_ret(1))

    # ---- purged K-fold + embargo (by row index; label window = horizon_bars) ----
    T = len(panel_train.index)
    t0 = np.arange(T)
    t1 = np.minimum(t0 + horizon_bars, T - 1)
    fold_ic, fold_sr = [], []
    net_by_fold = []
    for tr, te in purged_kfold(t0, t1, n_splits=n_folds, embargo_pct=embargo):
        if len(te) < 30:
            continue
        idx = panel_train.index[np.sort(te)]
        s_te = sig.loc[idx]
        f_te = panel_train.fwd_ret(1).loc[idx]
        fold_ic.append(rank_ic(s_te, f_te))
        p_te = signal_to_positions(s_te, q=q, mode=mode)
        r_te = bt(panel_train.close.loc[idx], p_te, cost, None)
        fold_sr.append(r_te["sharpe_net"])
        net_by_fold.append(r_te["net"])

    # ---- placebo: random cross-sectional ranking ----
    plc_sr = []
    base_names = sig.notna()
    for _ in range(n_placebo):
        noise = pd.DataFrame(rng.normal(size=sig.shape), index=sig.index, columns=sig.columns)
        noise = noise.where(base_names, np.nan)
        p = signal_to_positions(noise, q=q, mode=mode)
        plc_sr.append(bt(panel_train.close, p, cost, panel_train.funding)["sharpe_net"])
    plc_sr = np.array(plc_sr)
    placebo_p = float((np.sum(plc_sr >= res["sharpe_net"]) + 1) / (n_placebo + 1))

    # ---- surrogate: block-shuffle forward returns, recompute |IC| ----
    f1 = panel_train.fwd_ret(1)
    obs_absic = abs(rank_ic(sig, f1))
    block = max(horizon_bars * 3, 24)
    surr = []
    arr = f1.values
    for _ in range(n_surrogate):
        nb = int(np.ceil(T / block))
        starts = rng.integers(0, T, nb)
        order = np.concatenate([np.arange(s, s + block) % T for s in starts])[:T]
        shuff = pd.DataFrame(arr[order], index=f1.index, columns=f1.columns)
        surr.append(abs(rank_ic(sig, shuff)))
    surr = np.array(surr)
    surr_q99 = float(np.nanquantile(surr, 0.99))
    surr_p = float((np.sum(surr >= obs_absic) + 1) / (n_surrogate + 1))

    # ---- DSR: trial count FROM LEDGER, whole-lab cumulative (R3) ----
    # deflated_sharpe() calls ledger.trial_count() itself; family=None/hyp_id=None
    # => the cumulative count across the entire lab, never reset.
    dsr = deflated_sharpe(res["net"].values, ledger=ledger)
    n_trials_ledger = dsr["n_trials"]

    out = {
        "name": name, "feature": feature, "params": params, "family": family,
        "horizon_bars": horizon_bars, "q": q, "mode": mode,
        "cost_per_turnover": cost,
        "ic_1bar": ic_full,
        "sharpe_net": res["sharpe_net"], "sharpe_gross": res["sharpe_gross"],
        "sharpe_net_ann": res["sharpe_net_ann"], "sharpe_gross_ann": res["sharpe_gross_ann"],
        "turnover_mean": res["turnover_mean"], "total_net": res["total_net"],
        "cv_fold_ic": fold_ic, "cv_ic_mean": float(np.nanmean(fold_ic)) if fold_ic else np.nan,
        "cv_fold_sharpe": fold_sr, "cv_sharpe_mean": float(np.nanmean(fold_sr)) if fold_sr else np.nan,
        "placebo_sharpe_q95": float(np.nanquantile(plc_sr, 0.95)),
        "placebo_p_value": placebo_p,
        "surrogate_obs_absic": obs_absic, "surrogate_q99": surr_q99, "surrogate_p_value": surr_p,
        "dsr": dsr["dsr"], "dsr_n_trials": n_trials_ledger,
        "dsr_benchmark_sr": dsr["sr_deflation_benchmark"],
        "net_pnl_daily": res["net"],
    }
    # log metrics
    for k in ["ic_1bar", "sharpe_net", "sharpe_gross", "sharpe_net_ann", "cv_ic_mean",
              "cv_sharpe_mean", "placebo_p_value", "surrogate_obs_absic", "surrogate_q99",
              "surrogate_p_value", "dsr", "turnover_mean"]:
        v = out[k]
        if v == v:
            ledger.log_metric(exp_id, k, float(v), context={"study": name})
    ledger.log_metric(exp_id, "dsr_n_trials", float(n_trials_ledger), context={"study": name})
    return out
