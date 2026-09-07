"""Hardened red-team validation of the single strongest lead in the project:

    F_OI :: oi_leverage_state
    lev_{i,t} = OI_{i,t} * price_{i,t} / mean_n(quote_volume_{i,t})
    signal    = -zscore_xsec( log(lev) )      # short the high-leverage names

In the F_OI sweep this was the only signal with a POSITIVE net Sharpe that also
cleared the surrogate noise ceiling (maker ~+1.1, taker ~+1.0, gross ~+1.3,
surrogate p ~0.008).  It still FAILED the ladder on |IC| (0.008 < 0.01) and on
family DSR (0.19, crushed by F_OI multiplicity).  Before writing the F_OI bound
we stress it properly:

  * SELECTION only on the first 70% of the panel; last 30% is a single held-out
    look (the F_OI test-seal is treated as already consumed by oi_study._panel,
    so this is an internal hold-out, reported once).
  * liquid-only universe (drop delisted / thin names that sank F_XVENUE)
  * per-coin P&L concentration (is it one name?)
  * monthly Sharpe path + 6-month walk-forward
  * taker cost stress (2x fee)  + turnover / capacity
  * q sensitivity, lookback sensitivity
  * placebo (per-coin sign flip) + block-shuffle surrogate
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lab.engine.cv import purged_kfold
from lab.ledger import Ledger
from lab.research.oi_study import _panel, _zx
from lab.stats.dsr import deflated_sharpe

ROOT = Path(__file__).resolve().parents[2]
_EXEC = json.loads((ROOT / "data" / "processed" / "exec_summary.json").read_text())
_V1_MK = _EXEC["hybrid_roundtrip_bps_mean"] / 2 * 1e-4       # RESEARCH ROUND 1 flat
_V1_TK = 0.0005 + 0.85e-4
ANN = np.sqrt(365 * 24)

# names with a continuous, liquid Binance perp for the whole 2y window
LIQUID = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT",
          "LINKUSDT", "AVAXUSDT", "ADAUSDT", "LTCUSDT", "BCHUSDT", "DOTUSDT",
          "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "INJUSDT",
          "FILUSDT", "LDOUSDT", "CRVUSDT", "XLMUSDT", "ICPUSDT", "UNIUSDT",
          "AAVEUSDT", "XMRUSDT", "DASHUSDT", "ZECUSDT"]

try:                                                         # universe-mean cost for placebo/CV
    from lab.exec.cost_model import cost_frac_by_coin as _cfbc
    MK = float(np.nanmean(_cfbc(LIQUID, mode="hybrid").values))
    TK = float(np.nanmean(_cfbc(LIQUID, mode="taker").values))
except Exception:
    MK, TK = _V1_MK, _V1_TK


def _lev_signal(close, qv, oi, n=168):
    lev = oi * close / qv.rolling(n, min_periods=n // 2).mean().replace(0, np.nan)
    return -_zx(np.log(lev))


def _positions(sig, q=0.2):
    r = sig.rank(axis=1, pct=True)
    lo = (r >= 1 - q).astype(float)
    sh = (r <= q).astype(float)
    nl = lo.sum(axis=1).replace(0, np.nan)
    ns = sh.sum(axis=1).replace(0, np.nan)
    return (lo.div(nl, axis=0) - sh.div(ns, axis=0)).fillna(0.0) / 2.0


def _sr(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / x.std(ddof=1))


def _cost_vecs(cols):
    """RESEARCH ROUND 2 · P0.2 — per-coin one-way cost fractions."""
    try:
        from lab.exec.cost_model import cost_frac_by_coin
        mk = cost_frac_by_coin(list(cols), mode="hybrid").reindex(cols)
        tk = cost_frac_by_coin(list(cols), mode="taker").reindex(cols)
        return mk.fillna(mk.median()), tk.fillna(tk.median())
    except Exception:
        return (pd.Series(_V1_MK, index=cols), pd.Series(_V1_TK, index=cols))


def _run_cfg(close, qv, oi, n, q, hold):
    sig = _lev_signal(close, qv, oi, n=n)
    valid = close.notna().sum(axis=1) >= 8
    sig = sig.where(valid)
    pos = _positions(sig, q=q)
    if hold > 1:
        pos = pos.rolling(hold, min_periods=1).mean()
    bwd = close / close.shift(1) - 1.0
    held = pos.shift(2)                                  # R9
    per_coin_gross = held * bwd
    gross = per_coin_gross.sum(axis=1)
    turn_coin = held.diff().abs()
    turn_coin = turn_coin.fillna(held.abs())
    turn = turn_coin.sum(axis=1)
    mk_vec, tk_vec = _cost_vecs(close.columns)
    net_mk = (gross - turn_coin.mul(mk_vec, axis=1).sum(axis=1)).dropna()
    net_tk = (gross - turn_coin.mul(tk_vec, axis=1).sum(axis=1)).dropna()
    fwd1 = close.shift(-1) / close - 1.0
    ic = float(sig.rank(axis=1).corrwith(fwd1.rank(axis=1), axis=1).mean(skipna=True))
    return dict(gross=gross, net_mk=net_mk, net_tk=net_tk, turn=turn,
                per_coin=per_coin_gross, ic=ic, held=held)


def run(ledger: Ledger, parent_hyp: str | None = None) -> dict:
    close_all, qv_all, oi_all = _panel("1h")
    T = len(close_all)
    cut = int(T * 0.70)
    sel = slice(0, cut)
    oos = slice(cut, T)

    hyp = ledger.add_hypothesis({
        "claim": "oi_leverage_state (OI-notional / avg-volume, shorted) is a tradeable "
                 "cross-sectional perp factor net of hybrid cost",
        "mechanism": "high open-interest relative to traded volume = crowded / high "
                     "effective leverage = fragile to funding & liquidation cascades = "
                     "underperforms; low OI/volume names outperform",
        "math_form": "sig = -zscore_xsec(log(OI*px / rollmean_n(quote_vol)))",
        "target": "xsec_fwd_return_1h", "horizon": "6-72h hold",
        "null_hypothesis": "IC=0 ; net maker Sharpe <= 0 ; edge concentrated in <=2 names",
        "family": "F_OI", "lessons_reviewed": True,
        **({"parent_id": parent_hyp} if parent_hyp else {})})
    exp = ledger.new_experiment(hyp, "F_OI",
                                {"study": "oi_leverage_hardened", "sel_frac": 0.70,
                                 "universe": "liquid-28"}, stage="discovery")

    out = {"hyp_id": hyp, "exp_id": exp, "configs": [], "oos": {}, "concentration": {}}

    # ---- 1. selection sweep on the first 70% (liquid-only) ----------------
    liq = [c for c in LIQUID if c in close_all.columns]
    cA, qA, oA = close_all[liq].iloc[sel], qv_all[liq].iloc[sel], oi_all[liq].iloc[sel]
    best = None
    for n in (120, 168, 240, 336):
        for q in (0.15, 0.2, 0.3):
            for hold in (6, 24, 72):
                tid = ledger.log_trial(exp, {"n": n, "q": q, "hold": hold, "phase": "select"},
                                       stage="discovery", status="running")
                r = _run_cfg(cA, qA, oA, n, q, hold)
                smk = _sr(r["net_mk"]) * ANN
                stk = _sr(r["net_tk"]) * ANN
                sgr = _sr(r["gross"]) * ANN
                ledger.log_metric(exp, "sel_maker_sr_ann", float(smk), trial_id=tid)
                ledger.log_metric(exp, "sel_taker_sr_ann", float(stk), trial_id=tid)
                ledger.log_metric(exp, "sel_ic", float(r["ic"]), trial_id=tid)
                ledger.mark_trial(tid, "done")
                row = {"n": n, "q": q, "hold": hold, "sel_maker_sr": smk,
                       "sel_taker_sr": stk, "sel_gross_sr": sgr, "sel_ic": r["ic"],
                       "sel_turn": float(r["turn"].mean())}
                out["configs"].append(row)
                if best is None or smk > best["sel_maker_sr"]:
                    best = row
    out["best_selected"] = best

    # ---- 2. ONE held-out look with the frozen best config ----------------
    n, q, hold = best["n"], best["q"], best["hold"]
    cO, qO, oO = close_all[liq].iloc[oos], qv_all[liq].iloc[oos], oi_all[liq].iloc[oos]
    rO = _run_cfg(cO, qO, oO, n, q, hold)
    smk_o = _sr(rO["net_mk"]) * ANN
    stk_o = _sr(rO["net_tk"]) * ANN
    sgr_o = _sr(rO["gross"]) * ANN

    # full-sample (sel+oos) for path / concentration / DSR
    rF = _run_cfg(close_all[liq], qv_all[liq], oi_all[liq], n, q, hold)
    net = rF["net_mk"]

    # monthly Sharpe path
    m = net.copy()
    m.index = pd.to_datetime(m.index)
    monthly = m.groupby([m.index.year, m.index.month]).apply(lambda s: _sr(s) * ANN)
    # 6-month walk-forward
    k = len(net) // 4
    wf = [round(_sr(net.iloc[i * k:(i + 1) * k]) * ANN, 2) for i in range(4)]

    # per-coin P&L concentration
    pc = rF["per_coin"].reindex(net.index).sum().sort_values()
    tot = pc.sum()
    top1 = float(pc.abs().max() / pc.abs().sum()) if pc.abs().sum() else 1.0
    top3 = float(pc.abs().sort_values().iloc[-3:].sum() / pc.abs().sum()) if pc.abs().sum() else 1.0

    # placebo: per-coin random sign flip
    rng = np.random.default_rng(20260907)
    absP = np.abs(np.nan_to_num(rF["held"].values))
    bwdv = np.nan_to_num((close_all[liq] / close_all[liq].shift(1) - 1.0).reindex(rF["held"].index).values)
    obs = _sr(net)
    plc = np.empty(300)
    for i in range(300):
        fl = rng.choice([-1.0, 1.0], size=absP.shape[1])
        h = absP * fl[None, :]
        g = np.nansum(h * bwdv, axis=1)
        dh = np.abs(np.diff(h, axis=0, prepend=0.0))
        nn = g - dh.sum(axis=1) * MK
        plc[i] = _sr(nn)
    placebo_p = float((np.sum(plc >= obs) + 1) / 301)

    # surrogate: block-shuffle forward returns, re-score frozen positions
    sig = _lev_signal(close_all[liq], qv_all[liq], oi_all[liq], n=n).rank(axis=1)
    fwd1 = (close_all[liq].shift(-1) / close_all[liq] - 1.0).rank(axis=1).values
    srv = sig.values
    src = srv - np.nanmean(srv, axis=1, keepdims=True)
    ss = np.nansum(src ** 2, axis=1)
    L = len(fwd1)
    sur = np.empty(300)
    for i in range(300):
        st = rng.integers(0, L, int(np.ceil(L / 24)))
        order = np.concatenate([np.arange(x, x + 24) % L for x in st])[:L]
        fr = fwd1[order]
        frc = fr - np.nanmean(fr, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = np.nansum(src * frc, axis=1) / np.sqrt(ss * np.nansum(frc ** 2, axis=1))
        sur[i] = abs(float(np.nanmean(rr)))
    surrogate_p = float((np.sum(sur >= abs(rF["ic"])) + 1) / 301)

    dsr = deflated_sharpe(net.values, ledger=ledger, family="F_OI")

    # purged CV on the full liquid panel
    g = rF["gross"].dropna()
    idx = g.index
    t0 = np.arange(len(g))
    t1 = np.minimum(t0 + hold + 2, len(g) - 1)
    netF = (rF["gross"] - rF["turn"] * MK)
    fold = []
    for tr, te in purged_kfold(t0, t1, n_splits=6, embargo_pct=0.02):
        if len(te) < 50:
            continue
        seg = netF.reindex(idx[np.sort(te)]).dropna()
        if len(seg) > 5 and seg.std() > 0:
            fold.append(_sr(seg) * ANN)

    out["oos"] = {"n": n, "q": q, "hold": hold,
                  "oos_maker_sr_ann": smk_o, "oos_taker_sr_ann": stk_o,
                  "oos_gross_sr_ann": sgr_o, "oos_ic": rO["ic"],
                  "oos_turn": float(rO["turn"].mean())}
    out["full"] = {"maker_sr_ann": _sr(net) * ANN, "taker_sr_ann": _sr(rF["net_tk"]) * ANN,
                   "gross_sr_ann": _sr(rF["gross"]) * ANN, "ic": rF["ic"],
                   "turn_per_bar": float(rF["turn"].mean()),
                   "walk_forward": wf, "cv_folds": len(fold),
                   "cv_sr_mean": float(np.nanmean(fold)) if fold else float("nan"),
                   "placebo_p": placebo_p, "surrogate_p": surrogate_p,
                   "dsr_family": dsr["dsr"], "dsr_family_n": dsr["n_trials"],
                   "monthly_sr_min": float(monthly.min()), "monthly_sr_med": float(monthly.median()),
                   "monthly_sr_pos_frac": float((monthly > 0).mean())}
    out["concentration"] = {"per_coin_pnl": {k_: round(float(v), 5) for k_, v in pc.items()},
                            "top1_share": top1, "top3_share": top3,
                            "n_coins": len(liq)}

    for kk, vv in out["full"].items():
        if isinstance(vv, (int, float)) and vv == vv:
            ledger.log_metric(exp, kk, float(vv), context={"study": "oi_leverage_hardened"})

    # ---- verdict ----
    f = out["full"]
    o = out["oos"]
    passes = (
        f["surrogate_p"] < 0.05 and abs(f["ic"]) >= 0.010
        and o["oos_maker_sr_ann"] > 0.3 and o["oos_taker_sr_ann"] > -0.2
        and all(x > -0.3 for x in f["walk_forward"])
        and f["placebo_p"] < 0.05
        and out["concentration"]["top1_share"] < 0.45
        and f["dsr_family"] >= 0.90
        and f["cv_sr_mean"] > 0
    )
    status = "PASS" if passes else "FAIL"
    objs = []
    if abs(f["ic"]) < 0.010:
        objs.append(f"|IC|={abs(f['ic']):.4f}<0.010")
    if o["oos_maker_sr_ann"] <= 0.3:
        objs.append(f"held-out maker SR={o['oos_maker_sr_ann']:.2f}")
    if o["oos_taker_sr_ann"] <= -0.2:
        objs.append(f"held-out taker SR={o['oos_taker_sr_ann']:.2f}")
    if any(x <= -0.3 for x in f["walk_forward"]):
        objs.append(f"WF fold neg {f['walk_forward']}")
    if f["placebo_p"] >= 0.05:
        objs.append(f"placebo p={f['placebo_p']:.3f}")
    if out["concentration"]["top1_share"] >= 0.45:
        objs.append(f"top1 coin {out['concentration']['top1_share']:.0%} of gross P&L")
    if f["dsr_family"] < 0.90:
        objs.append(f"DSR(F_OI)={f['dsr_family']:.2f}")
    if f["cv_sr_mean"] != f["cv_sr_mean"] or f["cv_sr_mean"] <= 0:
        objs.append(f"CV SR={f['cv_sr_mean']:.2f}")
    obj = "; ".join(objs) if objs else "clears the hardened ladder"

    ledger.add_verdict(hyp, "final", status, "red-team", [exp],
                       rationale=(f"oi_leverage_state hardened: held-out(30%) maker SR "
                                  f"{o['oos_maker_sr_ann']:.2f} / taker {o['oos_taker_sr_ann']:.2f}, "
                                  f"full maker SR {f['maker_sr_ann']:.2f}, IC {f['ic']:.4f}, "
                                  f"WF {f['walk_forward']}, top1 {out['concentration']['top1_share']:.0%}, "
                                  f"DSR(F_OI) {f['dsr_family']:.2f}"),
                       strongest_surviving_objection=obj)
    if status != "PASS":
        ledger.add_lesson(root_cause=obj,
                          lesson=(f"F_OI oi_leverage_state is the strongest single lead in the "
                                  f"project (held-out maker SR {o['oos_maker_sr_ann']:.2f}, "
                                  f"taker {o['oos_taker_sr_ann']:.2f}, full-sample maker "
                                  f"{f['maker_sr_ann']:.2f}, gross {f['gross_sr_ann']:.2f}) but "
                                  f"still fails the hardened ladder: {obj}."),
                          family="F_OI", subject_id=hyp, ladder_level="L3")
    ledger.close_experiment(exp, "done")
    out["status"] = status
    out["objection"] = obj
    return out


if __name__ == "__main__":
    import os
    os.environ.setdefault("CRYPTO_LAB_LEDGER_SQLITE", "./ledger/lab.sqlite")
    L = Ledger()
    r = run(L)
    print(json.dumps({k: v for k, v in r.items()
                      if k in ("best_selected", "oos", "full", "status", "objection")},
                     indent=1, default=float))
    print("\nper-coin P&L:")
    for k, v in r["concentration"]["per_coin_pnl"].items():
        print(f"  {k:12} {v:+.4f}")
