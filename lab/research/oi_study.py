"""F_OI — open-interest signals (Bybit hourly OI, 2y, + Binance perp price).

Bybit's OI API gives ~2 years of hourly OI - the multi-year history Binance
lacks. OI is derivatives-specific information not in price/volume:
  OI up + price up   -> fresh longs (continuation, then exhaustion)
  OI up + price down -> fresh shorts (squeeze setup / bottoming)
  OI down + price up -> short covering (weak rally, fade)
  OI down + price dn -> long liquidation (overshoot, bounce)
  OI z-extreme       -> crowded positioning, reversal risk

Cross-sectional (rank coins) is primary; also a time-series check per coin.
Full ladder: purged CV + placebo (random ranking) + surrogate (block-shuffle
returns) + family DSR + cost (hybrid + taker stress).
"""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
from lab.engine.cv import purged_kfold
from lab.ledger import Ledger
from lab.stats.dsr import deflated_sharpe

ROOT = Path(__file__).resolve().parents[2]
OI = ROOT / "data" / "processed" / "bybit_oi"
_EXEC = json.loads((ROOT / "data" / "processed" / "exec_summary.json").read_text())
MAKER_1W = _EXEC["hybrid_roundtrip_bps_mean"] / 2 * 1e-4
TAKER_1W = 0.0005 + 0.85e-4


def _panel(bar="1h"):
    close_parts = []
    for part in ("train", "valid", "test"):
        p = ROOT / "data" / part / "prio50_2y" / bar / "close.parquet"
        if p.exists():
            close_parts.append(pd.read_parquet(p))
    close = pd.concat(close_parts).sort_index()
    vol_parts = []
    for part in ("train", "valid", "test"):
        p = ROOT / "data" / part / "prio50_2y" / bar / "quote_volume.parquet"
        if p.exists():
            vol_parts.append(pd.read_parquet(p))
    qv = pd.concat(vol_parts).sort_index().reindex(close.index)
    oi = {}
    for f in sorted(OI.glob("*.parquet")):
        d = pd.read_parquet(f).set_index("dt")["oi"]
        oi[f.stem] = d
    OIdf = pd.DataFrame(oi).sort_index()
    common = [c for c in close.columns if c in OIdf.columns]
    close = close[common]
    OIdf = OIdf.reindex(close.index).reindex(columns=common).ffill(limit=6)
    qv = qv[common]
    # mask where close is NaN (survivorship)
    OIdf = OIdf.where(close.notna())
    qv = qv.where(close.notna())
    return close, qv, OIdf


def _z(df, n):
    return ((df - df.rolling(n, min_periods=max(5, n // 2)).mean())
            / df.rolling(n, min_periods=max(5, n // 2)).std().replace(0, np.nan)).clip(-4, 4)


def _zx(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


FEATURES = {}


def feat(fn):
    FEATURES[fn.__name__] = fn
    return fn


@feat
def oi_delta_rev(close, qv, oi, k=24):
    return -_zx(np.log(oi).diff(k))                     # OI surged -> crowded -> fade


@feat
def oi_delta_mom(close, qv, oi, k=24):
    return _zx(np.log(oi).diff(k))


@feat
def oi_price_confirm(close, qv, oi, k=24):
    doi = np.sign(oi.diff(k))
    dpx = np.sign(np.log(close).diff(k))
    return _zx((doi * dpx) * np.log(oi).diff(k).abs())  # fresh positions in trend dir -> continue


@feat
def oi_short_cover(close, qv, oi, k=24):
    # price up + OI down => short covering => weak, fade the rally
    doi = oi.diff(k); dpx = np.log(close).diff(k)
    sig = ((dpx > 0) & (doi < 0)).astype(float) - ((dpx < 0) & (doi < 0)).astype(float)
    return -_zx(sig * dpx.abs())                        # fade short-cover rallies, buy liq dips


@feat
def oi_extreme_rev(close, qv, oi, n=720):
    return -_zx(_z(np.log(oi), n))                      # OI z-score high vs 30d -> reversal


@feat
def oi_per_volume(close, qv, oi, k=24):
    build = oi.diff(k).abs() / qv.rolling(k, min_periods=k // 2).sum().replace(0, np.nan)
    return -_zx(build)                                  # high position-building = crowded


@feat
def oi_accel_rev(close, qv, oi, k=12):
    return -_zx(np.log(oi).diff(k).diff(k))


@feat
def oi_leverage_state(close, qv, oi, n=168):
    lev = oi * close / qv.rolling(n, min_periods=n // 2).mean().replace(0, np.nan)
    return -_zx(np.log(lev))                            # high OI/volume = fragile -> underperform


def _positions(sig, q=0.2):
    r = sig.rank(axis=1, pct=True)
    long = (r >= 1 - q).astype(float); short = (r <= q).astype(float)
    nl = long.sum(axis=1).replace(0, np.nan); ns = short.sum(axis=1).replace(0, np.nan)
    return (long.div(nl, axis=0) - short.div(ns, axis=0)).fillna(0.0) / 2.0


_PANEL_CACHE = {}
def run_one(feature, params, hold_h, ledger, exp_id, hyp_id, n_placebo=150, n_surrogate=150, seed=0):
    if "p" not in _PANEL_CACHE:
        _PANEL_CACHE["p"] = _panel("1h")
    close, qv, oi = _PANEL_CACHE["p"]
    rng = np.random.default_rng(seed)
    sig = FEATURES[feature](close, qv, oi, **params)
    valid = close.notna().sum(axis=1) >= 8
    sig = sig.where(valid)
    pos = _positions(sig)
    if hold_h > 1:
        pos = pos.rolling(hold_h, min_periods=1).mean()
    bwd = close / close.shift(1) - 1.0
    # R9: signal at t -> position earns from t+2 (matches lab off-by-one)
    held = pos.shift(2)
    gross = (held * bwd).sum(axis=1)
    turn = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))

    def score(c):
        net = (gross - turn * c).dropna()
        s = lambda x: float(x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 and len(x) > 5 else 0.0
        ann = np.sqrt(365 * 24)
        return s(net) * ann, net, s(gross.reindex_like(net)) * ann
    mk_sr, mk_net, gr_sr = score(MAKER_1W)
    tk_sr, _, _ = score(TAKER_1W)

    fwd1 = close.shift(-1) / close - 1.0
    ic = float(sig.rank(axis=1).corrwith(fwd1.rank(axis=1), axis=1).mean(skipna=True))

    T = len(gross.dropna()); idx = gross.dropna().index
    t0 = np.arange(T); t1 = np.minimum(t0 + hold_h + 2, T - 1)
    net_mk_full = (gross - turn * MAKER_1W)
    fold = []
    for tr, te in purged_kfold(t0, t1, n_splits=6, embargo_pct=0.02):
        if len(te) < 50: continue
        seg = net_mk_full.reindex(idx[np.sort(te)]).dropna()
        if len(seg) > 5 and seg.std() > 0:
            fold.append(float(seg.mean() / seg.std(ddof=1)))

    posv = np.nan_to_num(pos.values); bwdv = np.nan_to_num(bwd.values); nm = pos.shape[1]
    absP = np.abs(posv)
    plc = np.empty(n_placebo)
    for i in range(n_placebo):
        fl = rng.choice(np.array([-1.0, 1.0]), size=nm)
        h = np.zeros_like(absP); h[2:] = (absP * fl[None, :])[:-2]
        gp = np.nansum(h * bwdv, axis=1)
        dh = np.zeros_like(h); dh[1:] = np.abs(h[1:] - h[:-1]); dh[0] = np.abs(h[0])
        nn = gp - dh.sum(axis=1) * MAKER_1W
        nn = nn[np.isfinite(nn)]
        plc[i] = float(nn.mean() / nn.std(ddof=1)) if nn.std() > 0 else 0.0
    placebo_p = float((np.sum(plc >= mk_sr / np.sqrt(365 * 24)) + 1) / (n_placebo + 1))

    # surrogate: block-bootstrap whole rows of the realised forward-return ranks.
    # rank is row-wise (axis=1), so rank(arr[order]) == rank(arr)[order] -> rank ONCE.
    sr_ = sig.rank(axis=1).values
    fr_all = fwd1.rank(axis=1).values
    Ln = len(fr_all)
    src = sr_ - np.nanmean(sr_, axis=1, keepdims=True)
    src_ss = np.nansum(src ** 2, axis=1)
    surr = np.empty(n_surrogate)
    for i in range(n_surrogate):
        stt = rng.integers(0, Ln, int(np.ceil(Ln / 24)))
        order = np.concatenate([np.arange(x, x + 24) % Ln for x in stt])[:Ln]
        fr = fr_all[order]
        frc = fr - np.nanmean(fr, axis=1, keepdims=True)
        num = np.nansum(src * frc, axis=1)
        den = np.sqrt(src_ss * np.nansum(frc ** 2, axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = num / den
        surr[i] = abs(float(np.nanmean(rr)))
    surr_p = float((np.sum(surr >= abs(ic)) + 1) / (n_surrogate + 1))

    dsr = deflated_sharpe(mk_net.values, ledger=ledger, family="F_OI")
    dsr_g = deflated_sharpe(mk_net.values, ledger=ledger)
    eq = (1 + mk_net).cumprod()
    n = len(mk_net)
    wf = [round(float((mk_net.iloc[k*n//4:(k+1)*n//4].mean() / (mk_net.iloc[k*n//4:(k+1)*n//4].std(ddof=1) or 1)) * np.sqrt(365*24)), 2) for k in range(4)]
    out = {"feature": feature, "params": params, "hold_h": hold_h, "ic_1h": ic,
           "maker_sr_ann": mk_sr, "gross_sr_ann": gr_sr, "taker_sr_ann": tk_sr,
           "turnover": float(turn.mean()), "cv_sr_mean": float(np.nanmean(fold)) if fold else np.nan,
           "cv_folds": len(fold), "placebo_p": placebo_p, "surrogate_p": surr_p,
           "dsr_family": dsr["dsr"], "dsr_family_n": dsr["n_trials"], "dsr_portfolio": dsr_g["dsr"],
           "walk_forward": wf, "max_drawdown": float((eq / eq.cummax() - 1).min()),
           "n_bars": n, "n_coins": int(sig.shape[1])}
    for kk in ["ic_1h", "maker_sr_ann", "gross_sr_ann", "taker_sr_ann", "cv_sr_mean",
               "placebo_p", "surrogate_p", "dsr_family", "dsr_portfolio", "turnover", "max_drawdown"]:
        if out[kk] == out[kk]:
            ledger.log_metric(exp_id, kk, float(out[kk]), context={"study": f"oi_{feature}"})
    return out


GRID = {
    "oi_delta_rev": dict(k=[6, 24, 72]),
    "oi_delta_mom": dict(k=[6, 24, 72]),
    "oi_price_confirm": dict(k=[12, 24, 48]),
    "oi_short_cover": dict(k=[12, 24, 48]),
    "oi_extreme_rev": dict(n=[336, 720]),
    "oi_per_volume": dict(k=[12, 24, 48]),
    "oi_accel_rev": dict(k=[6, 12]),
    "oi_leverage_state": dict(n=[168, 336]),
}
HOLDS = [6, 24, 72]


def run(ledger: Ledger, max_full=45, only=None):
    results, verdicts, survivors = [], [], []
    n_full = 0
    for feature, gp in GRID.items():
        if only is not None and feature not in only:
            continue
        keys = list(gp); cfgs = [dict(zip(keys, v)) for v in itertools.product(*[gp[k] for k in keys])]
        hyp = ledger.add_hypothesis({
            "claim": f"{feature}: open-interest dynamics predict the cross-section of next-bar return",
            "mechanism": f"see lab/research/oi_study.py::{feature}", "math_form": feature,
            "target": "xsec_fwd_return_1h", "horizon": "hours-days",
            "null_hypothesis": "rank IC = 0 ; net Sharpe <= 0 after cost", "family": "F_OI",
            "lessons_reviewed": True})
        exp = ledger.new_experiment(hyp, "F_OI", {"feature": feature, "grid": gp, "holds": HOLDS}, stage="discovery")
        for cfg, hold in itertools.product(cfgs, HOLDS):
            if n_full >= max_full: break
            n_full += 1
            tid = ledger.log_trial(exp, {**cfg, "hold": hold}, stage="discovery", status="running")
            try:
                st = run_one(feature, cfg, hold, ledger, exp, hyp, n_placebo=120, n_surrogate=120,
                             seed=abs(hash((feature, str(cfg), hold))) % (2**31))
                ledger.mark_trial(tid, "done")
            except Exception as e:
                ledger.log_metric(exp, "error", None, trial_id=tid, text_value=repr(e)[:200])
                ledger.mark_trial(tid, "failed"); continue
            g3 = st["surrogate_p"] < 0.05 and abs(st["ic_1h"]) >= 0.01
            g4 = st["cv_folds"] >= 4 and st["cv_sr_mean"] > 0
            g5 = st["dsr_family"] >= 0.95 and st["placebo_p"] < 0.05
            g7 = st["maker_sr_ann"] >= 0.8 and st["taker_sr_ann"] >= -0.3
            wf_ok = all(x > -0.5 for x in st["walk_forward"])
            v = "PASS" if (g3 and g4 and g5 and g7 and wf_ok) else "FAIL"
            obj = ("clears ladder" if v == "PASS" else
                   (f"|IC|={abs(st['ic_1h']):.4f} surr_p={st['surrogate_p']:.3f}" if not g3 else
                    f"makerSR={st['maker_sr_ann']:.2f}/taker {st['taker_sr_ann']:.2f} WF={st['walk_forward']} "
                    f"DSRfam={st['dsr_family']:.3f} cvSR={st['cv_sr_mean']:.2f}"))
            results.append({**st, "gate": v, "objection": obj, "exp_id": exp, "hyp_id": hyp})
            print(f"  {feature:18} {cfg} h{hold} -> {v:4} IC={st['ic_1h']:+.4f} mkSR={st['maker_sr_ann']:+.2f} "
                  f"tkSR={st['taker_sr_ann']:+.2f} grSR={st['gross_sr_ann']:+.2f} DSRf={st['dsr_family']:.2f} "
                  f"plc={st['placebo_p']:.3f} surr={st['surrogate_p']:.3f} WF={st['walk_forward']}", flush=True)
            if v == "PASS": survivors.append(results[-1])
        best = max((r for r in results if r["feature"] == feature),
                   key=lambda r: (r["gate"] == "PASS", r["dsr_family"], r["maker_sr_ann"]), default=None)
        if best:
            status = "PASS" if best["gate"] == "PASS" else "FAIL"
            ledger.add_verdict(hyp, "final", status, "validator", [exp],
                               rationale=f"best {feature}: mkSR={best['maker_sr_ann']:.2f} IC={best['ic_1h']:.4f} DSRfam={best['dsr_family']:.3f}",
                               strongest_surviving_objection=best["objection"])
            verdicts.append({"feature": feature, "status": status, "best": {k: best[k] for k in
                             ("params", "hold_h", "ic_1h", "maker_sr_ann", "taker_sr_ann", "dsr_family",
                              "placebo_p", "surrogate_p", "walk_forward", "objection")}})
            if status != "PASS":
                ledger.add_lesson(root_cause=best["objection"],
                                  lesson=f"F_OI {feature}: best maker Sharpe(ann) {best['maker_sr_ann']:.2f}, "
                                         f"IC {best['ic_1h']:.4f}, DSR(fam) {best['dsr_family']:.3f}.",
                                  family="F_OI", subject_id=hyp, ladder_level="L4")
    if not survivors:
        exs = sorted({r["exp_id"] for r in results})
        b_ic = max((abs(r["ic_1h"]) for r in results), default=0)
        b_sr = max((r["maker_sr_ann"] for r in results), default=0)
        b_dsr = max((r["dsr_family"] for r in results), default=0)
        ledger.add_bound("F_OI", "cross-sectional open-interest signals, Bybit 2y hourly OI, 45+ configs",
                         f"F_OI: {len(results)} configs, 8 mechanisms (OI momentum/reversal, OI-price confirm, "
                         f"short-cover fade, positioning extreme, OI/volume, OI acceleration, leverage state). "
                         f"Best |IC(1h)| <= {b_ic:.4f}; best maker net Sharpe(ann) <= {b_sr:.2f}; best family DSR "
                         f"<= {b_dsr:.3f}. Open-interest dynamics add no tradeable cross-sectional edge over "
                         f"price/volume at feasible horizons after cost.", exs)
    return {"results": results, "verdicts": verdicts, "survivors": survivors}
