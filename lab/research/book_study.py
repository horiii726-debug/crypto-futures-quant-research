"""F_BOOK - intraday DOM (bookDepth) signals, 2023-09..2024-05, 12 coins, 5m/15m.

Features (from lab/data/bookdepth.py bars): depth_imb_1pct/5pct, book_slope,
total_depth, microprice_tilt, imb_vol, depth_withdraw.

Signals:
  book_imb_follow   depth imbalance -> price follows (more bids => up)
  book_imb_fade     extreme imbalance -> exhausts, reverses
  microprice_drift  microprice tilt -> next-bar drift
  withdraw_fade     liquidity pulled + last move -> that move overshot, fade
  slope_fragile     steep book (fragile) -> amplify: follow the last move

Cross-sectional (rank 12 coins). Full ladder: purged CV, placebo (random
ranking), surrogate (block-shuffle returns), family DSR, maker + taker cost.
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
BOOK = ROOT / "data" / "processed" / "bookdepth_bars"
_EXEC = json.loads((ROOT / "data" / "processed" / "exec_summary.json").read_text())
SYMS = ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","BNBUSDT","LINKUSDT","AVAXUSDT","ADAUSDT","LTCUSDT","BCHUSDT","INJUSDT"]
_V1_MK = _EXEC["hybrid_roundtrip_bps_mean"] / 2 * 1e-4
_V1_TK = 0.0005 + 0.85e-4
try:                                                            # RESEARCH ROUND 2 · P0.2
    from lab.exec.cost_model import cost_frac_by_coin as _cfbc
    import numpy as _np
    MK = float(_np.nanmean(_cfbc(SYMS, mode="hybrid").values))
    TK = float(_np.nanmean(_cfbc(SYMS, mode="taker").values))
except Exception:                                               # pragma: no cover
    MK, TK = _V1_MK, _V1_TK


def _panels(bar_min):
    book, close = {}, {}
    for s in SYMS:
        bf = BOOK / f"{s}_{bar_min}m.parquet"
        if not bf.exists():
            continue
        b = pd.read_parquet(bf)
        book[s] = b
    idx = None
    for s in book:
        idx = book[s].index if idx is None else idx.union(book[s].index)
    # price: 1m klines resampled to bar
    from lab.data.ingest import _load_1m
    months = [str(p) for p in pd.period_range("2023-09", "2024-05", freq="M")]
    for s in list(book):
        d = _load_1m(s, months)
        if d.empty:
            book.pop(s); continue
        px = d.set_index("dt")["close"].resample(f"{bar_min}min", label="left", closed="left").last()
        close[s] = px
    common = [s for s in book if s in close]
    C = pd.DataFrame({s: close[s] for s in common}).sort_index()
    idx = C.index
    B = {k: pd.DataFrame({s: book[s][k].reindex(idx) for s in common}) for k in
         ["depth_imb_1pct", "depth_imb_5pct", "book_slope", "total_depth", "microprice_tilt", "imb_vol", "depth_withdraw"]}
    return C, B


def _zx(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


FEATURES = {
    "book_imb_follow": lambda C, B, k=3: _zx(B["depth_imb_1pct"].rolling(k, min_periods=1).mean()),
    "book_imb_fade": lambda C, B, k=1: -_zx(B["depth_imb_1pct"].rolling(k, min_periods=1).mean()),
    "book_imb5_follow": lambda C, B, k=3: _zx(B["depth_imb_5pct"].rolling(k, min_periods=1).mean()),
    "microprice_drift": lambda C, B, k=1: _zx(B["microprice_tilt"].rolling(k, min_periods=1).mean()),
    "withdraw_fade": lambda C, B, k=3: -_zx((B["depth_withdraw"].rolling(k, min_periods=1).mean())
                                            * np.sign(np.log(C).diff(k))),
    "slope_amplify": lambda C, B, k=3: _zx((-B["book_slope"].rolling(k, min_periods=2).mean())
                                           * np.sign(np.log(C).diff(k))),
    "imbvol_fade": lambda C, B, k=3: -_zx(B["imb_vol"].rolling(k, min_periods=1).mean()
                                          * np.sign(np.log(C).diff(k))),
}
GRID = {f: dict(k=[1, 3, 6]) for f in FEATURES}
HOLDS = [1, 3, 6]


def _pos(sig, q=0.25):
    r = sig.rank(axis=1, pct=True)
    lo = (r >= 1 - q).astype(float); sh = (r <= q).astype(float)
    nl = lo.sum(axis=1).replace(0, np.nan); ns = sh.sum(axis=1).replace(0, np.nan)
    return (lo.div(nl, axis=0) - sh.div(ns, axis=0)).fillna(0.0) / 2.0


_PANEL_CACHE = {}
def run_one(bar_min, feature, k, hold, ledger, exp_id, hyp_id, n_placebo=120, n_surrogate=120, seed=0):
    if bar_min not in _PANEL_CACHE:
        _PANEL_CACHE[bar_min] = _panels(bar_min)
    C, B = _PANEL_CACHE[bar_min]
    rng = np.random.default_rng(seed)
    sig = FEATURES[feature](C, B, k=k)
    sig = sig.where(C.notna().sum(axis=1) >= 6)
    pos = _pos(sig)
    if hold > 1:
        pos = pos.rolling(hold, min_periods=1).mean()
    bwd = C / C.shift(1) - 1.0
    held = pos.shift(2)
    gross = (held * bwd).sum(axis=1)
    turn = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    ann = np.sqrt(365 * 24 * 60 / bar_min)

    def sc(c):
        net = (gross - turn * c).dropna()
        s = lambda x: float(x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 and len(x) > 5 else 0.0
        return s(net) * ann, net
    mk_sr, mk_net = sc(MK); tk_sr, _ = sc(TK)
    gr_sr = (lambda x: float(x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else 0.0)(gross.dropna()) * ann

    fwd1 = C.shift(-1) / C - 1.0
    ic = float(sig.rank(axis=1).corrwith(fwd1.rank(axis=1), axis=1).mean(skipna=True))

    T = len(gross.dropna()); idx = gross.dropna().index
    t0 = np.arange(T); t1 = np.minimum(t0 + hold + 2, T - 1)
    nf = (gross - turn * MK)
    fold = []
    for tr, te in purged_kfold(t0, t1, n_splits=6, embargo_pct=0.02):
        if len(te) < 50: continue
        seg = nf.reindex(idx[np.sort(te)]).dropna()
        if len(seg) > 5 and seg.std() > 0: fold.append(float(seg.mean() / seg.std(ddof=1)))

    plc = []
    names = sig.notna()
    for _ in range(n_placebo):
        n = pd.DataFrame(rng.normal(size=sig.shape), index=sig.index, columns=sig.columns).where(names)
        p = _pos(n)
        if hold > 1: p = p.rolling(hold, min_periods=1).mean()
        h = p.shift(2)
        gp = (h * bwd).sum(axis=1); tp = h.diff().abs().sum(axis=1).fillna(h.abs().sum(axis=1))
        nn = (gp - tp * MK).dropna()
        plc.append(float(nn.mean() / nn.std(ddof=1)) if nn.std() > 0 else 0.0)
    placebo_p = float((np.sum(np.array(plc) >= mk_sr / ann) + 1) / (n_placebo + 1))

    # surrogate: block-shuffle whole rows of the forward-return rank matrix
    # (row-wise rank is permutation-invariant across rows) -> rank ONCE.
    frr = fwd1.rank(axis=1).values
    srr = sig.rank(axis=1).values
    src = srr - np.nanmean(srr, axis=1, keepdims=True)
    ss = np.nansum(src ** 2, axis=1)
    La = len(frr); surr = np.empty(n_surrogate)
    for i in range(n_surrogate):
        st = rng.integers(0, La, int(np.ceil(La / 20)))
        order = np.concatenate([np.arange(s, s + 20) % La for s in st])[:La]
        fr = frr[order]; frc = fr - np.nanmean(fr, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = np.nansum(src * frc, axis=1) / np.sqrt(ss * np.nansum(frc ** 2, axis=1))
        surr[i] = abs(float(np.nanmean(rr)))
    surr_p = float((np.sum(surr >= abs(ic)) + 1) / (n_surrogate + 1))

    dsr = deflated_sharpe(mk_net.values, ledger=ledger, family="F_BOOK")
    dsr_g = deflated_sharpe(mk_net.values, ledger=ledger)
    n = len(mk_net)
    wf = [round(float((mk_net.iloc[j*n//4:(j+1)*n//4].mean() / (mk_net.iloc[j*n//4:(j+1)*n//4].std(ddof=1) or 1)) * ann), 2) for j in range(4)]
    out = {"bar_min": bar_min, "feature": feature, "k": k, "hold": hold, "ic_1bar": ic,
           "maker_sr_ann": mk_sr, "gross_sr_ann": gr_sr, "taker_sr_ann": tk_sr,
           "turnover": float(turn.mean()), "cv_sr_mean": float(np.nanmean(fold)) if fold else np.nan,
           "cv_folds": len(fold), "placebo_p": placebo_p, "surrogate_p": surr_p,
           "dsr_family": dsr["dsr"], "dsr_family_n": dsr["n_trials"], "dsr_portfolio": dsr_g["dsr"],
           "walk_forward": wf, "max_drawdown": float(((1 + mk_net).cumprod() / (1 + mk_net).cumprod().cummax() - 1).min()),
           "n_bars": n}
    for kk in ["ic_1bar", "maker_sr_ann", "gross_sr_ann", "taker_sr_ann", "cv_sr_mean",
               "placebo_p", "surrogate_p", "dsr_family", "dsr_portfolio", "turnover", "max_drawdown"]:
        if out[kk] == out[kk]:
            ledger.log_metric(exp_id, kk, float(out[kk]), context={"study": f"book_{feature}"})
    return out


def run(ledger: Ledger, max_full=126):
    results, verdicts, survivors = [], [], []
    n_full = 0
    for feature in FEATURES:
        hyp = ledger.add_hypothesis({
            "claim": f"{feature}: DOM depth structure predicts the cross-section of next-bar return",
            "mechanism": f"see lab/research/book_study.py::{feature}", "math_form": feature,
            "target": "xsec_fwd_return", "horizon": "minutes", "null_hypothesis": "rank IC = 0 ; net Sharpe <= 0",
            "family": "F_BOOK", "lessons_reviewed": True})
        exp = ledger.new_experiment(hyp, "F_BOOK", {"feature": feature, "ks": [1, 3, 6], "holds": HOLDS,
                                                    "window": "2023-09..2024-05", "bars": [5, 15]}, stage="discovery")
        for bar_min, k, hold in itertools.product([15, 5], [1, 3, 6], HOLDS):
            if n_full >= max_full: break
            n_full += 1
            tid = ledger.log_trial(exp, {"bar": bar_min, "k": k, "hold": hold}, stage="discovery", status="running")
            try:
                st = run_one(bar_min, feature, k, hold, ledger, exp, hyp, n_placebo=60, n_surrogate=80,
                             seed=abs(hash((feature, bar_min, k, hold))) % (2**31))
                ledger.mark_trial(tid, "done")
            except Exception as e:
                ledger.log_metric(exp, "error", None, trial_id=tid, text_value=repr(e)[:200])
                ledger.mark_trial(tid, "failed"); continue
            g3 = st["surrogate_p"] < 0.05 and abs(st["ic_1bar"]) >= 0.01
            g4 = st["cv_folds"] >= 4 and st["cv_sr_mean"] > 0
            g5 = st["dsr_family"] >= 0.95 and st["placebo_p"] < 0.05
            g7 = st["maker_sr_ann"] >= 0.8 and st["taker_sr_ann"] >= -0.3
            wf_ok = all(x > -0.5 for x in st["walk_forward"])
            v = "PASS" if (g3 and g4 and g5 and g7 and wf_ok) else "FAIL"
            obj = ("clears ladder" if v == "PASS" else
                   (f"|IC|={abs(st['ic_1bar']):.4f} surr_p={st['surrogate_p']:.3f}" if not g3 else
                    f"mkSR={st['maker_sr_ann']:.2f}/tk {st['taker_sr_ann']:.2f} WF={st['walk_forward']} "
                    f"DSRf={st['dsr_family']:.3f} cvSR={st['cv_sr_mean']:.2f}"))
            results.append({**st, "gate": v, "objection": obj, "exp_id": exp, "hyp_id": hyp})
            print(f"  {feature:18} {bar_min}m k{k} h{hold} -> {v:4} IC={st['ic_1bar']:+.4f} "
                  f"mkSR={st['maker_sr_ann']:+.2f} grSR={st['gross_sr_ann']:+.2f} DSRf={st['dsr_family']:.2f} "
                  f"plc={st['placebo_p']:.3f} surr={st['surrogate_p']:.3f} WF={st['walk_forward']}", flush=True)
            if v == "PASS": survivors.append(results[-1])
        best = max((r for r in results if r["feature"] == feature),
                   key=lambda r: (r["gate"] == "PASS", r["dsr_family"], r["maker_sr_ann"]), default=None)
        if best:
            status = "PASS" if best["gate"] == "PASS" else "FAIL"
            ledger.add_verdict(hyp, "final", status, "validator", [exp],
                               rationale=f"best {feature}: mkSR={best['maker_sr_ann']:.2f} IC={best['ic_1bar']:.4f} DSRf={best['dsr_family']:.3f}",
                               strongest_surviving_objection=best["objection"])
            verdicts.append({"feature": feature, "status": status, "best": {kk: best[kk] for kk in
                             ("bar_min", "k", "hold", "ic_1bar", "maker_sr_ann", "taker_sr_ann", "dsr_family",
                              "placebo_p", "surrogate_p", "walk_forward", "objection")}})
            if status != "PASS":
                ledger.add_lesson(root_cause=best["objection"],
                                  lesson=f"F_BOOK {feature}: best maker Sharpe(ann) {best['maker_sr_ann']:.2f}, IC {best['ic_1bar']:.4f}, DSR(fam) {best['dsr_family']:.3f}.",
                                  family="F_BOOK", subject_id=hyp, ladder_level="L4")
    if not survivors:
        exs = sorted({r["exp_id"] for r in results})
        b_ic = max((abs(r["ic_1bar"]) for r in results), default=0)
        b_sr = max((r["maker_sr_ann"] for r in results), default=0)
        b_gr = max((r["gross_sr_ann"] for r in results), default=0)
        b_dsr = max((r["dsr_family"] for r in results), default=0)
        ledger.add_bound("F_BOOK", "cross-sectional DOM depth signals, bookDepth 2023-09..2024-05, 12 coins, 5m/15m",
                         f"F_BOOK: {len(results)} configs, 7 mechanisms (depth imbalance follow/fade at +-1% & +-5%, "
                         f"microprice drift, liquidity-withdrawal fade, book-slope amplify, imbalance-vol fade). "
                         f"Best |IC(1-bar)| <= {b_ic:.4f}; best GROSS Sharpe(ann) <= {b_gr:.2f}; best maker net "
                         f"Sharpe(ann) <= {b_sr:.2f}; best family DSR <= {b_dsr:.3f}. bookDepth (+-1..+-5% bands, "
                         f"30s snapshots) carries no tradeable cross-sectional directional edge at 5-15min after "
                         f"cost. Real book-microstructure edge is at the touch / sub-second, needs full L2.", exs)
    return {"results": results, "verdicts": verdicts, "survivors": survivors}
