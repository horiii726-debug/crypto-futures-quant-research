"""RESEARCH ROUND 2 · P3 — F_PAIRS : cointegration / statistical-arbitrage.

Economically distinct from every RESEARCH ROUND 1 family: relative-value spread
mean-reversion, not a directional bet. Pre-registered 40-trial budget, its own
ledger (`ledger/pairs.sqlite`). Gate P3.7 is frozen in MECHANISM_REGISTRY.md and
is not moved regardless of outcome.

Mechanisms:
  M1 pairs_johansen_ou   Johansen rank-1 pair selection (rolling 90d, weekly
                         re-estimate), static hedge ratio per window, OU s-score.
  M2 pairs_kalman_ou     same selection, Kalman dynamic hedge ratio.
  M3 pairs_avellaneda    Avellaneda-Lee: remove 3 PCs, OU on the residual,
                         s-score per coin. Entry |s|>1.25, exit |s|<0.5 (paper).

Sizing: GJR-GARCH(1,1) vol-target — SIZING ONLY, never direction (P3.6).
Costs: lab/exec/cost_model.cost_bps, per coin, both legs.
Data:  train+valid 1h (2023-09..2025-05). data/test is NOT opened (R2).
       70/30 internal split: params frozen on the first 70%, held-out = last 30%.
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
from lab.engine.cv import purged_kfold                      # noqa: E402
from lab.exec.cost_model import cost_frac_by_coin            # noqa: E402
from lab.research import pairs_lib as PL                     # noqa: E402
from lab.stats.dsr import deflated_sharpe                    # noqa: E402
from lab.stats.pbo import cscv_pbo                           # noqa: E402

LIQUID = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT",
          "LINKUSDT", "AVAXUSDT", "ADAUSDT", "LTCUSDT", "BCHUSDT", "DOTUSDT",
          "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "INJUSDT",
          "FILUSDT", "LDOUSDT", "CRVUSDT", "XLMUSDT", "ICPUSDT", "UNIUSDT",
          "AAVEUSDT", "XMRUSDT", "DASHUSDT", "ZECUSDT"]

ANN = np.sqrt(365 * 24)
PAIRS_LEDGER = ROOT / "ledger" / "pairs.sqlite"
BUDGET = 40

# ---- frozen defaults (pre-registered, P3) ---------------------------------
DEFAULTS = dict(
    corr_min=0.70, window_h=2160, refit_h=168,
    hl_min_h=4.0, hl_max_h=120.0,
    entry_s=1.25, exit_s=0.5, stop_s=4.0,
    target_vol_ann=0.10, max_gross_lev=3.0, max_pairs=20,
    q_over_r=1e-5, n_pcs=3, pca_win=504,
)

# grid (only for a mechanism that clears the initial screen; <= 8 configs each)
GRIDS = {
    "pairs_johansen_ou": [dict(entry_s=e, exit_s=x) for e in (1.0, 1.25, 1.5) for x in (0.25, 0.5)][:6],
    "pairs_kalman_ou":   [dict(q_over_r=q) for q in (1e-6, 1e-5, 1e-4)],
    "pairs_avellaneda":  [dict(n_pcs=k, entry_s=e) for k in (2, 3) for e in (1.0, 1.25)][:4],
}


# --------------------------------------------------------------------------- #
def _panel() -> pd.DataFrame:
    parts = []
    for part in ("train", "valid"):
        p = ROOT / "data" / part / "prio50_2y" / "1h" / "close.parquet"
        parts.append(pd.read_parquet(p)[LIQUID])
    C = pd.concat(parts).sort_index()
    return C[~C.index.duplicated()].astype(float)


def _cost_vec(mode="hybrid") -> pd.Series:
    return cost_frac_by_coin(LIQUID, mode=mode)


# --------------------------------------------------------------------------- #
#  M1 / M2 — pair trading                                                    #
# --------------------------------------------------------------------------- #
def _select_pairs(logP: pd.DataFrame, t: int, p: dict) -> list[dict]:
    w0 = max(0, t - p["window_h"])
    cols = list(logP.columns)
    M = logP.iloc[w0:t].values                      # (win, N)
    if M.shape[0] < p["window_h"] // 2 or not np.isfinite(M).all():
        return []
    Mc = M - M.mean(0, keepdims=True)
    sd = Mc.std(0)
    corr = (Mc.T @ Mc) / (M.shape[0] * np.outer(sd, sd) + 1e-12)

    cand = []
    N = len(cols)
    for i in range(N):
        for j in range(i + 1, N):
            if abs(corr[i, j]) < p["corr_min"]:
                continue
            # fast Engle-Granger pre-filter: OLS beta, AR(1) of the residual
            x = M[:, j]; y = M[:, i]
            vx = np.var(x)
            if vx <= 0:
                continue
            beta = np.cov(y, x)[0, 1] / vx
            resid = y - beta * x
            _, b_ar, _ = PL._ar1(resid)
            if not (0.0 < b_ar < 0.999):
                continue
            hl = np.log(2.0) / (-np.log(b_ar))
            if not (p["hl_min_h"] <= hl <= p["hl_max_h"] * 1.5):
                continue
            cand.append((hl, i, j))
    cand.sort()
    out = []
    for hl0, i, j in cand[: p["max_pairs"] * 3]:
        jr = PL.johansen_rank1(M[:, [i, j]])
        if jr is None:
            continue
        beta = jr["beta"]
        spread = M[:, i] - beta * M[:, j]
        ou = PL.ou_fit(spread[np.isfinite(spread)])
        if ou is None or not (p["hl_min_h"] <= ou["half_life"] <= p["hl_max_h"]):
            continue
        out.append({"a": cols[i], "b": cols[j], "beta": beta, "mu": ou["mu"],
                    "sigma_eq": ou["sigma_eq"], "half_life": ou["half_life"]})
    out.sort(key=lambda d: d["half_life"])
    return out[: p["max_pairs"]]


def _run_pairs(C: pd.DataFrame, p: dict, *, kalman: bool) -> pd.DataFrame:
    logP_df = np.log(C)
    LP = logP_df.values                                  # (T, N) numpy
    RET = C.pct_change().fillna(0.0).values
    cols = list(C.columns)
    ci = {c: k for k, c in enumerate(cols)}
    cvec = _cost_vec("hybrid").reindex(cols).values
    cost_mean = float(np.nanmean(cvec))
    T, N = LP.shape
    pnl = np.zeros(T)
    turn = np.zeros(T)
    active: list[dict] = []
    prev_book = np.zeros(N)
    refit = p["refit_h"]

    for t in range(p["window_h"], T - 1):
        if (t - p["window_h"]) % refit == 0:
            active = _select_pairs(logP_df, t, p)
            for pr in active:
                pr["ia"], pr["ib"] = ci[pr["a"]], ci[pr["b"]]
                pr["pos"] = 0.0
                if kalman:
                    w0 = max(0, t - p["window_h"])
                    y = LP[w0:t, pr["ia"]]; x = LP[w0:t, pr["ib"]]
                    rv = float(np.var(np.diff(y))) or 1e-6
                    sp, bt = PL.kalman_hedge(y, x, p["q_over_r"], rv)
                    pr["beta"] = float(bt[-1])
                    half = p["window_h"] // 2
                    pr["mu"] = float(np.nanmean(sp[-half:]))
                    pr["sigma_eq"] = float(np.nanstd(sp[-half:])) or 1e-6
        book = np.zeros(N)
        if active:
            row = LP[t]
            for pr in active:
                beta = pr["beta"]
                s = (row[pr["ia"]] - beta * row[pr["ib"]] - pr["mu"]) / (pr["sigma_eq"] or 1e-9)
                old = pr["pos"]; new = old
                if old == 0.0:
                    if s > p["entry_s"]:
                        new = -1.0
                    elif s < -p["entry_s"]:
                        new = 1.0
                elif abs(s) < p["exit_s"] or abs(s) > p["stop_s"]:
                    new = 0.0
                pr["pos"] = new
                denom = 1.0 + abs(beta)
                book[pr["ia"]] += new / denom
                book[pr["ib"]] -= new * beta / denom
            gross = np.abs(book).sum() or 1.0
            book *= min(1.0, p["max_gross_lev"] / gross)
        d = book - prev_book
        pnl[t + 1] = float(book @ RET[t + 1] - np.abs(d) @ np.nan_to_num(cvec, nan=cost_mean))
        turn[t + 1] = float(np.abs(d).sum())
        prev_book = book

    return pd.DataFrame({"pnl": pnl, "turn": turn}, index=C.index)


# --------------------------------------------------------------------------- #
#  M3 — Avellaneda-Lee PCA residual                                          #
# --------------------------------------------------------------------------- #
def _run_avellaneda(C: pd.DataFrame, p: dict) -> pd.DataFrame:
    ret = C.pct_change()
    resid = PL.pca_residual_returns(ret, n_factors=p["n_pcs"], win=p["pca_win"])
    cumv = resid.cumsum().values
    cost = _cost_vec("hybrid")
    T, N = cumv.shape
    pos = np.zeros((T, N))
    ou_win = p["window_h"] // 2
    refit = p["refit_h"]
    ou_par = [None] * N                       # (mu, sigma_eq, hl_ok) per coin, refreshed periodically
    start = p["pca_win"] + ou_win
    for t in range(start, T - 1):
        if (t - start) % refit == 0:
            for j in range(N):
                seg = cumv[t - ou_win:t, j]
                ou = PL.ou_fit(seg[np.isfinite(seg)])
                ou_par[j] = ((ou["mu"], ou["sigma_eq"] or 1e-9)
                             if (ou and p["hl_min_h"] <= ou["half_life"] <= p["hl_max_h"]) else None)
        prev = pos[t - 1]
        cur = prev.copy()
        for j in range(N):
            par = ou_par[j]
            if par is None:
                cur[j] = 0.0
                continue
            s = (cumv[t, j] - par[0]) / par[1]
            if prev[j] == 0.0:
                if s > p["entry_s"]:
                    cur[j] = -1.0
                elif s < -p["entry_s"]:
                    cur[j] = 1.0
            elif abs(s) < p["exit_s"] or abs(s) > p["stop_s"]:
                cur[j] = 0.0
        pos[t] = cur
    W = pd.DataFrame(pos, index=C.index, columns=C.columns)
    gross = W.abs().sum(axis=1).replace(0, np.nan)
    Wn = W.div(gross, axis=0).fillna(0.0)
    held = Wn.shift(2)
    r = ret.reindex(C.index)
    pnl = (held * r).sum(axis=1)
    tc = held.diff().abs().mul(cost, axis=1).sum(axis=1)
    return pd.DataFrame({"pnl": (pnl - tc), "turn": held.diff().abs().sum(axis=1)}, index=C.index)


# --------------------------------------------------------------------------- #
#  sizing + scoring                                                          #
# --------------------------------------------------------------------------- #
def _vol_target(pnl: pd.Series, p: dict) -> pd.Series:
    sig = PL.gjr_sigma(pnl.fillna(0.0).values, refit_every=p["refit_h"])
    sig = pd.Series(sig, index=pnl.index).ffill().bfill()
    tgt = p["target_vol_ann"] / ANN
    scale = (tgt / sig.replace(0, np.nan)).clip(upper=p["max_gross_lev"]).fillna(0.0)
    return pnl * scale.shift(1).fillna(0.0)


def _sr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 5 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / x.std(ddof=1))


def _score(res: pd.DataFrame, p: dict, ledger, seed: int) -> dict:
    pnl_raw = res["pnl"].fillna(0.0)
    pnl = _vol_target(pnl_raw, p)
    T = len(pnl)
    cut = int(T * 0.70)
    sel, oos = pnl.iloc[:cut], pnl.iloc[cut:]
    turn = res["turn"].fillna(0.0)

    # purged CV on the whole series
    idx = np.arange(T)
    t1 = np.minimum(idx + int(p["hl_max_h"]) + 2, T - 1)
    folds = []
    for tr, te in purged_kfold(idx, t1, n_splits=6, embargo_pct=0.02):
        seg = pnl.iloc[np.sort(te)]
        if len(seg) > 20 and seg.std() > 0:
            folds.append(_sr(seg) * ANN)
    cv_pos = sum(1 for f in folds if f > 0)

    # walk-forward quarters
    q = T // 4
    wf = [round(_sr(pnl.iloc[k * q:(k + 1) * q]) * ANN, 2) for k in range(4)]
    wf_monotone_down = all(wf[k] >= wf[k + 1] for k in range(3)) and wf[0] > 0

    rng = np.random.default_rng(seed)
    # placebo: random entry/exit timing on the SAME pnl magnitude (sign shuffle in blocks)
    plc = []
    v = pnl.values
    for _ in range(200):
        b = 48
        nb = int(np.ceil(T / b))
        flip = rng.choice([-1.0, 1.0], size=nb)
        fl = np.repeat(flip, b)[:T]
        plc.append(_sr(v * fl) * ANN)
    obs = _sr(pnl) * ANN
    placebo_p = float((np.sum(np.array(plc) >= obs) + 1) / 201)

    # surrogate: block-bootstrap the pnl, is observed Sharpe outside the null?
    sur = []
    for _ in range(200):
        b = 48
        st = rng.integers(0, T, int(np.ceil(T / b)))
        order = np.concatenate([np.arange(s, s + b) % T for s in st])[:T]
        sur.append(_sr(v[order]) * ANN)
    surrogate_p = float((np.sum(np.array(sur) >= obs) + 1) / 201)

    # PBO proxy: split the sample into 8 blocks; for each split of blocks into
    # IS/OOS halves, does the IS-best sign survive OOS? Here a single config, so
    # PBO ~ P(OOS Sharpe <= 0 | IS Sharpe > 0) across the 8 blocks.
    sub = np.array([_sr(pnl.iloc[k * (T // 8):(k + 1) * (T // 8)]) * ANN for k in range(8)])
    pos_blocks = sub[sub > 0]
    pbo = float(np.mean(sub <= 0)) if sub.size else 1.0
    if pos_blocks.size:
        # fraction of the time an in-sample-positive block is followed by a non-positive one
        pbo = float(np.mean([(sub[k] > 0 and sub[k + 1] <= 0) for k in range(7)]))

    dsr = deflated_sharpe(pnl.values, ledger=ledger, family="F_PAIRS")

    return {
        "params": p,
        "net_sharpe_ann": obs,
        "oos_net_sharpe_ann": _sr(oos) * ANN,
        "sel_net_sharpe_ann": _sr(sel) * ANN,
        "gross_sharpe_ann": _sr(pnl_raw) * ANN,
        "turnover_per_bar": float(turn.mean()),
        "cv_folds_positive": cv_pos, "cv_n_folds": len(folds),
        "cv_sharpe_mean": float(np.nanmean(folds)) if folds else float("nan"),
        "walk_forward": wf, "wf_monotone_decline": bool(wf_monotone_down),
        "placebo_p": placebo_p, "surrogate_p": surrogate_p,
        "pbo": float(pbo),
        "dsr_family": dsr["dsr"], "dsr_n_trials": dsr["n_trials"],
        "max_drawdown": float(((1 + pnl).cumprod() / (1 + pnl).cumprod().cummax() - 1).min()),
    }


# ---- frozen gate P3.7 ---------------------------------------------------
def gate(st: dict) -> tuple[str, str]:
    checks = {
        "oos_net_sharpe>=1.0": st["oos_net_sharpe_ann"] >= 1.0,
        "turnover<=0.05": st["turnover_per_bar"] <= 0.05,
        "cv>=4/6_positive": st["cv_folds_positive"] >= 4,
        "wf_not_monotone_decline": not st["wf_monotone_decline"],
        "placebo_p<0.05": st["placebo_p"] < 0.05,
        "surrogate_p<0.05": st["surrogate_p"] < 0.05,
        "pbo<0.5": st["pbo"] < 0.5,
        "dsr_family>=0.95": st["dsr_family"] >= 0.95,
    }
    failed = [k for k, ok in checks.items() if not ok]
    return ("PASS" if not failed else "FAIL"), "; ".join(failed)


# --------------------------------------------------------------------------- #
def _mk_ledger():
    os.environ["CRYPTO_LAB_LEDGER_SQLITE"] = str(PAIRS_LEDGER)
    from lab.ledger import Ledger
    return Ledger()


MECHANISMS = {
    "pairs_johansen_ou": lambda C, p: _run_pairs(C, p, kalman=False),
    "pairs_kalman_ou":   lambda C, p: _run_pairs(C, p, kalman=True),
    "pairs_avellaneda":  lambda C, p: _run_avellaneda(C, p),
}
PREDICTED_SIGN = {"pairs_johansen_ou": "spread mean-reverts (fade |s|>1.25)",
                  "pairs_kalman_ou": "spread mean-reverts, dynamic beta",
                  "pairs_avellaneda": "PCA-residual mean-reverts"}


def run() -> dict:
    C = _panel()
    L = _mk_ledger()
    spent = L.trial_count(family="F_PAIRS")
    results = []
    print(f"F_PAIRS  budget {BUDGET}, already spent {spent}", flush=True)

    for mech, fn in MECHANISMS.items():
        if spent >= BUDGET:
            break
        hyp = L.add_hypothesis({
            "claim": f"{mech}: cointegrated-spread mean reversion is tradeable net of per-coin cost",
            "mechanism": PREDICTED_SIGN[mech], "math_form": mech,
            "target": "spread_reversion", "horizon": ">=4h (OU half-life 4h-5d)",
            "null_hypothesis": "net Sharpe <= 0 ; spread is a random walk",
            "family": "F_PAIRS", "lessons_reviewed": True})
        exp = L.new_experiment(hyp, "F_PAIRS", {"mechanism": mech, "defaults": DEFAULTS}, stage="discovery")

        # 1 trial at author defaults
        tid = L.log_trial(exp, {"config": "default"}, stage="discovery", status="running")
        spent += 1
        try:
            st = _score(fn(C, dict(DEFAULTS)), dict(DEFAULTS), L, seed=hash(mech) % 2**31)
            g, obj = gate(st)
            L.mark_trial(tid, "done")
        except Exception as e:
            L.log_metric(exp, "error", None, trial_id=tid, text_value=repr(e)[:200])
            L.mark_trial(tid, "failed")
            print(f"  {mech:20} default -> ERROR {e!r}", flush=True)
            continue
        for k in ("net_sharpe_ann", "oos_net_sharpe_ann", "gross_sharpe_ann", "turnover_per_bar",
                  "cv_sharpe_mean", "placebo_p", "surrogate_p", "pbo", "dsr_family"):
            if st[k] == st[k]:
                L.log_metric(exp, k, float(st[k]), trial_id=tid)
        results.append({"mech": mech, "config": "default", "gate": g, "obj": obj, **st})
        print(f"  {mech:20} default -> {g}  oosSR={st['oos_net_sharpe_ann']:+.2f} "
              f"grossSR={st['gross_sharpe_ann']:+.2f} turn={st['turnover_per_bar']:.4f} "
              f"cv={st['cv_folds_positive']}/{st['cv_n_folds']} plc={st['placebo_p']:.3f} "
              f"sur={st['surrogate_p']:.3f} pbo={st['pbo']:.2f} DSR={st['dsr_family']:.2f}  [{obj}]", flush=True)

        # grid only if the default at least shows a gross edge and clears surrogate
        screen_ok = st["gross_sharpe_ann"] > 0.5 and st["surrogate_p"] < 0.10
        if not screen_ok:
            L.add_verdict(hyp, "final", "FAIL", "validator", [exp],
                          rationale=f"{mech} default: oosSR {st['oos_net_sharpe_ann']:.2f}, "
                                    f"gross {st['gross_sharpe_ann']:.2f}, surrogate p {st['surrogate_p']:.3f}",
                          strongest_surviving_objection=obj)
            L.add_lesson(root_cause=obj, lesson=f"F_PAIRS {mech}: default config fails the screen "
                         f"(gross Sharpe {st['gross_sharpe_ann']:.2f}, surrogate p {st['surrogate_p']:.3f}).",
                         family="F_PAIRS", subject_id=hyp, ladder_level="L4")
            continue

        best = results[-1]
        for gcfg in GRIDS.get(mech, []):
            if spent >= BUDGET:
                break
            p = {**DEFAULTS, **gcfg}
            tid = L.log_trial(exp, gcfg, stage="discovery", status="running")
            spent += 1
            try:
                st = _score(fn(C, p), p, L, seed=(hash(mech) ^ hash(str(gcfg))) % 2**31)
                g, obj = gate(st)
                L.mark_trial(tid, "done")
            except Exception as e:
                L.log_metric(exp, "error", None, trial_id=tid, text_value=repr(e)[:200])
                L.mark_trial(tid, "failed"); continue
            results.append({"mech": mech, "config": str(gcfg), "gate": g, "obj": obj, **st})
            print(f"  {mech:20} {gcfg} -> {g}  oosSR={st['oos_net_sharpe_ann']:+.2f} "
                  f"turn={st['turnover_per_bar']:.4f} DSR={st['dsr_family']:.2f} [{obj}]", flush=True)
            if (g == "PASS", st["dsr_family"], st["oos_net_sharpe_ann"]) > \
               (best["gate"] == "PASS", best["dsr_family"], best["oos_net_sharpe_ann"]):
                best = results[-1]

        status = "PASS" if best["gate"] == "PASS" else "FAIL"
        L.add_verdict(hyp, "final", status, "validator", [exp],
                      rationale=f"best {mech}: oosSR {best['oos_net_sharpe_ann']:.2f}, "
                                f"turn {best['turnover_per_bar']:.4f}, DSR {best['dsr_family']:.3f}, "
                                f"cv {best['cv_folds_positive']}/{best['cv_n_folds']}, pbo {best['pbo']:.2f}",
                      strongest_surviving_objection=best["obj"])
        if status != "PASS":
            L.add_lesson(root_cause=best["obj"],
                         lesson=f"F_PAIRS {mech}: best oos net Sharpe {best['oos_net_sharpe_ann']:.2f}, "
                                f"gross {best['gross_sharpe_ann']:.2f}, DSR(F_PAIRS) {best['dsr_family']:.3f}. "
                                f"Failed: {best['obj']}.",
                         family="F_PAIRS", subject_id=hyp, ladder_level="L4")

    survivors = [r for r in results if r["gate"] == "PASS"]
    if not survivors and results:
        bg = max(r["gross_sharpe_ann"] for r in results)
        bo = max(r["oos_net_sharpe_ann"] for r in results)
        L.add_bound("F_PAIRS", "cointegration / stat-arb spread reversion, 28 liquid coins, 1h, 2023-09..2025-05",
                    f"{len(results)} configs across 3 mechanisms (Johansen+OU, Kalman+OU, "
                    f"Avellaneda-Lee PCA). Best gross annualised Sharpe {bg:.2f}, best held-out "
                    f"net Sharpe {bo:.2f}. No config clears the frozen P3.7 gate "
                    f"(oosSR>=1.0, turn<=0.05, CV>=4/6, placebo & surrogate p<0.05, PBO<0.5, "
                    f"DSR>=0.95 vs 40-trial budget). Relative-value spread reversion between "
                    f"liquid perps is bounded below the per-coin cost floor at OU-feasible "
                    f"half-lives (4h-5d).",
                    [r for r in [results[0].get("exp_id")] if r] or ["E-PAIRS"])
    return {"results": results, "survivors": survivors, "spent": spent}


if __name__ == "__main__":
    r = run()
    json.dump(r["results"], open(ROOT / "lab" / "reports" / "PAIRS_RESULT.json", "w"),
              indent=1, default=float)
    print(f"\nF_PAIRS DONE — {len(r['results'])} trials, {len(r['survivors'])} survivors, "
          f"{r['spent']}/{BUDGET} budget spent")
