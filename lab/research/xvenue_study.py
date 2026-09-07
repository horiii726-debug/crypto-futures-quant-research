"""Cross-venue funding-spread arbitrage (F_XVENUE).

For each coin, spread = funding_binance - funding_bybit (per 8h). When the
spread is wide, put on a market-neutral, delta-neutral pair:
  spread > 0  -> SHORT Binance perp + LONG Bybit perp  (collect +spread/8h)
  spread < 0  -> the reverse
Both perps track the same index, so the price legs ~cancel; the residual is the
inter-venue basis drift (small). Held with hysteresis to cut the 4-leg cost.

This is an ARBITRAGE (a genuine inefficiency), so unlike the single-venue carry
it SHOULD beat the surrogate (shuffled spread) if it is real.
"""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from lab.engine.cv import purged_kfold
from lab.ledger import Ledger
from lab.stats.dsr import deflated_sharpe

ROOT = Path(__file__).resolve().parents[2]
BN = ROOT / "data" / "processed" / "basis"
BY = ROOT / "data" / "processed" / "bybit_funding"
# 4-leg round trip: (short Binance + long Bybit) open, then close both.
# Binance taker 5bp + Bybit taker 5.5bp + ~3bp spread each leg, x2 legs x2 (open/close)
RT_TAKER = 2 * 2 * (0.000525 + 0.0003)   # ~33 bps 4-leg
RT_MAKER = 2 * 2 * (0.00015 + 0.00010)    # ~10 bps 4-leg: maker fee ~1.5bp + adverse-sel ~1bp per leg
RT = RT_TAKER
FUND_YR = 3 * 365


@njit(cache=False)
def _hyst(s, enter, exitl):
    n, m = s.shape
    out = np.zeros((n, m))
    for j in range(m):
        cur = 0.0
        for i in range(n):
            x = s[i, j]
            if x != x:
                out[i, j] = cur; continue
            if cur == 0.0:
                if x >= enter: cur = -1.0     # spread>0 -> short Binance (w=-1 on Binance leg)
                elif x <= -enter: cur = 1.0
            elif cur == -1.0:
                if x < exitl: cur = 0.0
            else:
                if x > -exitl: cur = 0.0
            out[i, j] = cur
    return out


def _load():
    bn, by = {}, {}
    for f in sorted(BN.glob("*.parquet")):
        d = pd.read_parquet(f)
        if "funding" in d and d["funding"].notna().sum() > 4000:
            bn[f.stem] = d["funding"].clip(-0.02, 0.02)
    for f in sorted(BY.glob("*.parquet")):
        d = pd.read_parquet(f).set_index("dt")["funding_bybit"]
        by[f.stem] = d.clip(-0.02, 0.02)
    B = pd.DataFrame(bn).sort_index().resample("8h").last()
    Y = pd.DataFrame(by).sort_index().resample("8h").last()
    common = [c for c in B.columns if c in Y.columns]
    B, Y = B[common], Y[common].reindex(B.index)
    return B, Y


def run_one(*, enter_bp, exit_bp, ma, rebalance, ledger, exp_id, hyp_id,
            execmode='taker', top_coins=0, n_placebo=150, n_surrogate=150, seed=0):
    rt = RT_MAKER if execmode == 'maker' else RT_TAKER
    B, Y = _load()
    rng = np.random.default_rng(seed)
    spread = (B - Y)
    # NOTE: point-in-time coin universe is applied inside the position loop
    # (see _pit_topk); do NOT filter columns here (that would be look-ahead).
    sig = spread.rolling(ma, min_periods=max(2, ma // 2)).mean() if ma > 1 else spread
    pos = pd.DataFrame(_hyst(np.ascontiguousarray(sig.values, np.float64),
                             enter_bp * 1e-4, exit_bp * 1e-4),
                       index=sig.index, columns=sig.columns)
    if rebalance > 1:
        keep = np.zeros(len(pos), bool); keep[::rebalance] = True
        pos = pos.where(pd.Series(keep, index=pos.index), np.nan).ffill().fillna(0.0)
    # point-in-time top-k: at each bar keep only the k coins with the largest
    # TRAILING mean |spread| as of that bar (no future info).
    if top_coins and top_coins < spread.shape[1]:
        trail = spread.abs().rolling(120, min_periods=40).mean().shift(1)
        rank = trail.rank(axis=1, ascending=False)
        pos = pos.where(rank <= top_coins, 0.0)
    held = pos.shift(1)
    non = held.abs().sum(axis=1).replace(0, np.nan)
    w = held.div(non, axis=0).fillna(0.0)
    wv = w.values
    sv = spread.reindex(w.index).values
    # pnl per 8h: w = -1 means short Binance / long Bybit -> collect (funding_bn - funding_by) = +spread
    #   held * (-spread)?  held=-1, want +spread => pnl = -held*spread
    pnl_spread = np.nansum(-wv * sv, axis=1)
    turn = np.abs(np.diff(wv, axis=0, prepend=0.0)).sum(axis=1)
    cost = turn * rt
    # inter-venue basis divergence: the two perps do NOT track perfectly; charge
    # a per-active-leg noise of ~0.4 bps/8h std (conservative; we lack Bybit index).
    rng_bd = np.random.default_rng(seed + 999)
    basis_div = np.nansum(np.abs(wv), axis=1) * rng_bd.normal(0, 0.4e-4, size=wv.shape[0])
    cost = cost + np.abs(basis_div)
    net = pd.Series(pnl_spread - cost, index=w.index).dropna()
    min_coins = float(held.abs().sum(axis=1).replace(0, np.nan).mean() or 0)
    g = pd.Series(pnl_spread, index=w.index).reindex_like(net)
    sr = lambda x: float(x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 and len(x) > 5 else 0.0
    ann = np.sqrt(FUND_YR)

    T = len(net)
    wf = [round(sr(net.iloc[k * T // 4:(k + 1) * T // 4]) * ann, 2) for k in range(4)]
    t0 = np.arange(T); t1 = np.minimum(t0 + max(rebalance, 1), T - 1)
    fold = []
    for tr, te in purged_kfold(t0, t1, n_splits=6, embargo_pct=0.02):
        if len(te) < 20: continue
        s = net.iloc[np.sort(te)]
        if s.std() > 0: fold.append(sr(s))

    # placebo: random per-coin sign
    absw = np.abs(wv)
    plc = np.empty(n_placebo)
    for b in range(n_placebo):
        fl = rng.choice(np.array([-1.0, 1.0]), size=wv.shape[1])
        nn = absw * fl[None, :]
        gp = np.nansum(-nn * sv, axis=1)
        tp = np.abs(np.diff(nn, axis=0, prepend=0.0)).sum(axis=1) * rt
        pn = (gp - tp); pn = pn[np.isfinite(pn)]
        plc[b] = sr(pd.Series(pn))
    placebo_p = float((np.sum(plc >= sr(net)) + 1) / (n_placebo + 1))

    # surrogate: block-shuffle the spread matrix rows (kills spread persistence)
    surr = np.empty(n_surrogate)
    cost_v = cost
    for b in range(n_surrogate):
        stt = rng.integers(0, T, int(np.ceil(T / 8)))
        order = np.concatenate([np.arange(s, s + 8) % T for s in stt])[:T]
        gs = np.nansum(-wv * sv[order], axis=1) - cost_v
        gs = gs[np.isfinite(gs)]
        surr[b] = sr(pd.Series(gs))
    surr_p = float((np.sum(surr >= sr(net)) + 1) / (n_surrogate + 1))

    dsr = deflated_sharpe(net.values, ledger=ledger, family="F_XVENUE")
    dsr_g = deflated_sharpe(net.values, ledger=ledger)
    eq = (1 + net).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    out = {"enter_bp": enter_bp, "exit_bp": exit_bp, "ma": ma, "rebalance": rebalance,
           "sr_net_ann": sr(net) * ann, "sr_gross_ann": sr(g) * ann,
           "ann_return_net": float(net.mean() * FUND_YR),
           "spread_harvest_ann": float(np.nanmean(pnl_spread) * FUND_YR),
           "cost_ann": float(np.nanmean(cost) * FUND_YR),
           "execmode": execmode, "rt_bps": rt*1e4, "turnover_mean": float(turn.mean()), "avg_coins_on": float(held.abs().sum(axis=1).mean()),
           "max_drawdown": mdd, "walk_forward_sr": wf,
           "cv_sr_mean": float(np.nanmean(fold)) if fold else np.nan, "cv_folds": len(fold),
           "placebo_p": placebo_p, "surrogate_p": surr_p,
           "dsr_family": dsr["dsr"], "dsr_family_n": dsr["n_trials"],
           "dsr_portfolio": dsr_g["dsr"], "n_bars": T, "n_coins": int(w.shape[1]), "min_coins_flag": min_coins}
    for k in ["sr_net_ann", "sr_gross_ann", "spread_harvest_ann", "cost_ann", "cv_sr_mean",
              "placebo_p", "surrogate_p", "dsr_family", "max_drawdown", "turnover_mean"]:
        if out[k] == out[k]:
            ledger.log_metric(exp_id, k, float(out[k]), context={"study": "xvenue"})
    return out


GRID = dict(enter_bp=[3.0, 6.0, 10.0], ma=[3, 9], rebalance=[3, 9], execmode=['maker','taker'], top_coins=[0, 20])


def run(ledger: Ledger, max_full=40):
    cfgs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    hyp = ledger.add_hypothesis({
        "claim": "the Binance-minus-Bybit perp funding spread is a mean-reverting inefficiency "
                 "harvestable market-neutral & delta-neutral after 4-leg cost",
        "mechanism": "different venue order flow -> different funding; a cross-venue pair collects the "
                     "differential while the price legs cancel",
        "math_form": "pnl = -w*(f_binance - f_bybit) - turnover*4leg_cost",
        "target": "xvenue_net_return", "horizon": "days-weeks",
        "null_hypothesis": "net Sharpe <= 0.5, or indistinguishable from shuffled spread (surrogate)",
        "family": "F_XVENUE", "lessons_reviewed": True})
    exp = ledger.new_experiment(hyp, "F_XVENUE", {"grid": GRID, "rt_bps": RT * 1e4}, stage="discovery")
    results = []
    for i, cfg in enumerate(cfgs):
        if i >= max_full: break
        tid = ledger.log_trial(exp, cfg, stage="discovery", status="running")
        try:
            st = run_one(exit_bp=1.0, ledger=ledger, exp_id=exp, hyp_id=hyp,
                         n_placebo=120, n_surrogate=120, seed=abs(hash(str(cfg))) % (2**31), **cfg)
            ledger.mark_trial(tid, "done")
        except Exception as e:
            ledger.log_metric(exp, "error", None, trial_id=tid, text_value=repr(e)[:200])
            ledger.mark_trial(tid, "failed"); continue
        # taker cost-stress: rerun with taker execution, must not collapse
        st_tk = run_one(enter_bp=cfg["enter_bp"], exit_bp=1.5, ma=cfg["ma"],
                        rebalance=cfg["rebalance"], execmode="taker",
                        top_coins=cfg.get("top_coins", 0), ledger=ledger, exp_id=exp,
                        hyp_id=hyp, n_placebo=1, n_surrogate=1,
                        seed=abs(hash(str(cfg))) % (2**31))
        g3 = st["surrogate_p"] < 0.05
        g4 = st["cv_folds"] >= 4 and st["cv_sr_mean"] >= 0.12
        g5 = st["dsr_family"] >= 0.95 and st["placebo_p"] < 0.05
        g7 = st["sr_net_ann"] >= 1.0 and st_tk["sr_net_ann"] >= -0.5   # cost-stress
        wf_ok = all(x > -0.5 for x in st["walk_forward_sr"])
        conc_ok = st.get("min_coins_flag", 0) >= 1.5                    # not a 1-coin bet
        st["taker_stress_sr"] = st_tk["sr_net_ann"]
        v = "PASS" if (g3 and g4 and g5 and g7 and wf_ok and conc_ok) else "FAIL"
        obj = ("clears ladder" if v == "PASS" else
               (f"surrogate p={st['surrogate_p']:.2f}" if not g3 else
                f"netSR={st['sr_net_ann']:.2f}/taker {st.get('taker_stress_sr',0):.2f}, WF={st['walk_forward_sr']}, cvSR={st['cv_sr_mean']:.2f}, coins={st.get('min_coins_flag',0):.1f}, DSRfam={st['dsr_family']:.3f}"))
        results.append({**{k: st[k] for k in st}, "gate": v, "objection": obj, "exp_id": exp, "hyp_id": hyp})
        print(f"  en{cfg['enter_bp']} ma{cfg['ma']} reb{cfg['rebalance']} -> {v:4} "
              f"netSR={st['sr_net_ann']:+.2f} gross={st['sr_gross_ann']:+.2f} "
              f"harvest={st['spread_harvest_ann']*100:+.1f}%/yr cost={st['cost_ann']*100:.1f}% "
              f"MDD={st['max_drawdown']*100:.1f}% WF={st['walk_forward_sr']} "
              f"surr={st['surrogate_p']:.2f} plc={st['placebo_p']:.2f} DSRf={st['dsr_family']:.2f} coins={st['avg_coins_on']:.0f}", flush=True)
    best = max(results, key=lambda r: (r["gate"] == "PASS", r["dsr_family"], r["sr_net_ann"]), default=None)
    survivors = [r for r in results if r["gate"] == "PASS"]
    if best:
        ledger.add_verdict(hyp, "final", "PASS" if survivors else "FAIL", "validator", [exp],
                           rationale=f"best xvenue: netSR={best['sr_net_ann']:.2f} surr={best['surrogate_p']:.2f} DSRfam={best['dsr_family']:.3f}",
                           strongest_surviving_objection=best["objection"])
        if not survivors:
            ledger.add_lesson(root_cause=best["objection"],
                              lesson=f"F_XVENUE: Binance-Bybit funding spread harvest ~{best['spread_harvest_ann']*100:.1f}%/yr gross "
                                     f"but net Sharpe {best['sr_net_ann']:.2f} after 4-leg cost {RT*1e4:.0f}bps; surrogate p={best['surrogate_p']:.2f}.",
                              family="F_XVENUE", subject_id=hyp, ladder_level="L6")
            ledger.add_bound("F_XVENUE", "Binance-Bybit funding-spread pair, 50 coins, 2y, 4-leg taker cost",
                             f"F_XVENUE: {len(results)} configs. Gross spread harvest <= {best['spread_harvest_ann']*100:.1f}%/yr, "
                             f"but the 4-leg round-trip (~{RT*1e4:.0f}bps) and spread noise leave net Sharpe <= {best['sr_net_ann']:.2f}; "
                             f"surrogate p <= {best['surrogate_p']:.2f}; walk-forward {best['walk_forward_sr']}. The inter-venue funding "
                             f"inefficiency is too small / too costly to harvest at retail 4-leg taker cost with public data.", [exp])
    return {"results": results, "survivors": survivors, "best": best}
