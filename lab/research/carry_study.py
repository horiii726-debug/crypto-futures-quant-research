"""Delta-neutral funding / basis carry (F_CARRY).

Position per coin: SHORT perp + LONG spot (a carry unit) when the coin's
expected forward funding is high enough; flat otherwise; REVERSE (long perp /
short spot) only when funding is deeply negative.

P&L per bar for a 1-unit short-perp/long-spot position, held from t-1:
    + funding_received[t]            (you are short the perp; positive funding pays you)
    - d_premium[t]                   (basis P&L: premium = (perp-index)/index; if it
                                      falls you gain on the short-perp/long-spot pair)
    - turnover * roundtrip_cost      (open+close BOTH legs; spot + perp)
    - borrow                         (spot long has ~0 cost on a margin account; a
                                      short leg would; we only ever go net short perp)

This is a risk premium, not IC alpha. Gate on net Sharpe after cost + DSR
(family F_CARRY, its own trial count) + placebo (shuffle which coins are on)
+ surrogate (block-shuffle the funding series).
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from lab.engine.cv import purged_kfold
from lab.ledger import Ledger
from lab.stats.dsr import deflated_sharpe


@njit(cache=False)
def _hysteresis(ef, enter_hi, exit_hi, enter_lo):
    n, m = ef.shape
    out = np.zeros((n, m))
    for j in range(m):
        cur = 0.0
        for i in range(n):
            x = ef[i, j]
            if x != x:
                out[i, j] = cur
                continue
            if cur == 0.0:
                if x >= enter_hi:
                    cur = -1.0
                elif x <= enter_lo:
                    cur = 1.0
            elif cur == -1.0:
                if x < exit_hi:
                    cur = 0.0
            else:
                if x > 0.0:
                    cur = 0.0
            out[i, j] = cur
    return out

ROOT = Path(__file__).resolve().parents[2]
BASIS = ROOT / "data" / "processed" / "basis"

# round-trip cost of opening AND closing BOTH legs.
#   taker:  2 * (spot taker 0.10% + perp taker 0.05% + ~0.06% spread) = ~42 bps
#   maker:  2 * (spot maker 0.075% + perp maker 0.02%)                = ~19 bps
# We report both; the gate uses the maker number (a carry desk works passively)
# but a candidate must also survive taker cost-stress at G_cost.
CARRY_RT_TAKER = 2 * (0.0010 + 0.0005 + 0.0006)
CARRY_RT_MAKER = 2 * (0.00075 + 0.0002)
CARRY_ROUNDTRIP = CARRY_RT_MAKER
FUNDING_PER_YEAR = 3 * 365                   # 8h intervals


def load_basis(min_bars=8000) -> dict:
    out = {}
    for f in sorted(BASIS.glob("*.parquet")):
        d = pd.read_parquet(f)
        if "funding" in d and d["funding"].notna().sum() > min_bars:
            out[f.stem] = d
    return out


def _carry_panel(basis: dict):
    prem = pd.DataFrame({s: d["premium"] for s, d in basis.items()}).sort_index()
    fund = pd.DataFrame({s: d["funding"] for s, d in basis.items()}).reindex(prem.index)
    # bar = 8h (funding cadence): resample hourly -> 8h
    prem8 = prem.resample("8h").last()
    fund8 = fund.resample("8h").last()          # funding rate applying at that stamp
    dprem = prem8.diff()
    # funding actually PAID over each 8h stamp is fund8 (rate per interval)
    return prem8, fund8, dprem


def run_carry_study(name: str, *, entry_z: float, exit_z: float, neg_entry: float,
                    signal: str, lookback: int, rebalance_bars: int,
                    ledger: Ledger, exp_id: str, hyp_id: str,
                    n_placebo=200, n_surrogate=200, seed=0) -> dict:
    basis = load_basis()
    prem8, fund8, dprem = _carry_panel(basis)
    rng = np.random.default_rng(seed)

    # expected forward funding signal (trailing mean of the realised funding rate)
    if signal == "funding_z":
        mu = fund8.rolling(lookback * 4, min_periods=lookback).mean()
        sd = fund8.rolling(lookback * 4, min_periods=lookback).std().replace(0, np.nan)
        exp_fund = (fund8 - mu) / sd
        enter_hi, exit_hi, enter_lo = entry_z, exit_z, neg_entry * 2000  # z units
    else:  # funding_ma / level -> rate units (per 8h)
        exp_fund = fund8.rolling(lookback, min_periods=max(3, lookback // 2)).mean() \
            if signal == "funding_ma" else fund8.copy()
        enter_hi = 0.00005          # 0.5 bp/8h ~ 5.5%/yr gross before hedging
        exit_hi = 0.00000           # exit the short only when carry goes to ~0
        enter_lo = neg_entry        # deeply negative -> flip long

    # HYSTERESIS state machine (kills the churn that eats carry): once SHORT,
    # stay short until exp_fund < exit_hi; once LONG, stay until exp_fund > 0.
    st_pos = _hysteresis(np.ascontiguousarray(exp_fund.values, dtype=np.float64),
                         float(enter_hi), float(exit_hi), float(enter_lo))
    pos = pd.DataFrame(st_pos, index=fund8.index, columns=fund8.columns)

    # only re-hedge (rebalance weights) every rebalance_bars
    if rebalance_bars > 1:
        keep = np.zeros(len(pos), dtype=bool)
        keep[::rebalance_bars] = True
        pos = pos.where(pd.Series(keep, index=pos.index), np.nan).ffill().fillna(0.0)

    held = pos.shift(1)
    n_on = held.abs().sum(axis=1).replace(0, np.nan)
    w = held.div(n_on, axis=0).fillna(0.0)          # equal weight across active carries

    # per-bar pnl: short perp collects funding, pays d_premium
    #   short-perp pnl from funding = -held * (-fund8) = held*(-1)*... :
    #   held = -1 (short) => receive +fund8 ; basis pnl = -held * dprem = +dprem?
    # short perp + long spot: if premium DROPS, perp underperforms spot -> you gain.
    #   pair pnl = -(perp_ret - spot_ret) = -d_premium (approx). held=-1 => +d? no.
    #   held is the perp position. perp pnl = held*perp_ret. spot pnl = -held*spot_ret
    #   (long spot to hedge a short perp). pair = held*(perp_ret - spot_ret)
    #                                          = held * d_premium.
    #   funding: short perp (held<0) with positive funding => receive |held|*fund
    #          = -held*fund8.
    funding_pnl = (-w * fund8).sum(axis=1)
    basis_pnl = (w * dprem).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    cost = turnover * CARRY_ROUNDTRIP
    cost_taker = turnover * CARRY_RT_TAKER
    gross = funding_pnl + basis_pnl
    net = (gross - cost).dropna()
    net_taker = (gross - cost_taker).dropna()
    g = gross.reindex_like(net)

    def sr(x):
        x = x[np.isfinite(x)]
        return float(x.mean() / x.std(ddof=1)) if x.size > 5 and x.std(ddof=1) > 0 else 0.0
    ann = np.sqrt(FUNDING_PER_YEAR)

    # purged CV
    T = len(net)
    t0 = np.arange(T)
    t1 = np.minimum(t0 + max(rebalance_bars, 1), T - 1)
    fold_sr = []
    for tr, te in purged_kfold(t0, t1, n_splits=6, embargo_pct=0.02):
        if len(te) < 20:
            continue
        seg = net.iloc[np.sort(te)]
        if seg.std() > 0:
            fold_sr.append(float(seg.mean() / seg.std(ddof=1)))

    # placebo: keep the SAME on/off timing per coin but randomise the direction
    # (short vs long). If the edge is real, "short the high-funding coin"
    # must beat a random sign choice.
    wv = w.values
    fv = fund8.reindex(w.index).values
    dv = dprem.reindex(w.index).values
    absw = np.abs(wv)
    plc = np.empty(n_placebo)
    for b in range(n_placebo):
        flip = rng.choice(np.array([-1.0, 1.0]), size=wv.shape[1])
        nn = absw * flip[None, :]
        gp = np.nansum(-nn * fv, axis=1) + np.nansum(nn * dv, axis=1)
        tp = np.abs(np.diff(nn, axis=0, prepend=0.0)).sum(axis=1)
        npnl = gp - tp * CARRY_ROUNDTRIP
        npnl = npnl[np.isfinite(npnl)]
        plc[b] = float(npnl.mean() / npnl.std()) if npnl.std() > 0 else 0.0
    placebo_p = float((np.sum(plc >= sr(net)) + 1) / (n_placebo + 1))

    # surrogate: block-shuffle the funding matrix rows (breaks the funding->
    # future-funding persistence the signal relies on)
    fa = fund8.reindex(w.index).values
    dva = dprem.reindex(w.index).values
    basis_v = np.nansum(wv * dva, axis=1)
    cost_v = (np.abs(np.diff(wv, axis=0, prepend=0.0)).sum(axis=1)) * CARRY_ROUNDTRIP
    surr = np.empty(n_surrogate)
    L = len(fa)
    for b in range(n_surrogate):
        stt = rng.integers(0, L, int(np.ceil(L / 8)))
        order = np.concatenate([np.arange(s, s + 8) % L for s in stt])[:L]
        gs = np.nansum(-wv * fa[order], axis=1) + basis_v - cost_v
        gs = gs[np.isfinite(gs)]
        surr[b] = float(gs.mean() / gs.std()) if gs.std() > 0 else 0.0
    surr_p = float((np.sum(surr >= sr(net)) + 1) / (n_surrogate + 1))

    dsr = deflated_sharpe(net.values, ledger=ledger, family="F_CARRY")
    dsr_g = deflated_sharpe(net.values, ledger=ledger)

    out = {
        "name": name, "family": "F_CARRY",
        "params": dict(entry_z=entry_z, exit_z=exit_z, neg_entry=neg_entry, signal=signal,
                       lookback=lookback, rebalance_bars=rebalance_bars),
        "sr_net_ann": sr(net) * ann, "sr_gross_ann": sr(g) * ann,
        "sr_net_taker_ann": sr(net_taker) * ann,
        "ann_return_net": float(net.mean() * FUNDING_PER_YEAR),
        "ann_return_gross": float(g.mean() * FUNDING_PER_YEAR),
        "funding_ann": float(funding_pnl.reindex_like(net).mean() * FUNDING_PER_YEAR),
        "basis_ann": float(basis_pnl.reindex_like(net).mean() * FUNDING_PER_YEAR),
        "cost_ann": float(cost.reindex_like(net).mean() * FUNDING_PER_YEAR),
        "turnover_mean": float(turnover.mean()),
        "avg_coins_on": float(held.abs().sum(axis=1).mean()),
        "cv_sr_mean": float(np.nanmean(fold_sr)) if fold_sr else np.nan, "cv_folds": len(fold_sr),
        "placebo_p": placebo_p, "surrogate_p": surr_p,
        "dsr_family": dsr["dsr"], "dsr_family_n": dsr["n_trials"],
        "dsr_portfolio": dsr_g["dsr"], "dsr_portfolio_n": dsr_g["n_trials"],
        "n_bars": int(T), "max_drawdown": _mdd(net),
        "net_series": net,
    }
    for k in ["sr_net_ann", "sr_gross_ann", "ann_return_net", "funding_ann", "basis_ann",
              "cost_ann", "cv_sr_mean", "placebo_p", "surrogate_p", "dsr_family",
              "dsr_portfolio", "turnover_mean", "max_drawdown"]:
        if out[k] == out[k]:
            ledger.log_metric(exp_id, k, float(out[k]), context={"study": name})
    return out


def _mdd(net: pd.Series) -> float:
    eq = (1 + net).cumprod()
    return float((eq / eq.cummax() - 1).min())


GRID = dict(
    signal=["funding_ma", "funding_z", "level"],
    lookback=[3, 9, 21],
    entry_z=[0.5, 1.0],
    neg_entry=[-0.0005, -0.0010],
    rebalance_bars=[1, 3, 9],
)


def run(ledger: Ledger, max_full=40) -> dict:
    import yaml
    keys = list(GRID)
    cfgs = [dict(zip(keys, v)) for v in itertools.product(*[GRID[k] for k in keys])]
    hyp = ledger.add_hypothesis({
        "claim": "delta-neutral short-perp/long-spot funding carry has positive net expectancy",
        "mechanism": "perp longs pay funding to shorts; a hedged short-perp harvests it minus basis moves and 2-leg costs",
        "math_form": "pnl = -w*funding - w*d_premium - turnover*cost", "target": "carry_net_return",
        "horizon": "days-weeks", "null_hypothesis": "net Sharpe <= 0 after 2-leg cost",
        "family": "F_CARRY", "lessons_reviewed": True})
    exp = ledger.new_experiment(hyp, "F_CARRY", {"grid": GRID}, stage="discovery")
    results = []
    n_full = 0
    for cfg in cfgs:
        if cfg["signal"] != "funding_z" and cfg["entry_z"] != GRID["entry_z"][0]:
            continue  # entry_z only matters for the z signal
        n_full += 1
        if n_full > max_full:
            break
        tid = ledger.log_trial(exp, cfg, stage="discovery", status="running")
        try:
            st = run_carry_study(f"carry|{cfg}", ledger=ledger, exp_id=exp, hyp_id=hyp,
                                 n_placebo=120, n_surrogate=120,
                                 seed=abs(hash(str(cfg))) % (2**31), **cfg)
            ledger.mark_trial(tid, "done")
        except Exception as e:  # noqa
            ledger.log_metric(exp, "error", None, trial_id=tid, text_value=repr(e)[:200])
            ledger.mark_trial(tid, "failed")
            continue
        g5 = st["dsr_family"] >= 0.95 and st["placebo_p"] < 0.05
        g4 = st["cv_folds"] >= 4 and st["cv_sr_mean"] > 0
        g7 = st["sr_net_ann"] >= 0.5
        g3 = st["surrogate_p"] < 0.05
        v = "PASS" if (g3 and g4 and g5 and g7) else "FAIL"
        obj = ("clears ladder" if v == "PASS" else
               f"netSR_ann={st['sr_net_ann']:.2f} DSRfam={st['dsr_family']:.3f}/{st['dsr_family_n']} "
               f"plc={st['placebo_p']:.3f} surr={st['surrogate_p']:.3f} cvSR={st['cv_sr_mean']:.2f}")
        results.append({**{k: st[k] for k in st if k != "net_series"}, "gate": v, "objection": obj,
                        "hyp_id": hyp, "exp_id": exp})
        print(f"  carry {cfg['signal']:10} lb{cfg['lookback']} reb{cfg['rebalance_bars']} -> {v:5} "
              f"netSR={st['sr_net_ann']:+.2f} (fund {st['funding_ann']*100:+.1f}% basis {st['basis_ann']*100:+.1f}% "
              f"cost {st['cost_ann']*100:.1f}%) DSRfam={st['dsr_family']:.2f} plc={st['placebo_p']:.3f} "
              f"surr={st['surrogate_p']:.3f} MDD={st['max_drawdown']*100:.0f}%", flush=True)
    best = max(results, key=lambda r: (r["gate"] == "PASS", r["dsr_family"], r["sr_net_ann"]),
              default=None)
    survivors = [r for r in results if r["gate"] == "PASS"]
    if best:
        ledger.add_verdict(hyp, "final", "PASS" if survivors else "FAIL", "validator", [exp],
                           rationale=f"best carry: netSR_ann={best['sr_net_ann']:.2f} "
                                     f"DSRfam={best['dsr_family']:.3f} funding {best['funding_ann']*100:.1f}%/yr",
                           strongest_surviving_objection=best["objection"])
        if not survivors:
            ledger.add_lesson(root_cause=best["objection"],
                              lesson=f"F_CARRY: best delta-neutral funding carry net Sharpe(ann) "
                                     f"{best['sr_net_ann']:.2f}; gross funding {best['funding_ann']*100:.1f}%/yr "
                                     f"but basis vol + 2-leg cost {best['cost_ann']*100:.1f}%/yr erode it.",
                              family="F_CARRY", subject_id=hyp, ladder_level="L6")
            ledger.add_bound("F_CARRY", "delta-neutral short-perp/long-spot funding harvest, 55 coins, 2y",
                             f"F_CARRY: {len(results)} configs. Best net Sharpe(ann) <= {best['sr_net_ann']:.2f}; "
                             f"gross funding harvest ~{best['funding_ann']*100:.1f}%/yr, eroded by basis P&L "
                             f"volatility and the ~{CARRY_ROUNDTRIP*1e4:.0f}bps 2-leg round-trip. "
                             f"DSR(family) <= {best['dsr_family']:.3f}.", [exp])
    return {"results": results, "survivors": survivors, "best": best}
