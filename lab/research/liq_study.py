"""F_LIQD — forced-liquidation pressure as a short-horizon timing signal.

Data: Binance coin-M liquidationSnapshot, BTCUSD_PERP + ETHUSD_PERP,
2023-06..2024-09, hourly (lab/data/liqsnap.py).  Only 2 correlated instruments
=> time-series only, pooled.  This family is expected to be UNDERPOWERED; the
study exists to bound it, not to rescue it.

Signals (all causal, formed on bar t, executed t+1, R9):
  liq_sell_spike     : long-liquidation (forced selling) intensity z -> bounce (reversal)
  liq_imb_reversal   : net forced-sell imbalance -> fade
  liq_imb_follow     : net forced-sell imbalance -> continuation (cascade)
  liq_intensity_rev  : total liquidation notional z -> reversal
  liq_exhaustion     : liq spike THEN decay -> entry after the cascade

Ladder-lite: purged K-fold CV + block-bootstrap surrogate + hybrid & taker cost.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lab.engine.cv import purged_kfold
from lab.ledger import Ledger
from lab.stats.dsr import deflated_sharpe

ROOT = Path(__file__).resolve().parents[2]
LIQ = ROOT / "data" / "processed" / "liqsnap"
_EXEC = json.loads((ROOT / "data" / "processed" / "exec_summary.json").read_text())
MK = _EXEC["hybrid_roundtrip_bps_mean"] / 2 * 1e-4
TK = 0.0005 + 0.85e-4
ANN = np.sqrt(365 * 24)

_MAP = {"BTCUSD_PERP": "BTCUSDT", "ETHUSD_PERP": "ETHUSDT"}


def _price_1h():
    parts = []
    for part in ("train", "valid", "test"):
        p = ROOT / "data" / part / "prio50_2y" / "1h" / "close.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p)[["BTCUSDT", "ETHUSDT"]])
    return pd.concat(parts).sort_index()


def _z(s, n):
    return ((s - s.rolling(n, min_periods=n // 2).mean())
            / s.rolling(n, min_periods=n // 2).std().replace(0, np.nan)).clip(-5, 5)


def _panel():
    px = _price_1h()
    out = {}
    for f, sym in _MAP.items():
        b = pd.read_parquet(LIQ / f"{f}.parquet")
        b = b.reindex(px.index)
        b["px"] = px[sym]
        b[["liq_sell_usd", "liq_buy_usd", "liq_total_usd", "liq_n"]] = \
            b[["liq_sell_usd", "liq_buy_usd", "liq_total_usd", "liq_n"]].fillna(0.0)
        out[sym] = b
    return out


def _signal(name, b, n=72):
    sell = b["liq_sell_usd"]
    tot = b["liq_total_usd"]
    imb = b["liq_imb"]
    ret1 = np.log(b["px"]).diff()
    if name == "liq_sell_spike":
        return _z(np.log1p(sell), n)                       # >0 => forced selling => go long (reversal)
    if name == "liq_imb_reversal":
        return imb.rolling(3, min_periods=1).mean()         # >0 => net long-liq => go long
    if name == "liq_imb_follow":
        return -imb.rolling(3, min_periods=1).mean()        # continuation
    if name == "liq_intensity_rev":
        # big total liquidation + recent down move => bounce
        return _z(np.log1p(tot), n) * (-np.sign(ret1.rolling(4, min_periods=1).sum()))
    if name == "liq_exhaustion":
        zt = _z(np.log1p(tot), n)
        return (zt.shift(2) - zt).clip(lower=0) * (-np.sign(ret1.rolling(6, min_periods=1).sum()))
    raise KeyError(name)


SIGNALS = ["liq_sell_spike", "liq_imb_reversal", "liq_imb_follow",
           "liq_intensity_rev", "liq_exhaustion"]
GRID_N = [48, 72, 168]
HOLDS = [2, 6, 12, 24]


def _sr(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / x.std(ddof=1))


def run_one(name, n, hold, panel, ledger, exp, hyp, seed=0):
    rng = np.random.default_rng(seed)
    pnl_parts, gross_parts, turn_parts = [], [], []
    ic_num = ic_den = 0.0
    per_sym = {}
    for sym, b in panel.items():
        sig = _signal(name, b, n=n)
        raw = np.tanh(sig.fillna(0.0))                       # bounded position
        pos = raw.rolling(hold, min_periods=1).mean()
        bwd = b["px"] / b["px"].shift(1) - 1.0
        held = pos.shift(2)                                  # R9
        gross = held * bwd
        turn = held.diff().abs().fillna(held.abs())
        m = np.isfinite(gross) & np.isfinite(turn)
        gross_parts.append(gross[m]); turn_parts.append(turn[m])
        fwd = b["px"].shift(-1) / b["px"] - 1.0
        s = sig.reindex(gross.index); f = fwd.reindex(gross.index)
        mm = np.isfinite(s) & np.isfinite(f)
        if mm.sum() > 20:
            c = np.corrcoef(s[mm], f[mm])[0, 1]
            ic_num += c * mm.sum(); ic_den += mm.sum()
        per_sym[sym] = float((gross[m] - turn[m] * MK).sum())
    G = pd.concat(gross_parts); TU = pd.concat(turn_parts)
    net_mk = (G - TU * MK).dropna()
    net_tk = (G - TU * TK).dropna()
    ic = ic_num / ic_den if ic_den else 0.0

    # purged CV on the pooled series (index order = concat order; fine for block CV)
    g = G.dropna().reset_index(drop=True)
    t0 = np.arange(len(g)); t1 = np.minimum(t0 + hold + 2, len(g) - 1)
    nf = (G - TU * MK).dropna().reset_index(drop=True)
    fold = []
    for tr, te in purged_kfold(t0, t1, n_splits=5, embargo_pct=0.02):
        seg = nf.iloc[np.sort(te)]
        if len(seg) > 10 and seg.std() > 0:
            fold.append(_sr(seg) * ANN)

    # block-bootstrap surrogate: shuffle 24-bar blocks of per-symbol returns,
    # re-score the SAME positions -> distribution of net Sharpe under no timing.
    surr = np.empty(200)
    base_pos, base_ret = [], []
    for sym, b in panel.items():
        sig = _signal(name, b, n=n)
        pos = np.tanh(sig.fillna(0.0)).rolling(hold, min_periods=1).mean().shift(2)
        r = (b["px"] / b["px"].shift(1) - 1.0)
        mm = np.isfinite(pos) & np.isfinite(r)
        base_pos.append(pos[mm].values); base_ret.append(r[mm].values)
    for i in range(200):
        tot = []
        for p_, r_ in zip(base_pos, base_ret):
            L = len(r_); st = rng.integers(0, L, int(np.ceil(L / 24)))
            order = np.concatenate([np.arange(x, x + 24) % L for x in st])[:L]
            rr = r_[order]
            dp = np.abs(np.diff(p_, prepend=0.0))
            tot.append(p_ * rr - dp * MK)
        nn = np.concatenate(tot)
        surr[i] = _sr(nn) * ANN
    obs = _sr(net_mk) * ANN
    surr_p = float((np.sum(surr >= obs) + 1) / 201)

    dsr = deflated_sharpe(net_mk.values, ledger=ledger, family="F_LIQD")
    n_eff = len(net_mk)
    wf = [round(_sr(net_mk.iloc[j * n_eff // 4:(j + 1) * n_eff // 4]) * ANN, 2) for j in range(4)]
    out = {"signal": name, "n": n, "hold": hold, "ic": ic,
           "maker_sr_ann": _sr(net_mk) * ANN, "taker_sr_ann": _sr(net_tk) * ANN,
           "gross_sr_ann": _sr(G.dropna()) * ANN, "turnover": float(TU.mean()),
           "cv_folds": len(fold), "cv_sr_mean": float(np.nanmean(fold)) if fold else float("nan"),
           "surrogate_p": surr_p, "dsr_family": dsr["dsr"], "dsr_family_n": dsr["n_trials"],
           "walk_forward": wf, "per_symbol_pnl": per_sym, "n_bars": n_eff}
    for kk in ("ic", "maker_sr_ann", "taker_sr_ann", "gross_sr_ann", "turnover",
               "cv_sr_mean", "surrogate_p", "dsr_family"):
        if out[kk] == out[kk]:
            ledger.log_metric(exp, kk, float(out[kk]), context={"study": f"liq_{name}"})
    return out


def run(ledger: Ledger) -> dict:
    panel = _panel()
    results, verdicts, survivors = [], [], []
    for name in SIGNALS:
        hyp = ledger.add_hypothesis({
            "claim": f"{name}: coin-M forced-liquidation pressure times BTC/ETH short-horizon returns",
            "mechanism": "forced (price-insensitive) liquidation flow overshoots -> mean reversion; "
                         "or triggers a cascade -> short continuation",
            "math_form": name, "target": "btc_eth_fwd_return_1h", "horizon": "2-24h",
            "null_hypothesis": "net maker Sharpe <= 0 ; surrogate-indistinguishable",
            "family": "F_LIQD", "lessons_reviewed": True})
        exp = ledger.new_experiment(hyp, "F_LIQD",
                                    {"signal": name, "ns": GRID_N, "holds": HOLDS,
                                     "instruments": ["BTCUSD_PERP", "ETHUSD_PERP"],
                                     "window": "2023-09..2024-09"}, stage="discovery")
        best = None
        for n in GRID_N:
            for hold in HOLDS:
                tid = ledger.log_trial(exp, {"n": n, "hold": hold}, stage="discovery", status="running")
                try:
                    st = run_one(name, n, hold, panel, ledger, exp, hyp,
                                 seed=abs(hash((name, n, hold))) % (2**31))
                    ledger.mark_trial(tid, "done")
                except Exception as e:
                    ledger.log_metric(exp, "error", None, trial_id=tid, text_value=repr(e)[:200])
                    ledger.mark_trial(tid, "failed"); continue
                g3 = st["surrogate_p"] < 0.05
                g4 = st["cv_folds"] >= 3 and st["cv_sr_mean"] > 0
                g5 = st["dsr_family"] >= 0.90
                g7 = st["maker_sr_ann"] >= 0.8 and st["taker_sr_ann"] >= -0.3
                wf_ok = all(x > -0.5 for x in st["walk_forward"])
                v = "PASS" if (g3 and g4 and g5 and g7 and wf_ok) else "FAIL"
                results.append({**st, "gate": v, "exp_id": exp, "hyp_id": hyp})
                print(f"  {name:18} n{n:3} h{hold:2} -> {v:4} "
                      f"IC={st['ic']:+.4f} mkSR={st['maker_sr_ann']:+.2f} tkSR={st['taker_sr_ann']:+.2f} "
                      f"grSR={st['gross_sr_ann']:+.2f} surr={st['surrogate_p']:.3f} "
                      f"DSRf={st['dsr_family']:.2f} cvSR={st['cv_sr_mean']:+.2f} WF={st['walk_forward']}",
                      flush=True)
                if v == "PASS":
                    survivors.append(results[-1])
                if best is None or st["maker_sr_ann"] > best["maker_sr_ann"]:
                    best = results[-1]
        if best:
            status = "PASS" if best["gate"] == "PASS" else "FAIL"
            obj = (f"best {best['signal']} n{best['n']} h{best['hold']}: maker SR "
                   f"{best['maker_sr_ann']:.2f}, taker {best['taker_sr_ann']:.2f}, "
                   f"surrogate p {best['surrogate_p']:.3f}, DSR(F_LIQD) {best['dsr_family']:.2f}, "
                   f"WF {best['walk_forward']} (2 correlated instruments only -> underpowered)")
            ledger.add_verdict(hyp, "final", status, "validator", [best["exp_id"]],
                               rationale=obj, strongest_surviving_objection=obj)
            verdicts.append({"signal": name, "status": status, "best": {k: best[k] for k in
                             ("n", "hold", "ic", "maker_sr_ann", "taker_sr_ann", "gross_sr_ann",
                              "surrogate_p", "dsr_family", "walk_forward")}})
            if status != "PASS":
                ledger.add_lesson(root_cause=obj[:120],
                                  lesson=f"F_LIQD {name}: best maker Sharpe {best['maker_sr_ann']:.2f}, "
                                         f"gross {best['gross_sr_ann']:.2f}, surrogate p {best['surrogate_p']:.3f}. "
                                         f"Only BTCUSD_PERP+ETHUSD_PERP coin-M liq data -> underpowered.",
                                  family="F_LIQD", subject_id=hyp, ladder_level="L4")
        ledger.close_experiment(exp, "done")

    if not survivors:
        exs = sorted({r["exp_id"] for r in results})
        b_gr = max((r["gross_sr_ann"] for r in results), default=0.0)
        b_mk = max((r["maker_sr_ann"] for r in results), default=0.0)
        ledger.add_bound("F_LIQD", "forced-liquidation timing of BTC/ETH (2-24h)",
                         f"Across {len(results)} configs / 5 signal forms on hourly coin-M "
                         f"liquidation pressure (BTCUSD_PERP+ETHUSD_PERP, 2023-09..2024-09), best "
                         f"gross annualised Sharpe {b_gr:.2f}, best maker-net {b_mk:.2f}. No config "
                         f"clears the surrogate + CV + cost + DSR ladder. Only 2 highly-correlated "
                         f"instruments are available (Binance publishes coin-M liquidationSnapshot "
                         f"only for BTCUSD_PERP/ETHUSD_PERP) so the effective sample is a single "
                         f"BTC/ETH stress series -> UNDERPOWERED and bounded.",
                         exs)
    return {"results": results, "verdicts": verdicts, "survivors": survivors}


if __name__ == "__main__":
    import os
    os.environ.setdefault("CRYPTO_LAB_LEDGER_SQLITE", "./ledger/lab.sqlite")
    L = Ledger()
    r = run(L)
    json.dump({"survivors": [s["signal"] for s in r["survivors"]],
               "verdicts": r["verdicts"], "n": len(r["results"])},
              open("reports/LIQ_RESULT.json", "w"), indent=1, default=float)
    print("\nLIQ DONE survivors=", len(r["survivors"]))
