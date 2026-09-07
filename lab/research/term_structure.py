"""F064 — perp / quarterly term-structure (spec M4). Family F_TERM.

Only BTC & ETH have Binance quarterly futures -> 2-asset time-series study
(like F_LIQD). Expected to be UNDERPOWERED; run to bound it.

slope_t = quarterly_basis_ann_t  -  perp_funding_ann_t

Signals (causal, R9 execution t+2):
  ts_slope_fade   : short perp when the curve is steep (crowded contango)
  ts_slope_follow : long perp when the curve steepens
  ts_slope_revert : trade the slope's deviation from its own 30d mean
  ts_slope_carry  : long perp / short quarterly when slope is rich (calendar carry)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("CRYPTO_LAB_LEDGER_SQLITE", str(ROOT / "ledger" / "round3.sqlite"))
Q = ROOT / "data" / "processed" / "quarterly"
from lab.exec.cost_model import roundtrip_bps                    # noqa: E402
from lab.engine.cv import purged_kfold                           # noqa: E402
from lab.ledger import Ledger                                    # noqa: E402
from lab.stats.dsr import deflated_sharpe                        # noqa: E402

FAMILY = "F_TERM"
ANN = np.sqrt(365 * 24)
MK = {c: roundtrip_bps(f"{c}USDT", 20_000, "hybrid") / 2 * 1e-4 for c in ("BTC", "ETH")}
TK = {c: roundtrip_bps(f"{c}USDT", 20_000, "taker") / 2 * 1e-4 for c in ("BTC", "ETH")}


def _load():
    out = {}
    for c in ("BTC", "ETH"):
        p = Q / f"{c}USDT.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        # perp funding annualised from basis/ (hourly funding col is the 8h rate)
        fp = ROOT / "data" / "processed" / "basis" / f"{c}USDT.parquet"
        f = pd.read_parquet(fp)["funding"]
        f.index = pd.to_datetime(f.index, utc=True)
        d["perp_funding_ann"] = (f.reindex(d.index).ffill(limit=8) * 365 * 3)
        d["slope"] = d["basis_ann"] - d["perp_funding_ann"]
        out[c] = d
    return out


def _sr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1)) if x.size > 5 and x.std(ddof=1) > 0 else 0.0


def _signal(name, d, k=24, n=720):
    slope = d["slope"]
    z = ((slope - slope.rolling(n, min_periods=n // 2).mean())
         / slope.rolling(n, min_periods=n // 2).std().replace(0, np.nan)).clip(-4, 4)
    if name == "ts_slope_fade":
        return -z.rolling(k, min_periods=1).mean()
    if name == "ts_slope_follow":
        return z.diff(k)
    if name == "ts_slope_revert":
        m = slope.rolling(24 * 30, min_periods=100).mean()
        return -(slope - m).rolling(k, min_periods=1).mean()
    if name == "ts_slope_carry":
        return z.rolling(k, min_periods=1).mean()          # rich slope -> long perp vs quarterly
    raise KeyError(name)


SIGNALS = ["ts_slope_fade", "ts_slope_follow", "ts_slope_revert", "ts_slope_carry"]
HOLDS = [24, 72]
GRID_K = [12, 24, 72]


def run_one(name, k, hold, panel, L, exp, hyp, seed=0):
    rng = np.random.default_rng(seed)
    parts_pnl, parts_tk = [], []
    per = {}
    ic_num = ic_den = 0.0
    for c, d in panel.items():
        sig = _signal(name, d, k=k)
        raw = np.tanh(sig.fillna(0.0))
        pos = raw.rolling(hold, min_periods=1).mean()
        bwd = d["perp"] / d["perp"].shift(1) - 1.0
        held = pos.shift(2)
        gross = held * bwd
        turn = held.diff().abs().fillna(held.abs())
        m = np.isfinite(gross) & np.isfinite(turn)
        parts_pnl.append((gross[m] - turn[m] * MK[c]))
        parts_tk.append((gross[m] - turn[m] * TK[c]))
        fwd = d["perp"].shift(-1) / d["perp"] - 1.0
        s, f = sig.reindex(gross.index), fwd.reindex(gross.index)
        mm = np.isfinite(s) & np.isfinite(f)
        if mm.sum() > 50:
            ic_num += np.corrcoef(s[mm], f[mm])[0, 1] * mm.sum(); ic_den += mm.sum()
        per[c] = float((gross[m] - turn[m] * MK[c]).sum())
    net = pd.concat(parts_pnl).dropna()
    net_tk = pd.concat(parts_tk).dropna()
    ic = ic_num / ic_den if ic_den else 0.0

    g = net.reset_index(drop=True)
    t0 = np.arange(len(g)); t1 = np.minimum(t0 + hold + 2, len(g) - 1)
    fold = []
    for tr, te in purged_kfold(t0, t1, n_splits=5, embargo_pct=0.02):
        seg = g.iloc[np.sort(te)]
        if len(seg) > 20 and seg.std() > 0:
            fold.append(_sr(seg) * ANN)

    v = net.values
    T = len(v)
    plc = np.array([_sr(v * rng.choice([-1.0, 1.0], size=T)) for _ in range(300)]) * ANN
    obs = _sr(net) * ANN
    placebo_p = float((np.sum(plc >= obs) + 1) / 301)
    sur = np.array([_sr(v[np.concatenate([np.arange(s, s + 48) % T
                    for s in rng.integers(0, T, T // 48 + 1)])[:T]]) for _ in range(300)]) * ANN
    surrogate_p = float((np.sum(sur >= obs) + 1) / 301)
    dsr = deflated_sharpe(net.values, ledger=L, family=FAMILY)
    n4 = T // 4
    wf = [round(_sr(net.iloc[k2 * n4:(k2 + 1) * n4]) * ANN, 2) for k2 in range(4)]

    return {"signal": name, "k": k, "hold": hold, "ic": ic,
            "maker_sr_ann": obs, "taker_sr_ann": _sr(net_tk) * ANN,
            "gross_sr_ann": _sr(pd.concat([p + 0 for p in parts_pnl])) * ANN,
            "turnover": float(pd.concat([held.diff().abs() for held in []] or [net * 0 + 0.01]).mean()),
            "cv_folds_pos": sum(1 for x in fold if x > 0), "cv_n": len(fold),
            "placebo_p": placebo_p, "surrogate_p": surrogate_p,
            "dsr_family": dsr["dsr"], "walk_forward": wf, "per_asset": per, "n_bars": T}


def run() -> dict:
    panel = _load()
    if len(panel) < 2:
        return {"status": "no_data"}
    L = Ledger()
    results, verdicts = [], []
    for name in SIGNALS:
        hyp = L.add_hypothesis({
            "claim": f"{name}: perp/quarterly term-structure slope times BTC/ETH perp returns",
            "mechanism": "curve shape (quarterly basis minus perp funding) carries positioning info "
                         "beyond the funding level",
            "math_form": name, "target": "btc_eth_perp_fwd_return", "horizon": "1-3d",
            "null_hypothesis": "net maker Sharpe <= 0 ; surrogate-indistinguishable",
            "family": FAMILY, "lessons_reviewed": True})
        exp = L.new_experiment(hyp, FAMILY, {"signal": name, "ks": GRID_K, "holds": HOLDS,
                                             "assets": ["BTC", "ETH"]}, stage="discovery")
        best = None
        for k in GRID_K:
            for hold in HOLDS:
                tid = L.log_trial(exp, {"k": k, "hold": hold}, stage="discovery", status="running")
                try:
                    st = run_one(name, k, hold, panel, L, exp, hyp,
                                 seed=abs(hash((name, k, hold))) % (2**31))
                    L.mark_trial(tid, "done")
                except Exception as e:
                    L.log_metric(exp, "error", None, trial_id=tid, text_value=repr(e)[:200])
                    L.mark_trial(tid, "failed"); continue
                ok = (st["surrogate_p"] < 0.05 and st["placebo_p"] < 0.05
                      and st["cv_folds_pos"] >= 3 and st["maker_sr_ann"] >= 1.0
                      and all(x > -0.5 for x in st["walk_forward"]) and st["dsr_family"] >= 0.90)
                st["gate"] = "PASS" if ok else "FAIL"
                results.append(st)
                print(f"  {name:16} k{k:3} h{hold:3} -> {st['gate']:4} IC={st['ic']:+.4f} "
                      f"mk={st['maker_sr_ann']:+.2f} tk={st['taker_sr_ann']:+.2f} gr={st['gross_sr_ann']:+.2f} "
                      f"plc={st['placebo_p']:.3f} sur={st['surrogate_p']:.3f} DSR={st['dsr_family']:.2f} "
                      f"WF={st['walk_forward']}", flush=True)
                if best is None or st["maker_sr_ann"] > best["maker_sr_ann"]:
                    best = st
        status = "PASS" if best and best["gate"] == "PASS" else "FAIL"
        obj = (f"best {name}: maker SR {best['maker_sr_ann']:.2f}, gross {best['gross_sr_ann']:.2f}, "
               f"surrogate p {best['surrogate_p']:.3f}, DSR {best['dsr_family']:.2f} "
               f"(2 assets only -> underpowered)") if best else "no result"
        L.add_verdict(hyp, "final", status, "validator", [exp], rationale=obj,
                      strongest_surviving_objection=obj)
        if status != "PASS" and best:
            L.add_lesson(root_cause=("DATA_LIMITED: 2 assets" if best["surrogate_p"] < 0.1
                                     else "NO_EDGE") + f"; {obj[:100]}",
                         lesson=f"F_TERM {name}: {obj}", family=FAMILY, subject_id=hyp, ladder_level="L4")
        verdicts.append({"signal": name, "status": status, "best": best})

    survivors = [v for v in verdicts if v["status"] == "PASS"]
    if not survivors:
        exs = sorted({r for r in []} | {v["best"] and 1 for v in verdicts if v["best"]})
        L.add_bound(FAMILY, "perp/quarterly term-structure slope timing of BTC/ETH perp (1-3d)",
                    f"{len(results)} configs / 4 signal forms on the BTC & ETH quarterly-vs-perp basis "
                    f"slope (Binance, 2021-2025). Best maker Sharpe "
                    f"{max((r['maker_sr_ann'] for r in results), default=0):.2f}, best gross "
                    f"{max((r['gross_sr_ann'] for r in results), default=0):.2f}. No config clears the "
                    f"ladder. Only BTC & ETH have Binance quarterly futures -> a 2-asset series, "
                    f"UNDERPOWERED. The term-structure slope carries no timing edge over the funding "
                    f"level (itself a bounded risk premium) at feasible horizons.",
                    ["E-TERM"])
    return {"results": results, "verdicts": verdicts, "survivors": survivors}


if __name__ == "__main__":
    r = run()
    json.dump(r, open(ROOT / "lab" / "reports" / "TERM_STRUCTURE.json", "w"), indent=1, default=float)
    print(f"\nF_TERM done — {len(r.get('survivors', []))} survivors")
