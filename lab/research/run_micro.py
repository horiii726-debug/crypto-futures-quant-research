"""S3 microstructure orchestrator (time-series, maker execution).

Two-tier (SYSTEM_SPEC [FIX 1]):
  cheap screen  -> rough IC + turnover + cost feasibility, NOT a trial
  full study    -> gate ladder, a family-scoped trial

Pre-registered grid per feature. Escalation: thr -> hold -> bar -> next feature
(L3/L2/L4). Survivors: portfolio DSR + one sealed S7 test look.
"""
from __future__ import annotations

import json
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lab.ledger import Ledger
from lab.research.micro import REGISTRY, FAMILY
from lab.research.micro_study import run_micro_study, MAKER_ONEWAY

ROOT = Path(__file__).resolve().parents[2]
TAPE = ROOT / "data" / "processed" / "tape"
GATES = yaml.safe_load((ROOT / "config" / "gates.yaml").read_text())
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT",
        "LINKUSDT", "AVAXUSDT", "ADAUSDT", "LTCUSDT"]

# pre-registered grid. lookback params already sensible in bars.
GRID = {
    "ofi_momentum":      dict(lb=[3, 6, 12], znorm=[200]),
    "ofi_reversal":      dict(znorm=[100, 200]),
    "aggr_imb_persist":  dict(lb=[4, 8, 16]),
    "cum_delta_slope":   dict(lb=[6, 12, 24]),
    "big_trade_follow":  dict(lb=[2, 4, 8]),
    "vwap_tilt":         dict(znorm=[100, 200]),
    "vwap_reversion":    dict(znorm=[100, 200]),
    "signac_momentum":   dict(lb=[4, 8], ac_lb=[20]),
    "intensity_breakout": dict(lb=[20, 40]),
    "ofi_plus_vwap":     dict(lb=[6, 12], znorm=[200]),
}
HOLDS = [1, 3, 6]
THRS = [0.0, 0.15]
BARS = [3]

TH = {**GATES["thresholds"]["default"]}


def load_panels(bar_min: int) -> dict:
    out = {}
    for s in SYMS:
        f = TAPE / f"{s}_{bar_min}m.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            if len(df) > 2000:
                out[s] = df
    return out


def cheap_screen(panel: dict, feature: str, params: dict, hold: int, thr: float,
                 bar_min: int) -> dict:
    """rough IC on the last 30% as a holdout, turnover, cost feasibility.
    NOT a trial."""
    fn = REGISTRY[feature]
    ics, turns, edges = [], [], []
    for s, df in panel.items():
        n = len(df)
        cut = int(n * 0.7)
        sig = fn(df, **params).astype(float)
        p = sig.where(sig.abs() >= thr, 0.0)
        if hold > 1:
            p = p.rolling(hold, min_periods=1).mean()
        r = np.log(df["close"]).diff()
        fwd = r.shift(-2)
        h = p.iloc[cut:]
        y = fwd.iloc[cut:]
        m = np.isfinite(h) & np.isfinite(y)
        if m.sum() < 200 or h[m].std() == 0:
            continue
        ics.append(float(np.corrcoef(h[m], y[m])[0, 1]))
        turns.append(float(p.diff().abs().mean()))
        edges.append(float((h[m] * y[m]).mean()))
    if not ics:
        return {"pass": False, "reason": "no_data"}
    ic = float(np.nanmean(ics))
    turn = float(np.nanmean(turns))
    gross_edge_bar = float(np.nanmean(edges))
    cost_bar = turn * MAKER_ONEWAY
    net_edge_bar = gross_edge_bar - cost_bar
    ok = abs(ic) >= 0.008 and net_edge_bar > 0 and turn < 2.0
    return {"pass": ok, "ic": ic, "turnover": turn,
            "gross_edge_bps": gross_edge_bar * 1e4, "net_edge_bps": net_edge_bar * 1e4,
            "cost_bps": cost_bar * 1e4}


def gate_verdict(st: dict) -> tuple[str, str]:
    mi = 0.01  # micro IC floor (intraday signals can be smaller than daily XS)
    g3 = st["surrogate_p"] < 0.05 and abs(st["ic_fwd2"]) >= mi
    g4 = (st["cv_folds"] >= 4 and st["cv_sr_mean"] == st["cv_sr_mean"]
          and st["cv_sr_mean"] > 0.0)
    g5 = st["dsr_family"] >= 0.95 and st["placebo_p"] < 0.05
    g7 = st["maker_sr_net_ann"] >= 0.5
    if not g3:
        return "FAIL", (f"in-sample |IC|={abs(st['ic_fwd2']):.4f} vs floor {mi}, "
                        f"surrogate p={st['surrogate_p']:.3f}")
    if not g4:
        return "FAIL", f"CV net Sharpe {st['cv_sr_mean']:.2f} over {st['cv_folds']} folds not > 0"
    if not g5:
        return "FAIL", (f"DSR(family)={st['dsr_family']:.3f} vs 0.95 against "
                        f"{st['dsr_family_n']} family trials; placebo p={st['placebo_p']:.3f}")
    if not g7:
        return "FAIL", (f"maker net Sharpe(ann) {st['maker_sr_net_ann']:.2f} < 0.5; "
                        f"cost drag {st['cost_drag_per_bar_bps']:.2f} bps/bar")
    return "PASS", "clears the discovery ladder maker-executed; OOS look pending"


def run(ledger: Ledger, max_full_per_family: int = 40) -> dict:
    panels = {b: load_panels(b) for b in BARS}
    panels = {b: p for b, p in panels.items() if len(p) >= 4}
    if not panels:
        return {"error": "no tape panels built yet"}
    print(f"panels: { {b: list(p) for b, p in panels.items()} }")

    fam_full = {}
    results, verdicts, survivors = [], [], []
    for feature, gp in GRID.items():
        fam = FAMILY[feature]
        keys = list(gp)
        cfgs = [dict(zip(keys, v)) for v in itertools.product(*[gp[k] for k in keys])]
        # rename lb -> the feature's actual arg
        argname = {"ofi_momentum": "lb", "aggr_imb_persist": "lb", "cum_delta_slope": "lb",
                   "big_trade_follow": "lb", "signac_momentum": "lb", "intensity_breakout": "lb",
                   "ofi_plus_vwap": "lb"}.get(feature)
        hyp = ledger.add_hypothesis({
            "claim": f"{feature}: tape microstructure predicts next-bar return",
            "mechanism": f"see lab/research/micro.py::{feature}",
            "math_form": feature, "target": "bar_fwd_return_t2", "horizon": "intraday",
            "null_hypothesis": "IC=0 ; maker net Sharpe <= 0", "family": fam,
            "lessons_reviewed": True})
        exp = ledger.new_experiment(hyp, fam, {"feature": feature, "grid": gp,
                                               "bars": BARS, "holds": HOLDS, "thrs": THRS},
                                    stage="discovery")
        for bar_min in panels:
            for cfg, hold, thr in itertools.product(cfgs, HOLDS, THRS):
                params = dict(cfg)
                if "lb" in params and argname and argname != "lb":
                    params[argname] = params.pop("lb")
                elif "lb" in params and feature in ("vwap_tilt", "vwap_reversion", "ofi_reversal"):
                    params.pop("lb", None)
                sc = cheap_screen(panels[bar_min], feature, params, hold, thr, bar_min)
                ledger.log_screen_eval(exp, {**params, "bar": bar_min, "hold": hold, "thr": thr}, sc)
                if not sc.get("pass"):
                    continue
                if fam_full.get(fam, 0) >= max_full_per_family:
                    continue
                fam_full[fam] = fam_full.get(fam, 0) + 1
                tid = ledger.log_trial(exp, {**params, "bar": bar_min, "hold": hold, "thr": thr},
                                       stage="discovery", status="running")
                try:
                    st = run_micro_study(f"{feature}|{bar_min}m|h{hold}|t{thr}|{params}",
                                         feature, params, bar_min=bar_min, hold=hold, thr=thr,
                                         tape_panel=panels[bar_min], ledger=ledger,
                                         exp_id=exp, hyp_id=hyp, n_placebo=150, n_surrogate=150,
                                         seed=abs(hash((feature, bar_min, hold, thr, str(params)))) % (2**31))
                    ledger.mark_trial(tid, "done")
                except Exception as e:  # noqa
                    ledger.log_metric(exp, "error", None, trial_id=tid, text_value=repr(e)[:200])
                    ledger.mark_trial(tid, "failed")
                    continue
                v, obj = gate_verdict(st)
                rec = {**{k: st[k] for k in st if k != "net"}, "gate": v, "objection": obj,
                       "hyp_id": hyp, "exp_id": exp, "screen": sc}
                results.append(rec)
                print(f"  {feature:18} {bar_min}m h{hold} t{thr} -> {v:5} "
                      f"IC={st['ic_fwd2']:+.4f} makerSR={st['maker_sr_net_ann']:+.2f} "
                      f"takerSR={st['taker_sr_net_ann']:+.2f} DSRfam={st['dsr_family']:.2f} "
                      f"plc={st['placebo_p']:.3f} surr={st['surrogate_p']:.3f}", flush=True)
                if v == "PASS":
                    survivors.append(rec)
        # per-feature verdict
        best = max((r for r in results if r["feature"] == feature),
                   key=lambda r: (r["gate"] == "PASS", r["dsr_family"], r["maker_sr_net_ann"]),
                   default=None)
        if best:
            status = "PASS" if best["gate"] == "PASS" else "FAIL"
            ledger.add_verdict(hyp, "final", status, "validator", [exp],
                               rationale=f"best of grid: {feature} makerSR={best['maker_sr_net_ann']:.2f} "
                                         f"IC={best['ic_fwd2']:.4f} DSRfam={best['dsr_family']:.3f}",
                               strongest_surviving_objection=best["objection"])
            verdicts.append({"feature": feature, "family": fam, "status": status,
                             "best": {k: best[k] for k in ("bar_min", "hold", "thr", "params",
                                      "ic_fwd2", "maker_sr_net_ann", "taker_sr_net_ann",
                                      "dsr_family", "placebo_p", "surrogate_p", "objection")}})
            if status != "PASS":
                ledger.add_lesson(root_cause=best["objection"],
                                  lesson=f"{feature} tape micro: best maker net Sharpe(ann) "
                                         f"{best['maker_sr_net_ann']:.2f}, IC {best['ic_fwd2']:.4f}, "
                                         f"DSR(fam) {best['dsr_family']:.3f}. Screen passed {sum(1 for r in results if r['feature']==feature)} configs to full.",
                                  family=fam, subject_id=hyp, ladder_level="L4")

    # families with no survivor -> BOUND
    for fam in set(FAMILY.values()):
        if not any(v["family"] == fam and v["status"] == "PASS" for v in verdicts):
            fr = [r for r in results if r["family"] == fam]
            ledger.add_bound(fam, "time-series tape microstructure, maker execution, M5/M15",
                             _bound(fam, fr, ledger), sorted({r["exp_id"] for r in fr}))
    return {"results": results, "verdicts": verdicts, "survivors": survivors,
            "family_full_trials": fam_full,
            "screen_evals": ledger.screen_count()}


def _bound(fam, fr, ledger) -> str:
    if not fr:
        return (f"{fam}: no config passed the cheap screen (rough |IC|>=0.008 + net-positive "
                f"per-bar edge + turnover<2). Tape-microstructure signal at M5/M15 with maker "
                f"execution does not clear the screen on 10 liquid coins.")
    b_ic = max(abs(r["ic_fwd2"]) for r in fr)
    b_sr = max(r["maker_sr_net_ann"] for r in fr)
    b_dsr = max(r["dsr_family"] for r in fr)
    n = ledger.trial_count(family=fam)
    return (f"{fam}: {len(fr)} full trials (of {ledger.screen_count(family=fam)} screened) on "
            f"10-coin tape at M5/M15, maker execution. Best |IC(t+2)| <= {b_ic:.4f}; best maker "
            f"net Sharpe(ann) <= {b_sr:.2f}; best family DSR <= {b_dsr:.3f} vs {n} trials. "
            f"Intraday tape microstructure at these bar sizes is bounded below the maker cost "
            f"floor for a bar-frequency directional signal.")
