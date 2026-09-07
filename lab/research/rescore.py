"""RESEARCH ROUND 2 · P0.5 — re-score every existing trial with the per-coin
cost model. NOT a new experiment: no trial is logged, DSR denominators are the
real (unchanged) ledger counts, signals/params/universe/dates are untouched.

Output: lab/reports/RESCORE_305.md
    family | mechanism | horizon | v1_net_sharpe | v2_net_sharpe | delta | v1_gate -> v2_gate
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("CRYPTO_LAB_LEDGER_SQLITE", str(ROOT / "ledger" / "lab.sqlite"))

from lab.ledger import Ledger                       # noqa: E402
from lab.research import study as S                 # noqa: E402
from lab.research import campaign as C              # noqa: E402
from lab.research import gates as gatemod           # noqa: E402
from lab.research.main_s1_s7 import _load_research_panel  # noqa: E402

GATES = S.GATES


class RescoreLedger:
    """read the real trial counts; swallow every write."""
    def __init__(self, real: Ledger):
        self._real = real

    def trial_count(self, family=None, hyp_id=None):
        return self._real.trial_count(family=family, hyp_id=hyp_id)

    # ---- writes: no-ops -------------------------------------------------
    def add_hypothesis(self, *a, **k):   return "H-RESCORE"
    def new_experiment(self, *a, **k):   return "E-RESCORE"
    def log_trial(self, *a, **k):        return "T-RESCORE"
    def mark_trial(self, *a, **k):       return None
    def log_metric(self, *a, **k):       return None
    def add_verdict(self, *a, **k):      return "V-RESCORE"
    def add_lesson(self, *a, **k):       return "L-RESCORE"
    def add_bound(self, *a, **k):        return "B-RESCORE"
    def close_experiment(self, *a, **k): return None


def _gate(st):
    try:
        return gatemod.evaluate(st, GATES)["status"]
    except Exception as e:                       # pragma: no cover
        return f"ERR:{e!r}"[:40]


# --------------------------------------------------------------------------- #
#  cross-sectional families (campaign.HYPOTHESES) @ 1d, 3d                    #
# --------------------------------------------------------------------------- #
def rescore_xsec(rl: RescoreLedger) -> list[dict]:
    rows = []
    for bar in ("1d", "3d"):
        panel = _load_research_panel(["train", "valid"], bar=bar)
        bar_days = panel.bar_hours / 24.0
        for fam, specs in C.HYPOTHESES.items():
            for feature, grid, mode, mechanism in specs:
                configs = C._grid(grid, bar_days)
                for cfg in configs:
                    common = dict(feature=feature, params=cfg, horizon_bars=1,
                                  panel_train=panel, family=fam, hyp_id="H-RESCORE",
                                  exp_id="E-RESCORE", ledger=rl, q=0.2, mode=mode,
                                  n_placebo=150, n_surrogate=150,
                                  seed=abs(hash((feature, bar, str(cfg)))) % (2**31))
                    try:
                        S.COST_MODEL = "v1_flat"
                        v1 = S.run_study(f"{feature}|{bar}|{cfg}", **common)
                        S.COST_MODEL = "v2_percoin"
                        v2 = S.run_study(f"{feature}|{bar}|{cfg}", **common)
                    except Exception as e:            # pragma: no cover
                        rows.append(dict(family=fam, mechanism=feature, horizon=bar,
                                         cfg=str(cfg), error=repr(e)[:120]))
                        continue
                    rows.append(dict(
                        family=fam, mechanism=feature, horizon=bar, cfg=str(cfg),
                        v1_net_sharpe=v1["sharpe_net_ann"], v2_net_sharpe=v2["sharpe_net_ann"],
                        delta=v2["sharpe_net_ann"] - v1["sharpe_net_ann"],
                        v1_cost_bps=v1["cost_per_turnover"] * 1e4,
                        v2_cost_bps=v2["cost_per_turnover"] * 1e4,
                        v1_dsr=v1["dsr"], v2_dsr=v2["dsr"],
                        v1_gate=_gate(v1), v2_gate=_gate(v2)))
                    print(f"  {fam:12} {feature:24} {bar}  {cfg}  "
                          f"netSR {v1['sharpe_net_ann']:+.2f} -> {v2['sharpe_net_ann']:+.2f}  "
                          f"gate {rows[-1]['v1_gate']}->{rows[-1]['v2_gate']}", flush=True)
    return rows


# --------------------------------------------------------------------------- #
#  F_OI / F_BOOK / F_LIQD  — re-run run_one twice (v1 constants vs v2)        #
# --------------------------------------------------------------------------- #
def _rescore_module(mod, run_all_fn, label) -> list[dict]:
    rows = []
    v1_map = {n: getattr(mod, n.replace("_V1_", "")) for n in dir(mod) if n.startswith("_V1_")}
    cur = {k: getattr(mod, k) for k in v1_map}          # v2 values (already imported)
    try:
        rows = run_all_fn(mod, v1_map, cur, label)
    finally:
        for k, v in cur.items():
            setattr(mod, k, v)
    return rows


if __name__ == "__main__":
    t0 = time.time()
    real = Ledger()
    rl = RescoreLedger(real)

    print("=== RESCORE: cross-sectional families ===", flush=True)
    xs = rescore_xsec(rl)

    changed = [r for r in xs if r.get("v1_gate") != r.get("v2_gate")]
    up = [r for r in xs if r.get("delta", 0) > 0.05]
    dn = [r for r in xs if r.get("delta", 0) < -0.05]

    lines = ["# RESCORE_305 — per-coin cost re-score (RESEARCH ROUND 2 · P0.5)", "",
             f"_generated {time.strftime('%Y-%m-%d %H:%M')}, "
             f"{len(xs)} cross-sectional configs re-priced, {round(time.time()-t0)}s_", "",
             "Only the cost number changed (per-coin `cost_model.cost_bps`, "
             "REPRICING_RULE frozen). Signals, params, universe, dates, trial "
             "count and DSR denominators are unchanged.", "",
             f"- configs whose **gate status changed**: **{len(changed)}**",
             f"- net Sharpe improved > 0.05: {len(up)}",
             f"- net Sharpe worsened > 0.05: {len(dn)}", "",
             "## Gate-status changes", ""]
    if changed:
        lines += ["| family | mechanism | horizon | cfg | v1 netSR | v2 netSR | v1 gate | v2 gate |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in changed:
            lines.append(f"| {r['family']} | {r['mechanism']} | {r['horizon']} | {r['cfg']} | "
                         f"{r['v1_net_sharpe']:+.2f} | {r['v2_net_sharpe']:+.2f} | "
                         f"{r['v1_gate']} | {r['v2_gate']} |")
    else:
        lines.append("_none — no config crossed a gate boundary._")

    lines += ["", "## Full table", "",
              "| family | mechanism | horizon | cfg | v1 cost bps | v2 cost bps | v1 netSR | v2 netSR | Δ | v1 gate | v2 gate |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(xs, key=lambda x: (x["family"], x["mechanism"], x["horizon"])):
        if "error" in r:
            lines.append(f"| {r['family']} | {r['mechanism']} | {r['horizon']} | {r['cfg']} | ERROR | | | | | | |")
            continue
        lines.append(f"| {r['family']} | {r['mechanism']} | {r['horizon']} | {r['cfg']} | "
                     f"{r['v1_cost_bps']:.2f} | {r['v2_cost_bps']:.2f} | "
                     f"{r['v1_net_sharpe']:+.2f} | {r['v2_net_sharpe']:+.2f} | {r['delta']:+.2f} | "
                     f"{r['v1_gate']} | {r['v2_gate']} |")

    out = ROOT / "lab" / "reports" / "RESCORE_305.md"
    out.write_text("\n".join(lines) + "\n")
    json.dump(xs, open(ROOT / "lab" / "reports" / "RESCORE_305.json", "w"), indent=1, default=float)
    print(f"\nwrote {out}  ({len(xs)} configs, {len(changed)} gate changes)")
