"""Apply the G0-G10 gate ladder to a study result.

Discovery gates (G3, G3.5, G4, G5, G6, G7) are evaluated on TRAIN/valid.
G9 (red team) is a separate hardened pass (red_team.py). G7 cost is baked into
every metric (net = after real costs). The single test look (G-OOS) happens
only in S7 via unseal_test.
"""
from __future__ import annotations

import numpy as np


def _th(gates_cfg, family):
    d = dict(gates_cfg["thresholds"]["default"])
    d.update(gates_cfg["thresholds"].get(family, {}))
    return d


def evaluate(study: dict, gates_cfg: dict, *, min_effect_ic: float | None = None) -> dict:
    fam = study["family"]
    th = _th(gates_cfg, fam)
    mi = min_effect_ic if min_effect_ic is not None else th["min_effect_ic"]
    v = {}

    # G3 discovery: in-sample signal must clear the surrogate noise ceiling AND
    # have an economically meaningful IC.
    g3 = (study["surrogate_obs_absic"] > study["surrogate_q99"]
          and study["surrogate_p_value"] < 0.05
          and abs(study["ic_1bar"]) >= mi)
    v["G3_discovery"] = "PASS" if g3 else "FAIL"

    # G3.5 power: enough independent OOS evidence. Proxy: >=4 usable CV folds
    # and CV IC sign-stable.
    folds = [x for x in study["cv_fold_ic"] if x == x]
    if len(folds) < 4:
        v["G3.5_power"] = "UNDERPOWERED"
    else:
        same_sign = np.mean(np.sign(folds) == np.sign(np.nanmean(folds)))
        v["G3.5_power"] = "PASS" if same_sign >= 0.66 else "UNDERPOWERED"

    # G4 OOS (purged CV): mean fold net Sharpe > threshold and CV IC mean beats mi
    g4 = (study["cv_sharpe_mean"] == study["cv_sharpe_mean"]
          and study["cv_sharpe_mean"] > th["oos_sharpe_min"]
          and abs(study["cv_ic_mean"]) >= mi * 0.7)
    v["G4_oos"] = "PASS" if g4 else "FAIL"

    # G5 multiplicity: DSR (ledger cumulative trial count) and placebo
    g5 = (study["dsr"] >= th["dsr_min"] and study["placebo_p_value"] < 0.05)
    v["G5_multiplicity"] = "PASS" if g5 else "FAIL"

    # G7 cost: net Sharpe (already after real costs) must clear the family floor
    g7 = study["sharpe_net_ann"] >= th["net_sharpe_min_at_cost_floor"]
    v["G7_cost"] = "PASS" if g7 else "FAIL"

    passed = all(x == "PASS" for x in v.values())
    underpowered = any(x == "UNDERPOWERED" for x in v.values()) and not any(x == "FAIL" for x in v.values())
    status = "PASS" if passed else ("UNDERPOWERED" if underpowered else "FAIL")

    strongest_obj = _objection(study, v, th)
    return {"gates": v, "status": status, "thresholds": th,
            "strongest_surviving_objection": strongest_obj}


def _objection(study, v, th):
    if study["sharpe_gross"] <= 0:
        return "gross edge is non-positive before costs - nothing to salvage"
    if v.get("G3_discovery") == "FAIL":
        if abs(study["ic_1bar"]) < th["min_effect_ic"]:
            return (f"|IC|={abs(study['ic_1bar']):.4f} below the {th['min_effect_ic']} "
                    f"economic-significance floor even in-sample")
        return (f"in-sample |IC|={study['surrogate_obs_absic']:.4f} does not clear the "
                f"surrogate noise ceiling q99={study['surrogate_q99']:.4f}")
    if v.get("G5_multiplicity") == "FAIL":
        if study["dsr"] < th["dsr_min"]:
            return (f"DSR={study['dsr']:.3f} < {th['dsr_min']} against {study['dsr_n_trials']} "
                    f"cumulative lab trials - indistinguishable from the best of that many tries")
        return f"placebo p={study['placebo_p_value']:.3f}: random long/short reproduces this Sharpe"
    if v.get("G4_oos") == "FAIL":
        return (f"purged-CV mean net Sharpe {study['cv_sharpe_mean']:.2f} does not hold out of sample "
                f"(fold IC mean {study['cv_ic_mean']:.4f})")
    if v.get("G7_cost") == "FAIL":
        return (f"net annualised Sharpe {study['sharpe_net_ann']:.2f} below the family floor "
                f"{th['net_sharpe_min_at_cost_floor']}; turnover {study['turnover_mean']:.2f}/bar "
                f"* cost {study['cost_per_turnover']*1e4:.1f}bps is the drag")
    if v.get("G3.5_power") == "UNDERPOWERED":
        return "too few sign-stable CV folds - sample cannot resolve the effect at this horizon"
    return "survives the discovery ladder in-sample; OOS test look (S7) not yet taken"
