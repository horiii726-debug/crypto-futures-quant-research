"""S3-S6 campaign: run a family's cross-sectional tournament with autonomous
escalation, then hand survivors to S7 (single sealed test look).

For each family the theorist queue is: try each admissible feature; for each,
a pre-registered param grid; every fit is a ledger trial BEFORE its result.
On failure climb the ladder (L1 wider sample already fixed by data; L2 change
horizon; L3 re-spec params -> already the grid; L4 next feature/mechanism;
L5 change target/mapping; L6 write a BOUND).
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lab.ledger import Ledger
from lab.research import gates as gatemod
from lab.research.panel import Panel, REGISTRY
from lab.research.study import run_study

ROOT = Path(__file__).resolve().parents[2]
GATES = yaml.safe_load((ROOT / "config" / "gates.yaml").read_text())
BUDGETS = yaml.safe_load((ROOT / "config" / "budgets.yaml").read_text())

# Pre-registered hypotheses per family. Lookback params are expressed in DAYS
# and converted to panel bars per research frequency (so the same hypothesis is
# tested coherently at 1d and 3d). Params NOT in LOOKBACK_PARAMS are passed
# through unchanged (e.g. n_pc, lead).
LOOKBACK_PARAMS = {"lookback", "skip", "fast", "slow", "corr_win", "mom_lb",
                   "n", "vol_n", "mom_n"}

HYPOTHESES = {
    "F_XSEC": [
        ("xsec_momentum", dict(lookback=[7, 14, 30, 60], skip=[1]), "decile",
         "under-reaction to coin-level trends; winners keep winning over weeks"),
        ("xsec_st_reversal", dict(lookback=[1, 3, 7]), "decile",
         "overreaction reverses over days"),
        ("xsec_low_vol", dict(lookback=[14, 30, 60]), "decile",
         "leverage-constrained buyers overpay for volatile coins"),
        ("xsec_idio_vol", dict(lookback=[21, 45]), "decile",
         "idiosyncratic-vol is priced negatively (Ang et al)"),
        ("xsec_amihud", dict(lookback=[14, 30, 60]), "decile",
         "illiquidity premium: less liquid coins earn more"),
        ("xsec_max", dict(lookback=[21, 45]), "decile",
         "lottery demand overprices recent big-up coins"),
        ("xsec_skew", dict(lookback=[21, 45]), "decile",
         "positive-skew coins overpriced"),
        ("xsec_52w_high", dict(lookback=[45, 90, 150]), "decile",
         "anchoring to the trailing high delays adjustment"),
        ("xsec_volume_trend", dict(fast=[7, 14], slow=[30, 60]), "decile",
         "rising relative volume precedes continued strength"),
        ("xsec_turnover_reversal", dict(lookback=[7, 14]), "decile",
         "high-turnover coins are crowded and revert"),
        ("xsec_residual_momentum", dict(lookback=[21, 45], skip=[1]), "decile",
         "momentum in market-model residuals, lower turnover"),
    ],
    "F_FUND": [
        ("fund_carry", dict(), "decile",
         "positive funding = crowded longs; short high-funding, long low"),
        ("fund_momentum", dict(lookback=[2, 7, 14]), "decile",
         "funding regimes persist over days"),
        ("fund_zscore", dict(lookback=[15, 30, 60]), "decile",
         "extreme funding vs own history mean-reverts"),
        ("fund_accel", dict(lookback=[1, 2]), "decile",
         "the change in funding, not the level, carries the signal"),
    ],
    "F_FLOW": [
        ("flow_imb_momentum", dict(lookback=[1, 3, 7]), "decile",
         "aggressor imbalance persists"),
        ("flow_imb_reversal", dict(lookback=[1, 2]), "decile",
         "extreme aggressor imbalance exhausts and reverses"),
        ("flow_signed_vol", dict(lookback=[2, 3, 7]), "decile",
         "cumulative signed dollar volume predicts continuation"),
        ("flow_kyle_lambda", dict(lookback=[7, 14, 30]), "decile",
         "low price-impact (liquid) coins outperform on a risk basis"),
    ],
    "F_XCOIN": [
        ("xcoin_btc_leadlag", dict(lead=[1]), "zscore",
         "BTC/ETH returns lead alt returns"),
        ("xcoin_peer_return", dict(lookback=[1, 3, 7]), "decile",
         "relative strength vs peers continues"),
        ("xcoin_spillover_momentum", dict(lookback=[7, 14], corr_win=[30]), "decile",
         "correlation-weighted peer momentum spills over"),
        ("xcoin_pca_residual", dict(lookback=[21, 45], n_pc=[3]), "decile",
         "residual after common factors mean-reverts cross-sectionally"),
        ("xcoin_dispersion_switch", dict(lookback=[14], mom_lb=[14, 30]), "zscore",
         "momentum in high-dispersion regimes, reversal in low"),
    ],
    # ---- P2 expansion divisions ----
    "F_DIR": [
        ("dir_rsi", dict(n=[7, 14, 30]), "decile", "RSI extremes fade (overreaction)"),
        ("dir_macd", dict(fast=[8, 12], slow=[21, 34]), "decile", "MACD histogram trend"),
        ("dir_ma_cross", dict(fast=[5, 10], slow=[30, 60]), "decile", "MA-cross trend"),
        ("dir_bollinger_z", dict(n=[14, 30]), "decile", "Bollinger z mean-reversion"),
        ("dir_donchian_pos", dict(n=[14, 30]), "decile", "position in the N-day channel"),
    ],
    "F_TRANSFORM": [
        ("transform_hurst", dict(n=[30, 60, 120]), "zscore",
         "Hurst > 0.5 trend / < 0.5 revert -> tilt"),
        ("transform_spectral_entropy", dict(n=[32, 64]), "decile",
         "low spectral entropy = exploitable structure"),
    ],
    "F_DEP": [
        ("dep_variance_ratio", dict(n=[30, 60]), "zscore",
         "VR < 1 mean-revert, > 1 trend"),
        ("dep_autocorr1", dict(n=[21, 45]), "zscore",
         "return AR(1): positive -> momentum, negative -> reversal"),
    ],
    "F_PATH": [
        ("path_run_length", dict(), "decile", "long directional runs revert"),
        ("path_mfe_mae", dict(n=[10, 20, 45]), "decile",
         "favourable-vs-adverse excursion ratio = trend quality"),
    ],
    "F_MULTI": [
        ("multi_tf_agreement", dict(), "decile",
         "count of timeframes agreeing on trend direction"),
        ("multi_fast_slow", dict(fast=[3, 7], slow=[21, 45]), "decile",
         "fast minus slow momentum (acceleration)"),
    ],
    "F_VOL": [
        ("vol_yang_zhang", dict(n=[14, 30]), "decile",
         "low realised (Yang-Zhang) vol tilt"),
        ("vol_semivar_skew", dict(n=[14, 30]), "decile",
         "negative signed-jump variation -> future underperformance"),
    ],
    "F_LIQ": [
        ("liq_kyle_bar", dict(n=[14, 30, 60]), "decile",
         "low bar-level price impact (liquid) coins outperform risk-adjusted"),
    ],
    "F_STATE": [
        ("state_vol_regime_mom", dict(vol_n=[14, 30], mom_n=[14, 30]), "zscore",
         "momentum in calm regimes, reversal in stressed regimes"),
    ],
}


def _grid(g: dict, bar_days: float):
    if not g:
        return [dict()]
    keys = list(g)
    combos = [dict(zip(keys, v)) for v in itertools.product(*[g[k] for k in keys])]
    out = []
    for c in combos:
        cc = {}
        for k, v in c.items():
            if k in LOOKBACK_PARAMS:
                cc[k] = max(2 if k != "skip" else 1, int(round(v / bar_days)))
            else:
                cc[k] = v
        out.append(cc)
    # de-dup after rounding
    seen, uniq = set(), []
    for c in out:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key); uniq.append(c)
    return uniq


def run_family(family: str, panel_train: Panel, horizons_bars: dict[str, int],
               ledger: Ledger, q: float = 0.2, max_configs: int | None = None) -> dict:
    budget = BUDGETS["families"].get(family, {}).get("max_configs", 400)
    max_configs = max_configs or budget
    results, verdicts = [], []
    spent = 0
    frozen = False

    for feature, grid, mode, mechanism in HYPOTHESES[family]:
        if frozen:
            break
        null_h = "cross-sectional rank IC = 0 ; net Sharpe <= 0 after real costs"
        hyp = ledger.add_hypothesis({
            "claim": f"{feature}: {mechanism}", "mechanism": mechanism,
            "math_form": feature, "target": "xsec_fwd_return", "horizon": "multi",
            "null_hypothesis": null_h, "family": family, "lessons_reviewed": True,
        })
        bar_days = panel_train.bar_hours / 24.0
        configs = _grid(grid, bar_days)
        # L2 (change horizon) is part of the pre-registered space:
        full = [(c, hname, hb) for c in configs for hname, hb in horizons_bars.items()]
        spec = {"feature": feature, "grid": grid, "horizons": list(horizons_bars),
                "mode": mode, "q": q, "n_points": len(full), "pre_registered": True}
        exp = ledger.new_experiment(hyp, family, spec, stage="discovery")

        best = None
        for (cfg, hname, hb) in full:
            if spent >= max_configs:
                frozen = True
                break
            tid = ledger.log_trial(exp, {**cfg, "horizon": hname}, stage="discovery",
                                   status="planned")
            spent += 1
            ledger.mark_trial(tid, "running")
            try:
                st = run_study(f"{feature}|{hname}|{cfg}", feature, cfg, hb,
                               panel_train, family=family, hyp_id=hyp, exp_id=exp,
                               ledger=ledger, q=q, mode=mode,
                               n_placebo=150, n_surrogate=150,
                               seed=abs(hash((feature, hname, str(cfg)))) % (2**31))
                ledger.mark_trial(tid, "done")
            except Exception as e:  # noqa
                ledger.log_metric(exp, "error", None, trial_id=tid, text_value=repr(e))
                ledger.mark_trial(tid, "failed")
                continue
            g = gatemod.evaluate(st, GATES)
            rec = {**{k: st[k] for k in st if k not in ("net_pnl_daily",)},
                   "gate_status": g["status"], "gates": g["gates"],
                   "objection": g["strongest_surviving_objection"],
                   "hyp_id": hyp, "exp_id": exp}
            results.append(rec)
            score = (g["status"] == "PASS", st["dsr"], st["sharpe_net_ann"])
            if best is None or score > best["_score"]:
                best = {**rec, "_score": score, "_study": st}

        # verdict for this mechanism
        if best is None:
            continue
        status = "PASS" if best["gate_status"] == "PASS" else (
            "UNDERPOWERED" if best["gate_status"] == "UNDERPOWERED" else "FAIL")
        ledger.add_verdict(best["hyp_id"], "final", status, "validator",
                           [best["exp_id"]],
                           rationale=(f"best of {len(full)} pre-registered configs: "
                                      f"{best['feature']} dsr={best['dsr']:.3f} "
                                      f"net_sharpe_ann={best['sharpe_net_ann']:.2f} "
                                      f"ic={best['ic_1bar']:.4f} placebo_p={best['placebo_p_value']:.3f}"),
                           strongest_surviving_objection=best["objection"])
        verdicts.append({"feature": feature, "status": status,
                         "best": {k: best[k] for k in ("feature", "params", "horizon_bars",
                                                       "ic_1bar", "sharpe_net_ann", "dsr",
                                                       "placebo_p_value", "surrogate_p_value",
                                                       "cv_sharpe_mean", "objection")}})
        if status != "PASS":
            ledger.add_lesson(
                root_cause=best["objection"],
                lesson=f"{feature} ({mechanism}) did not clear the ladder on train/valid "
                       f"for {family}; best net Sharpe(ann)={best['sharpe_net_ann']:.2f}, "
                       f"DSR={best['dsr']:.3f} vs {best['dsr_n_trials']} cumulative trials.",
                family=family, subject_id=best["hyp_id"], ladder_level="L4")

    survivors = [v for v in verdicts if v["status"] == "PASS"]
    return {"family": family, "n_configs": spent, "frozen": frozen,
            "verdicts": verdicts, "survivors": survivors,
            "results": results}


def write_bound_if_empty(family: str, all_results: list, ledger: Ledger) -> str | None:
    ex_ids = sorted({r["exp_id"] for r in all_results})
    bid = ledger.add_bound(family, "cross-sectional long/short, real Binance USDT-M costs",
                           _bound_stmt(family, all_results), ex_ids)
    return bid


def _bound_stmt(family, results) -> str:
    if not results:
        return f"{family}: no evaluable config."
    best_ic = max((abs(r["ic_1bar"]) for r in results), default=0)
    best_sr = max((r["sharpe_net_ann"] for r in results), default=0)
    best_dsr = max((r["dsr"] for r in results), default=0)
    n = results[0]["dsr_n_trials"] if results else 0
    return (f"{family}: across {len(results)} pre-registered cross-sectional configs on "
            f"2y of real Binance USDT-M perp data with measured costs, no mechanism cleared "
            f"the ladder. Best in-sample |IC(1-bar)| <= {best_ic:.4f}; best net annualised "
            f"Sharpe <= {best_sr:.2f}; best DSR <= {best_dsr:.3f} against {n}+ cumulative lab "
            f"trials. Cross-sectional predictability at feasible horizons is bounded below the "
            f"cost floor for this universe and period.")
