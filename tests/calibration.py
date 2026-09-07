#!/usr/bin/env python3
"""S0 end-to-end calibration.

Runs the WHOLE pipeline - synth data -> causal feature -> t+1 backtest with
costs -> surrogate noise ceiling -> DSR (trial count from the ledger) -> gate
decision - on two datasets:

  control  (--edge 0)     must be classified NOT A CANDIDATE
  planted  (--edge 0.35)  must be classified A CANDIDATE

If the control passes as a candidate there is a LEAK. Stop and report; do NOT
fix it by loosening a gate.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# S0 self-tests must NEVER write to the production research ledger - they would
# pollute the cumulative trial count that DSR reads (R3). Force an isolated DB.
os.environ["CRYPTO_LAB_LEDGER_SQLITE"] = str(ROOT / "ledger" / "_selftest.sqlite")

from lab.engine.backtest import run_backtest          # noqa: E402
from lab.engine.signals import FeatureContext, build_signal  # noqa: E402
from lab.ledger import Ledger                         # noqa: E402
from lab.make_synth import generate                   # noqa: E402
from lab.stats.dsr import deflated_sharpe             # noqa: E402
from lab.stats.surrogate import surrogate_threshold   # noqa: E402

# calibration cost floor: plausible Binance-like taker + slippage. Marked as a
# calibration constant, NOT a venue spec (venue.yaml stays null until S1).
CALIB_COSTS = {"taker_fee": 0.0004, "maker_fee": 0.0002, "slippage_bps": 1.0}
GRID = [{"lookback": lb, "q": q} for lb in (12, 24, 48) for q in (0.2, 0.3, 0.4)]


def _net_sharpe_metric(feature_rank: np.ndarray, prices: np.ndarray, q: float):
    """Turn a cross-sectional rank into a t+1 long/short book and score it."""
    pos = np.zeros_like(feature_rank)
    pos[feature_rank >= (1 - q)] = 1.0
    pos[feature_rank <= q] = -1.0
    pos[~np.isfinite(feature_rank)] = 0.0
    r = run_backtest(prices, pos, CALIB_COSTS, execution_lag=1)
    return r


def evaluate_dataset(edge: float, seed: int, family: str, L: Ledger,
                     n_bars: int = 16000, n_coins: int = 12,
                     n_surr: int = 200) -> dict:
    d = generate(edge=edge, seed=seed, n_bars=n_bars, n_coins=n_coins)
    close = d["close"].T                       # (T, N)
    ctx = FeatureContext(close=close)

    hyp = L.add_hypothesis({
        "claim": f"xsec momentum rank predicts fwd return (calib edge={edge})",
        "mechanism": "calibration only - synthetic planted/absent edge",
        "math_form": "rank_t(mom_{i,t}) -> long top q, short bottom q at t+1",
        "target": "ret_fwd_1bar", "horizon": "1h", "null_hypothesis": "IC=0",
        "family": family, "lessons_reviewed": True,
    })
    exp = L.new_experiment(hyp, family, {"grid": GRID, "calib": True})

    best = None
    for cfg in GRID:
        tid = L.log_trial(exp, cfg, stage="discovery", status="running")
        sig = build_signal({"feature": "xsec_rank",
                            "params": {"lookback": cfg["lookback"]},
                            "map": "long_top_short_bottom",
                            "map_params": {"q": cfg["q"]}}, ctx)
        r = run_backtest(close, sig, CALIB_COSTS, execution_lag=1)
        L.log_metric(exp, "sharpe_net", r["sharpe_net"], trial_id=tid)
        L.log_metric(exp, "sharpe_gross", r["sharpe_gross"], trial_id=tid)
        L.log_metric(exp, "total_net", r["total_net"], trial_id=tid)
        L.mark_trial(tid, "done")
        if best is None or r["sharpe_net"] > best["r"]["sharpe_net"]:
            best = {"cfg": cfg, "r": r, "sig": sig}

    # in-sample daily pnl of the best config
    pnl = best["r"]["net_pnl"]

    # surrogate noise ceiling: block-shuffle the realised forward returns and
    # re-score the SAME positions.
    rank_feat = build_signal({"feature": "xsec_rank",
                              "params": {"lookback": best["cfg"]["lookback"]},
                              "map": "zscore"}, ctx)
    fwd = np.zeros_like(close)
    fwd[:-1] = close[1:] / close[:-1] - 1.0

    def ic_metric(feat, lab):
        f = feat.ravel()
        y = lab.ravel()
        m = np.isfinite(f) & np.isfinite(y)
        if m.sum() < 10 or f[m].std() == 0 or y[m].std() == 0:
            return 0.0
        return abs(float(np.corrcoef(f[m], y[m])[0, 1]))

    surr = surrogate_threshold(ic_metric, rank_feat, fwd, n_surr=n_surr,
                               block=48, seed=seed + 7)

    # DSR - trial count read from the ledger (R3), NOT passed in
    dsr = deflated_sharpe(pnl, family=family, ledger=L)

    L.close_experiment(exp, "done")
    return {
        "edge": edge, "family": family, "exp_id": exp,
        "best_cfg": best["cfg"],
        "sharpe_gross": best["r"]["sharpe_gross"],
        "sharpe_net": best["r"]["sharpe_net"],
        "total_net": best["r"]["total_net"],
        "total_cost": best["r"]["total_cost"],
        "ledger_trial_count": L.trial_count(family=family),
        "surrogate_obs": surr["observed"],
        "surrogate_q99": surr["q99"],
        "clears_noise": bool(surr["clears_noise"]),
        "dsr": dsr["dsr"],
        "n_trials_used_by_dsr": dsr["n_trials"],
        "trials_source": dsr["trials_source"],
    }


def classify(res: dict, gates: dict) -> dict:
    fam = res["family"]
    th = gates["thresholds"].get(fam, gates["thresholds"]["default"])
    dsr_min = th["dsr_min"]
    is_candidate = (
        res["clears_noise"]
        and res["dsr"] >= dsr_min
        and res["sharpe_net"] > 0.0
    )
    return {
        **res, "dsr_min": dsr_min, "is_candidate": bool(is_candidate),
        "verdict": "CANDIDATE" if is_candidate else "NOT A CANDIDATE",
    }


def main() -> int:
    gates = yaml.safe_load((ROOT / "config" / "gates.yaml").read_text())
    L = Ledger()

    control = classify(evaluate_dataset(0.0, seed=101, family="F_CALIB_CTRL", L=L), gates)
    planted = classify(evaluate_dataset(0.35, seed=202, family="F_CALIB_EDGE", L=L), gates)

    for tag, r in (("CONTROL  (edge 0)", control), ("PLANTED  (edge 0.35)", planted)):
        print(f"\n=== {tag} ===")
        print(f"  best config           {r['best_cfg']}")
        print(f"  sharpe gross / net    {r['sharpe_gross']:+.3f} / {r['sharpe_net']:+.3f}")
        print(f"  total net / cost      {r['total_net']:+.4f} / {r['total_cost']:.4f}")
        print(f"  surrogate obs vs q99  {r['surrogate_obs']:.4f} vs {r['surrogate_q99']:.4f}"
              f"  clears={r['clears_noise']}")
        print(f"  DSR (trials={r['n_trials_used_by_dsr']}, src={r['trials_source']})"
              f"  = {r['dsr']:.3f}  (need >= {r['dsr_min']})")
        print(f"  ledger trial count    {r['ledger_trial_count']}")
        print(f"  --> {r['verdict']}")

    ok = True
    if control["is_candidate"]:
        print("\n!!! LEAK: the zero-edge control passed as a CANDIDATE. "
              "STOP. Do not loosen a gate. Investigate the pipeline.")
        ok = False
    else:
        print("\nOK: zero-edge control is NOT a candidate (loses ~cost, "
              "fails noise ceiling).")
    if not planted["is_candidate"]:
        print("!!! The planted edge (IC=0.35) was NOT detected as a candidate. "
              "The pipeline is too conservative or broken.")
        ok = False
    else:
        print("OK: planted edge IS a candidate.")

    print("\nCALIBRATION", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
