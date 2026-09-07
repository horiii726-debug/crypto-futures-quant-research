"""Grid runner: enumerate a PRE-REGISTERED parameter grid and log every fit to
the ledger BEFORE its result is known.

R3  Every fit becomes a trial row. dsr.py reads the count from those rows.
R4  Discovery and validation are walled off. The grid is fixed up front; you
    cannot add a point after seeing results without registering a new
    hypothesis (parent_id link).

experiment-runner is the only role that calls run_grid with stage="test".
"""
from __future__ import annotations

import itertools
import json

import numpy as np

from lab.ledger import Ledger


def expand_grid(grid: dict) -> list[dict]:
    keys = sorted(grid.keys())
    combos = itertools.product(*[grid[k] for k in keys])
    return [dict(zip(keys, c)) for c in combos]


def run_grid(hyp_id: str, family: str, grid: dict, evaluate, *,
             stage: str = "discovery", ledger: Ledger | None = None,
             data_partition: str = "train", notes: str = "") -> dict:
    """
    evaluate(config, partition) -> dict of {metric_name: float}
        called AFTER the trial row is written. It must not change the grid.
    """
    L = ledger or Ledger()
    configs = expand_grid(grid)
    spec = {
        "grid": {k: list(v) for k, v in grid.items()},
        "n_points": len(configs),
        "stage": stage,
        "data_partition": data_partition,
        "notes": notes,
        "pre_registered": True,
    }
    exp_id = L.new_experiment(hyp_id, family, spec, stage=stage)

    # Pre-register EVERY trial before any evaluation. Result unknown at this point.
    trial_ids = []
    for cfg in configs:
        tid = L.log_trial(exp_id, cfg, stage=stage, status="planned",
                          counts_as_trial=True)
        trial_ids.append(tid)

    results = []
    for cfg, tid in zip(configs, trial_ids):
        L.mark_trial(tid, "running")
        try:
            metrics = evaluate(cfg, data_partition)
            for name, val in metrics.items():
                if isinstance(val, (int, float, np.floating, np.integer)):
                    L.log_metric(exp_id, name, float(val), trial_id=tid,
                                 context={"config": cfg})
                else:
                    L.log_metric(exp_id, name, None, trial_id=tid,
                                 text_value=str(val), context={"config": cfg})
            L.mark_trial(tid, "done")
            results.append({"config": cfg, "trial_id": tid, **metrics})
        except Exception as e:  # a failed fit still counts as a trial (R3)
            L.log_metric(exp_id, "error", None, trial_id=tid, text_value=repr(e))
            L.mark_trial(tid, "failed")
            results.append({"config": cfg, "trial_id": tid, "error": repr(e)})

    L.close_experiment(exp_id, "done")
    return {
        "exp_id": exp_id,
        "family": family,
        "stage": stage,
        "n_trials": len(configs),
        "trial_count_in_ledger": L.trial_count(family=family),
        "results": results,
    }


if __name__ == "__main__":
    import os
    os.environ.setdefault("CRYPTO_LAB_LEDGER_SQLITE", "./ledger/lab.sqlite")
    L = Ledger()
    hyp = L.add_hypothesis({
        "claim": "demo", "mechanism": "demo", "math_form": "f(x)=x",
        "target": "ret_1", "horizon": "1h", "null_hypothesis": "IC=0",
        "family": "F_DEMO", "lessons_reviewed": True,
    })

    def ev(cfg, part):
        rng = np.random.default_rng(hash(json.dumps(cfg, sort_keys=True)) % (2**32))
        return {"sharpe_oos": float(rng.normal(0, 0.3))}

    out = run_grid(hyp, "F_DEMO", {"lookback": [10, 20, 40], "q": [0.2, 0.3]}, ev)
    print("exp", out["exp_id"], "trials", out["n_trials"],
          "ledger count(F_DEMO)=", out["trial_count_in_ledger"])
