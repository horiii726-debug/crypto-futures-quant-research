"""Monte Carlo risk simulation for a candidate strategy on a $25,000 account.

Block-bootstrap the realised daily net-PnL series (preserving autocorrelation
and fat tails), apply fixed-fraction sizing + halve-after-drawdown, and model
an explicit crash regime where cross-coin correlation -> ~1 (diversification
vanishes: on crash days the strategy's long-short net exposure is hit by a
common shock).

Outputs: P(max drawdown breach), P(ruin = equity < 50% start), equity
distribution quantiles.
"""
from __future__ import annotations

import numpy as np
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RISK = yaml.safe_load((ROOT / "config" / "risk.yaml").read_text())


def simulate(net_pnl_daily: np.ndarray, *, start_equity: float = 25_000.0,
             horizon_days: int = 365, n_paths: int = 10_000, block: int = 10,
             base_fraction: float | None = None, seed: int = 0,
             crash_prob_daily: float = 0.01, crash_shock_sigma: float = 3.0) -> dict:
    r = np.asarray(net_pnl_daily, float)
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return {"error": "pnl series too short", "n": len(r)}
    rng = np.random.default_rng(seed)
    base_fraction = base_fraction or RISK["sizing"]["base_fraction"]
    max_dd = RISK["drawdown"]["max_drawdown"]
    halve_trig = RISK["sizing"]["halve_after_drawdown"]["trigger_drawdown"]
    halve_mult = RISK["sizing"]["halve_after_drawdown"]["multiplier"]

    mu, sd = r.mean(), r.std(ddof=1)
    n = len(r)
    finals, max_dds, breached, ruined = [], [], 0, 0

    for _ in range(n_paths):
        # stationary block bootstrap of the daily strategy return
        path = np.empty(horizon_days)
        i = 0
        while i < horizon_days:
            s = rng.integers(0, n)
            L = min(block, horizon_days - i)
            seg = r[np.arange(s, s + L) % n]
            path[i:i + L] = seg
            i += L
        # explicit crash days: diversification fails -> extra common negative shock
        crash = rng.random(horizon_days) < crash_prob_daily
        path[crash] -= np.abs(rng.normal(0, crash_shock_sigma * sd, crash.sum()))

        eq = start_equity
        peak = eq
        frac = base_fraction
        halved = False
        ddmin = 0.0
        for x in path:
            # x is a "unit-gross" daily return; scale by current fraction / base
            eq *= (1.0 + x * (frac / base_fraction))
            peak = max(peak, eq)
            dd = eq / peak - 1.0
            ddmin = min(ddmin, dd)
            if not halved and dd <= -halve_trig:
                frac *= halve_mult
                halved = True
            elif halved and dd >= -0.03:
                frac = base_fraction
                halved = False
            if eq <= start_equity * 0.5:
                break
        finals.append(eq)
        max_dds.append(ddmin)
        breached += (ddmin <= -max_dd)
        ruined += (eq <= start_equity * 0.5)

    finals = np.array(finals)
    max_dds = np.array(max_dds)
    return {
        "start_equity": start_equity, "horizon_days": horizon_days, "n_paths": n_paths,
        "pnl_daily_mean": float(mu), "pnl_daily_std": float(sd),
        "p_breach_maxdd": breached / n_paths, "max_dd_limit": max_dd,
        "p_ruin_50pct": ruined / n_paths,
        "equity_final_q05": float(np.percentile(finals, 5)),
        "equity_final_q25": float(np.percentile(finals, 25)),
        "equity_final_median": float(np.percentile(finals, 50)),
        "equity_final_q75": float(np.percentile(finals, 75)),
        "equity_final_q95": float(np.percentile(finals, 95)),
        "median_max_drawdown": float(np.percentile(max_dds, 50)),
        "worst_1pct_drawdown": float(np.percentile(max_dds, 1)),
        "prob_profit": float((finals > start_equity).mean()),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    demo = rng.normal(0.0006, 0.02, 500)  # ~sharpe 0.5 daily-ish
    import json
    print(json.dumps(simulate(demo, n_paths=2000), indent=1))
