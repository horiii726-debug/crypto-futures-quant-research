"""Entry timing inside a fixed rebalance window (spec F).

HARD CONSTRAINTS:
  - the window is fixed; it never stretches (no lookahead)
  - direction and size are never changed here
  - if the window ends unexecuted -> forced taker at the last bar
As long as these hold this is pure execution, not a new trial. The only metric
is Δ realised cost bps vs a fixed-offset baseline — never Sharpe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lab.exec.cost_model import cost_bps, half_spread_bps
from lab.exec.vol import ewma_sigma_ann


@dataclass
class TimingConfig:
    c_threshold: float = 1.0          # F1: entry band = c * sigma_hat (relative)
    cost_margin_bps: float = 0.5      # F2: need this much improvement vs window median
    spread_stress_mult: float = 2.0   # F3: skip if spread > mult * 30d median
    baseline_offset: int = 2          # bars after signal, the R9 convention


def pick_execution_bar(coin: str, side: str, notional: float, window_ts: pd.DatetimeIndex,
                       intrabar_ret: pd.Series, *, cfg: TimingConfig = TimingConfig(),
                       half_spread_now: pd.Series | None = None,
                       hs_30d_median: float | None = None) -> dict:
    """Choose which bar in `window_ts` to execute in.

    intrabar_ret : realised return of each bar in the window (for the cost-of-
                   waiting / favourable-drift check; causal — we only ever act
                   on bars up to `now`).
    Returns {exec_bar_index, mode, predicted_cost_bps, forced}.
    """
    W = len(window_ts)
    sig_ann = ewma_sigma_ann(intrabar_ret).values
    sig_bar = sig_ann / np.sqrt(365 * 24)
    hs_base = hs_30d_median if hs_30d_median is not None else half_spread_bps(coin)[0]

    costs = []
    for i in range(W):
        c = cost_bps(coin, side, notional, window_ts[i], "hybrid", leg="entry",
                     sigma_1h_ratio=(sig_bar[i] / np.nanmedian(sig_bar) if np.nanmedian(sig_bar) else 1.0))
        costs.append(c["total_bps"])
    costs = np.array(costs)
    med_cost = float(np.nanmedian(costs))

    for i in range(W - 1):
        # F3 spread-state gate
        hs_i = (half_spread_now.iloc[i] if half_spread_now is not None else hs_base)
        if hs_i > cfg.spread_stress_mult * hs_base:
            continue
        # F1 adaptive threshold: only "jump the queue" with a taker-ish fill when
        # the favourable intrabar move exceeds c*sigma; otherwise stay patient
        drift = -np.sign(1 if side == "buy" else -1) * np.nansum(intrabar_ret.values[:i + 1])
        if drift < cfg.c_threshold * sig_bar[i]:
            # F2 cost-aware gate
            if costs[i] < med_cost - cfg.cost_margin_bps * 1e0:
                return {"exec_bar_index": i, "mode": "hybrid",
                        "predicted_cost_bps": float(costs[i]), "forced": False}
    # forced execution at the last bar
    cF = cost_bps(coin, side, notional, window_ts[-1], "taker", leg="entry")
    return {"exec_bar_index": W - 1, "mode": "taker",
            "predicted_cost_bps": float(cF["total_bps"]), "forced": True}


def evaluate_overlay(rebalances: list[dict], *, cfg: TimingConfig = TimingConfig()) -> dict:
    """rebalances: list of {coin, side, notional, window_ts, intrabar_ret,
    baseline_realized_bps}. Returns Δ realised cost bps (overlay − baseline)."""
    d = []
    n_forced = 0
    for rb in rebalances:
        res = pick_execution_bar(rb["coin"], rb["side"], rb["notional"],
                                 rb["window_ts"], rb["intrabar_ret"], cfg=cfg)
        n_forced += int(res["forced"])
        # realised cost of executing at res["exec_bar_index"] vs the baseline offset:
        # the extra adverse move between the baseline bar and the chosen bar
        r = rb["intrabar_ret"].values
        b0 = min(cfg.baseline_offset, len(r) - 1)
        bi = res["exec_bar_index"]
        sgn = 1.0 if rb["side"] == "buy" else -1.0
        base_move = sgn * np.nansum(r[1:b0 + 1]) * 1e4
        ovl_move = sgn * np.nansum(r[1:bi + 1]) * 1e4
        d.append(ovl_move - base_move)          # + = overlay paid a worse price
    d = np.array(d)
    return {"n": len(d), "n_forced": n_forced,
            "delta_realized_cost_bps_mean": float(np.nanmean(d)),
            "delta_realized_cost_bps_se": float(np.nanstd(d) / np.sqrt(max(len(d), 1))),
            "note": "negative = overlay executes at better prices for the same trade"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rbs = []
    for _ in range(2000):
        r = pd.Series(rng.normal(0, 0.0006, 12))
        ts = pd.date_range("2024-06-01", periods=12, freq="5min", tz="UTC")
        rbs.append({"coin": "SOLUSDT", "side": rng.choice(["buy", "sell"]),
                    "notional": 20_000, "window_ts": ts, "intrabar_ret": r})
    print(evaluate_overlay(rbs))
