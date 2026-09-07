"""Execution layer — order split, maker-first ladder, netting (spec C + I).

This is the real source of the +1.22 maker vs +0.90 taker gap. Pure execution:
no directional decision is made here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lab.exec.cost_model import cost_bps, impact_bps


# --------------------------------------------------------------------------- #
#  C1 — order split (Almgren-Chriss: impact ~ sqrt(Q))                        #
# --------------------------------------------------------------------------- #
def split_order(notional: float, *, n_slices: int = 4, min_slice_usd: float = 20.0) -> list[float]:
    """Split Q into n equal slices. impact_total ~ impact_single/sqrt(n).
    Slices below the venue minimum are merged."""
    n = max(1, int(n_slices))
    while n > 1 and notional / n < min_slice_usd:
        n -= 1
    base = notional / n
    return [base] * n


def expected_split_saving_bps(coin: str, notional: float, n_slices: int = 4) -> dict:
    single = impact_bps(coin, notional)
    slices = split_order(notional, n_slices=n_slices)
    split_total = sum(impact_bps(coin, q) for q in slices) / len(slices)
    return {"impact_single_bps": single, "impact_split_bps": split_total,
            "saving_bps": single - split_total, "n_slices": len(slices)}


# --------------------------------------------------------------------------- #
#  C2 / I1 — maker-first ladder                                              #
# --------------------------------------------------------------------------- #
@dataclass
class LadderConfig:
    tif_sec: float = 20.0
    max_attempts: int = 3          # then taker
    tick_reprice: int = 1         # move 1 tick more aggressive each retry
    target_p_fill: float = 0.60   # I4 — measure, don't assume


@dataclass
class FillResult:
    filled: bool
    mode: str                     # 'maker' | 'taker'
    attempts: int
    predicted_cost_bps: float
    p_fill_predicted: float       # cumulative over the ladder
    realized_cost_bps: float | None = None
    latency_ms: float | None = None


def plan_ladder(coin: str, side: str, notional: float, ts, *,
                cfg: LadderConfig = LadderConfig(), sigma_1h_ratio: float = 1.0) -> FillResult:
    """Predict the outcome of a maker-first ladder for one slice.

    Each attempt is a passive limit with `tif_sec`; on miss, reprice 1 tick and
    retry, up to `max_attempts`, then cross with a taker. The per-attempt p_fill
    comes from cost_model.p_fill (queue model). Repricing 1 tick more aggressive
    raises the next attempt's fill probability (modelled as a +8% relative bump)."""
    from lab.exec.cost_model import p_fill as _pf
    p_miss = 1.0
    maker_leg = cost_bps(coin, side, notional, ts, "maker", leg="entry",
                         sigma_1h_ratio=sigma_1h_ratio)
    taker_leg = cost_bps(coin, side, notional, ts, "taker", leg="entry",
                         sigma_1h_ratio=sigma_1h_ratio)
    p_any = 0.0
    for a in range(cfg.max_attempts):
        pa = _pf(coin, leg="entry", notional=notional, sigma_1h_ratio=sigma_1h_ratio)
        pa = min(0.99, pa * (1.0 + 0.08 * a))       # each reprice is a bit more aggressive
        p_any += p_miss * pa
        p_miss *= (1.0 - pa)
    exp_cost = p_any * maker_leg["total_bps"] + p_miss * taker_leg["total_bps"]
    return FillResult(filled=True, mode=("maker" if p_any >= 0.5 else "taker"),
                      attempts=cfg.max_attempts,
                      predicted_cost_bps=float(exp_cost),
                      p_fill_predicted=float(p_any))


# --------------------------------------------------------------------------- #
#  C3 — netting: send the delta, never close-then-reopen the same side       #
# --------------------------------------------------------------------------- #
def net_orders(target: dict[str, float], current: dict[str, float],
               *, dust: float = 1e-4) -> list[dict]:
    """target / current are per-coin signed weights. Returns the minimal set of
    delta orders. Same-sign adjustments are a single resize, not a round-trip."""
    orders = []
    for coin in set(target) | set(current):
        tw = target.get(coin, 0.0)
        cw = current.get(coin, 0.0)
        dw = tw - cw
        if abs(dw) < dust:
            continue
        orders.append({"coin": coin, "delta_w": dw,
                       "side": "buy" if dw > 0 else "sell",
                       "kind": "resize" if (tw != 0 and np.sign(tw) == np.sign(cw))
                       else ("open" if cw == 0 else ("close" if tw == 0 else "flip"))})
    return orders


def turnover_of(orders: list[dict]) -> float:
    return float(sum(abs(o["delta_w"]) for o in orders))


if __name__ == "__main__":
    print("C1 split saving:", expected_split_saving_bps("SOLUSDT", 200_000, 4))
    fr = plan_ladder("ETHUSDT", "buy", 20_000, "2024-06-01T00:00:00Z")
    print(f"C2/I1 ladder ETH $20k: p_fill_predicted={fr.p_fill_predicted:.2f} "
          f"predicted_cost={fr.predicted_cost_bps:.2f} bps mode={fr.mode}")
    tgt = {"BTCUSDT": 0.3, "ETHUSDT": -0.2, "SOLUSDT": 0.1}
    cur = {"BTCUSDT": 0.1, "ETHUSDT": -0.25, "XRPUSDT": 0.15}
    print("C3 net:", net_orders(tgt, cur))
