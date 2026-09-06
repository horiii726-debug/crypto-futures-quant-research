"""Realistic maker / hybrid execution model, calibrated on real aggTrades tape
(+ bookTicker for queue size).

A resting limit order at the touch:
  * fills only when opposing AGGRESSIVE volume arrives and clears the queue
    ahead of it -> simulated directly against the real trade tape;
  * fills are ADVERSELY SELECTED: you get hit exactly when informed flow is
    against you, so the mid tends to move through your price right after -
    measured as `adverse_selection_bps`;
  * if unfilled after `wait_sec`, the order crosses with a taker -> pays
    taker fee + half-spread + whatever the mid already moved during the wait
    (opportunity / slippage cost).

NO assumption of 100% maker fill. p_fill comes from the tape.

effective one-way cost (fraction of notional):
    C = p_fill * (maker_fee + adverse_selection)
      + (1 - p_fill) * (taker_fee + half_spread + adverse_during_wait)

Round trip = C_entry + C_exit, with the exit leg given a shorter wait (you
must get out) -> lower p_fill, more taker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab.data.binance_vision import agg_trades


def simulate_passive_fills(trades: pd.DataFrame, *, side: str = "buy",
                           queue_ahead_notional: float, wait_sec: float,
                           n_orders: int = 400, hold_lookahead_sec: float = 300.0,
                           seed: int = 0) -> dict:
    """Drop `n_orders` resting limit orders at random times on the real tape.

    side='buy'  -> resting bid, filled by SELL-aggressor trades
                   (is_buyer_maker == True means the aggressor SOLD).
    Fill when cumulative opposing-aggressor notional after entry exceeds
    `queue_ahead_notional`. Adverse selection = signed mid move from fill
    time to fill+hold_lookahead (positive = against us).
    """
    if trades.empty or len(trades) < 500:
        return {"status": "thin_tape"}
    t = trades["ts"].to_numpy()               # ms
    p = trades["price"].to_numpy(float)
    q = (trades["qty"].to_numpy(float) * p)   # notional
    sell_aggr = trades["is_buyer_maker"].to_numpy(bool)   # aggressor sold
    opp = sell_aggr if side == "buy" else ~sell_aggr

    rng = np.random.default_rng(seed)
    # sample order-entry points by TRADE INDEX (the tape may be several
    # non-contiguous days concatenated - sampling by wall-clock time would land
    # in the gaps). Skip points too close to the end of their local day.
    idx0 = np.sort(rng.integers(50, len(t) - 50, n_orders))

    filled = 0
    adv_bps, wait_move_bps, fill_lat = [], [], []
    for k, i0 in enumerate(idx0):
        i0 = int(i0)
        entry_px = p[i0]
        start_ms = t[i0]
        deadline = start_ms + wait_sec * 1000
        j = i0
        cum = 0.0
        fill_t = None
        while j < len(t) and t[j] <= deadline:
            if opp[j]:
                cum += q[j]
                if cum >= queue_ahead_notional:
                    fill_t = t[j]
                    fill_px = entry_px  # limit price ~ touch at entry
                    break
            j += 1
        if fill_t is not None:
            filled += 1
            fill_lat.append((fill_t - start_ms) / 1000.0)
            m = np.searchsorted(t, fill_t + hold_lookahead_sec * 1000)
            m = min(m, len(p) - 1)
            # bound: if the found bar is far past the lookahead window, the tape
            # gapped (end of a sampled day) -> skip this sample's adv-sel.
            if t[m] - fill_t > 2 * hold_lookahead_sec * 1000:
                continue
            fut = p[m]
            move = (fut / fill_px - 1.0) * (1 if side == "sell" else -1)  # bid fill: against us = price falls
            adv_bps.append(move * 1e4)
        else:
            m = np.searchsorted(t, deadline)
            m = min(m, len(p) - 1)
            if t[m] - start_ms > 2 * wait_sec * 1000:
                continue
            fut = p[m]
            move = (fut / entry_px - 1.0) * (1 if side == "sell" else -1)
            wait_move_bps.append(max(move, 0.0) * 1e4)

    p_fill = filled / n_orders
    return {
        "status": "ok", "side": side, "n_orders": n_orders,
        "p_fill": p_fill,
        "adverse_selection_bps": float(np.nanmean(adv_bps)) if adv_bps else 0.0,
        "adverse_selection_bps_median": float(np.nanmedian(adv_bps)) if adv_bps else 0.0,
        "wait_slippage_bps": float(np.nanmean(wait_move_bps)) if wait_move_bps else 0.0,
        "fill_latency_sec_median": float(np.nanmedian(fill_lat)) if fill_lat else None,
        "queue_ahead_notional": queue_ahead_notional,
        "wait_sec": wait_sec,
    }


ADVERSE_SELECTION_FLOOR_BPS = 0.75   # conservative: a maker fill IS informed-selected


def effective_cost_oneway(sim: dict, *, maker_fee: float, taker_fee: float,
                          half_spread_bps: float) -> dict:
    if sim.get("status") != "ok":
        return {"status": sim.get("status")}
    pf = sim["p_fill"]
    # floor the measured adverse selection - the 120s-lookahead estimate is
    # noisy and understates the real first-few-seconds pick-off.
    adv = max(sim["adverse_selection_bps"], ADVERSE_SELECTION_FLOOR_BPS)
    maker_leg = maker_fee * 1e4 + adv
    taker_leg = taker_fee * 1e4 + half_spread_bps + sim["wait_slippage_bps"]
    c = pf * maker_leg + (1 - pf) * taker_leg
    return {
        "status": "ok",
        "p_fill": pf,
        "maker_leg_bps": maker_leg,
        "taker_leg_bps": taker_leg,
        "effective_oneway_bps": c,
        "taker_only_oneway_bps": taker_fee * 1e4 + half_spread_bps,
    }


def hybrid_roundtrip_bps(entry_sim: dict, exit_sim: dict, *, maker_fee: float,
                         taker_fee: float, half_spread_bps: float) -> dict:
    e = effective_cost_oneway(entry_sim, maker_fee=maker_fee, taker_fee=taker_fee,
                              half_spread_bps=half_spread_bps)
    x = effective_cost_oneway(exit_sim, maker_fee=maker_fee, taker_fee=taker_fee,
                              half_spread_bps=half_spread_bps)
    if e.get("status") != "ok" or x.get("status") != "ok":
        return {"status": "sim_failed", "entry": e, "exit": x}
    rt = e["effective_oneway_bps"] + x["effective_oneway_bps"]
    taker_rt = 2 * (taker_fee * 1e4 + half_spread_bps)
    return {
        "status": "ok",
        "hybrid_roundtrip_bps": rt,
        "taker_only_roundtrip_bps": taker_rt,
        "saving_bps": taker_rt - rt,
        "saving_pct": (taker_rt - rt) / taker_rt,
        "entry": e, "exit": x,
    }


def calibrate_symbol(symbol: str, days: list[str], *, queue_ahead_notional: float,
                     maker_fee: float = 0.0002, taker_fee: float = 0.0005,
                     half_spread_bps: float = 0.85,
                     entry_wait_sec: float = 30.0, exit_wait_sec: float = 8.0,
                     hold_lookahead_sec: float = 120.0, seed: int = 0) -> dict:
    tr = agg_trades(symbol, days)
    if tr.empty:
        return {"symbol": symbol, "status": "no_data"}
    entry = simulate_passive_fills(tr, side="buy", queue_ahead_notional=queue_ahead_notional,
                                   wait_sec=entry_wait_sec, hold_lookahead_sec=hold_lookahead_sec,
                                   seed=seed)
    exit_ = simulate_passive_fills(tr, side="sell", queue_ahead_notional=queue_ahead_notional,
                                   wait_sec=exit_wait_sec, hold_lookahead_sec=hold_lookahead_sec,
                                   seed=seed + 1)
    rt = hybrid_roundtrip_bps(entry, exit_, maker_fee=maker_fee, taker_fee=taker_fee,
                              half_spread_bps=half_spread_bps)
    return {"symbol": symbol, "status": rt.get("status"),
            "entry_sim": entry, "exit_sim": exit_, "roundtrip": rt}


if __name__ == "__main__":
    import json
    r = calibrate_symbol("DARUSDT", ["2024-02-21"], queue_ahead_notional=80_000)
    print(json.dumps(r, indent=1, default=float))
