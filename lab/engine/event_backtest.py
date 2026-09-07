"""Event-driven backtest: signal -> discrete entry event -> hold -> barrier exit.

This is the architecture that fixes the COST_BOUND problem found in Rounds 1-3.
A continuous cross-sectional rebalance pays cost every bar; here a trade pays
exactly one round trip and holds until a volatility-scaled barrier is touched.

Constraints enforced (user spec):
  * min_hold_bars  — no 3/5-minute in-and-out
  * cooldown_bars  — a coin cannot re-enter immediately after an exit
  * max_concurrent — portfolio-level position cap
  * SL/TP are volatility-scaled and wick-floored (see barriers.py)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lab.engine.barriers import (BarrierConfig, average_uniqueness, label_events,
                                 sigma_blend)


@dataclass
class EventConfig:
    entry_z: float = 1.5           # signal z-score threshold to open
    cooldown_bars: int = 24        # bars a coin must wait after an exit
    max_concurrent: int = 8        # simultaneous open positions
    max_per_coin: int = 1
    side: str = "both"             # 'both' | 'long' | 'short'
    execution_lag: int = 1         # R9: decide t, fill t+1
    barrier: BarrierConfig = field(default_factory=BarrierConfig)


def signal_to_events(sig: pd.DataFrame, cfg: EventConfig) -> pd.DataFrame:
    """Cross a z-score threshold -> one entry event. Not a continuous weight."""
    s = sig.copy()
    idx = s.index
    long_ev = s >= cfg.entry_z
    short_ev = s <= -cfg.entry_z
    if cfg.side == "long":
        short_ev &= False
    elif cfg.side == "short":
        long_ev &= False
    # only fire on a NEW crossing (avoid re-firing every bar while above)
    long_new = long_ev & ~long_ev.shift(1, fill_value=False)
    short_new = short_ev & ~short_ev.shift(1, fill_value=False)
    rows = []
    for coin in s.columns:
        for t in idx[long_new[coin].fillna(False)]:
            rows.append({"ts": t, "coin": coin, "side": 1.0})
        for t in idx[short_new[coin].fillna(False)]:
            rows.append({"ts": t, "coin": coin, "side": -1.0})
    if not rows:
        return pd.DataFrame(columns=["ts", "coin", "side"])
    ev = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    # R9 execution lag: shift the fill one bar forward
    if cfg.execution_lag:
        pos = {t: i for i, t in enumerate(idx)}
        ev["ts"] = [idx[min(pos[t] + cfg.execution_lag, len(idx) - 1)] for t in ev["ts"]]
    return ev


def _apply_portfolio_constraints(lab: pd.DataFrame, cfg: EventConfig) -> pd.DataFrame:
    """Sequentially accept trades subject to cooldown / concurrency / per-coin caps."""
    lab = lab.sort_values("i_entry").reset_index(drop=True)
    open_until: dict[str, int] = {}     # coin -> bar index it is free again
    open_slots: list[int] = []          # exit indices of currently open trades
    keep = []
    for r in lab.itertuples():
        # release finished slots
        open_slots = [x for x in open_slots if x > r.i_entry]
        if len(open_slots) >= cfg.max_concurrent:
            continue
        if open_until.get(r.coin, -1) > r.i_entry:
            continue
        keep.append(r.Index)
        open_slots.append(r.i_exit)
        open_until[r.coin] = r.i_exit + cfg.cooldown_bars
    return lab.loc[keep].reset_index(drop=True)


def run_event_backtest(px: dict, sig: pd.DataFrame, cfg: EventConfig,
                       cost_frac: pd.Series | float = 0.0015,
                       sigma: pd.DataFrame | None = None) -> dict:
    """px: dict of open/high/low/close frames. sig: (time x coin) z-scores.
    cost_frac: per-coin ROUND-TRIP cost fraction (or a scalar)."""
    if sigma is None:
        sigma = sigma_blend(px["open"], px["high"], px["low"], px["close"],
                            cfg.barrier.sigma_window)
    ev = signal_to_events(sig, cfg)
    if ev.empty:
        return {"trades": pd.DataFrame(), "n_trades": 0, "status": "no_events"}
    lab = label_events(px, ev, cfg.barrier, sigma=sigma)
    if lab.empty:
        return {"trades": pd.DataFrame(), "n_trades": 0, "status": "no_labels"}
    lab = lab[lab["hold_bars"] >= cfg.barrier.min_hold_bars]
    # a delisted coin can produce a NaN exit price on the time-barrier path
    lab = lab[np.isfinite(lab["ret_gross"])]
    if lab.empty:
        return {"trades": pd.DataFrame(), "n_trades": 0, "status": "all_below_min_hold"}
    lab = _apply_portfolio_constraints(lab, cfg)
    if lab.empty:
        return {"trades": pd.DataFrame(), "n_trades": 0, "status": "constrained_out"}

    if isinstance(cost_frac, pd.Series):
        c = lab["coin"].map(cost_frac).fillna(float(cost_frac.median()))
    else:
        c = pd.Series(float(cost_frac), index=lab.index)
    lab["cost"] = c.values
    lab["ret_net"] = lab["ret_gross"] - lab["cost"]
    lab["w_uniq"] = average_uniqueness(lab, len(px["close"].index))
    return {"trades": lab, "n_trades": len(lab), "status": "ok"}


# --------------------------------------------------------------------------- #
#  performance of an equal-risk trade book                                    #
# --------------------------------------------------------------------------- #
def trade_stats(trades: pd.DataFrame, bar_hours: float, capital_slots: int = 8) -> dict:
    """Convert a trade list into portfolio statistics. Each trade takes 1/N of
    capital; returns are compounded on the calendar-time equity curve."""
    if trades.empty:
        return {"n_trades": 0}
    r = trades["ret_net"].to_numpy()
    g = trades["ret_gross"].to_numpy()
    hold = trades["hold_bars"].to_numpy()
    win = r > 0
    span_bars = trades["i_exit"].max() - trades["i_entry"].min() + 1
    years = span_bars * bar_hours / (365 * 24)
    # calendar-time Sharpe: allocate 1/capital_slots per trade, aggregate per bar
    per_trade = r / capital_slots
    total = float(np.prod(1 + per_trade) - 1)
    trades_per_year = len(r) / max(years, 1e-9)
    mu = per_trade.mean() * trades_per_year
    sd = per_trade.std(ddof=1) * np.sqrt(trades_per_year) if len(r) > 2 else np.nan
    eq = np.cumprod(1 + per_trade)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min()) if len(eq) else 0.0
    payoff = (r[win].mean() / -r[~win].mean()) if win.any() and (~win).any() else np.nan
    return {
        "n_trades": int(len(r)),
        "years": round(years, 2),
        "trades_per_year": round(trades_per_year, 1),
        "win_rate": round(float(win.mean()), 4),
        "avg_ret_gross_bps": round(float(g.mean()) * 1e4, 2),
        "avg_ret_net_bps": round(float(r.mean()) * 1e4, 2),
        "avg_cost_bps": round(float(trades["cost"].mean()) * 1e4, 2),
        "payoff_ratio": round(float(payoff), 3) if np.isfinite(payoff) else None,
        "profit_factor": round(float(r[win].sum() / -r[~win].sum()), 3) if (~win).any() and r[~win].sum() < 0 else None,
        "expectancy_bps": round(float(r.mean()) * 1e4, 2),
        "sharpe_ann": round(float(mu / sd), 3) if sd and np.isfinite(sd) and sd > 0 else None,
        "total_return": round(total, 4),
        "max_drawdown": round(dd, 4),
        "avg_hold_bars": round(float(hold.mean()), 1),
        "avg_hold_hours": round(float(hold.mean()) * bar_hours, 1),
        "tp_hits": int((trades["barrier"] == 1).sum()),
        "sl_hits": int((trades["barrier"] == -1).sum()),
        "time_exits": int((trades["barrier"] == 0).sum()),
    }
