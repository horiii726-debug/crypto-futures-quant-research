"""Portfolio-level, cost-aware backtest across coins.

R7  Features must be live-computable (enforced elsewhere).
R9  A signal formed on bar t is executed at t+1. Never at t.

Inputs
------
prices   : (T, N) close price per bar per coin
signal   : (T, N) target position in [-1, 1] per bar per coin, formed using
           information up to and including bar t
costs    : dict from venue.yaml -> {taker_fee, maker_fee, slippage_bps, ...}
           If any required field is None the backtest REFUSES to run (venue.yaml
           null blocks the pipeline on purpose).

Execution model
---------------
position held during (t, t+1] equals signal[t-1] (shifted by one bar).
gross pnl at t = position[t] * simple_return(t-1 -> t)
turnover at t  = sum |position[t] - position[t-1]|
cost at t      = turnover[t] * (taker_fee + slippage) ; funding applied on
                 funding bars.
"""
from __future__ import annotations

import numpy as np

REQUIRED_COST_FIELDS = ("taker_fee", "maker_fee", "slippage_bps")


class VenueNotConfigured(RuntimeError):
    pass


def _check_costs(costs: dict):
    missing = [k for k in REQUIRED_COST_FIELDS
               if costs.get(k) is None]
    if missing:
        raise VenueNotConfigured(
            f"venue.yaml fields still null: {missing}. The pipeline is blocked "
            f"on purpose until the official Binance USDT-M spec is filled in "
            f"(S1). Do not guess these numbers.")


def simple_returns(prices: np.ndarray) -> np.ndarray:
    """Backward close-to-close return: r[t] = P[t]/P[t-1] - 1 (realised at t)."""
    p = np.asarray(prices, float)
    r = np.zeros_like(p)
    r[1:] = p[1:] / p[:-1] - 1.0
    r[~np.isfinite(r)] = 0.0
    return r


def forward_returns(prices: np.ndarray) -> np.ndarray:
    """Forward return earned by a position opened at close[t]:
    fr[t] = P[t+1]/P[t] - 1. Last bar undefined -> 0."""
    p = np.asarray(prices, float)
    fr = np.zeros_like(p)
    fr[:-1] = p[1:] / p[:-1] - 1.0
    fr[~np.isfinite(fr)] = 0.0
    return fr


def run_backtest(prices: np.ndarray, signal: np.ndarray, costs: dict, *,
                 funding: np.ndarray | None = None,
                 funding_every: int | None = None,
                 gross_leverage: float = 1.0,
                 execution_lag: int = 1) -> dict:
    _check_costs(costs)
    prices = np.asarray(prices, float)
    signal = np.asarray(signal, float)
    T, N = prices.shape
    assert signal.shape == prices.shape, "signal/prices shape mismatch"
    assert execution_lag >= 1, "R9: execution_lag must be >= 1 (signal t -> fill t+1)"

    # R9 execution model. signal[t] is computed from information through the
    # close of bar t. It cannot be traded on bar t (already closed); the fill
    # happens `execution_lag` bars later, and the position then earns the bar
    # AFTER the fill onward. So a position that earns the return realised at
    # bar k was decided at bar k - execution_lag - 1.
    #   pnl[t] uses backward return rets[t]  (P[t]/P[t-1]-1)
    #   pos[t] = signal[t - execution_lag - 1]
    # With execution_lag=1 this is pos[t] = signal[t-2]: an oracle that knows
    # bar t+1's move does NOT profit; one that knows bar t+2's move does.
    rets = simple_returns(prices)
    shift = execution_lag + 1

    pos = np.zeros_like(signal)
    if shift < T:
        pos[shift:] = signal[:-shift]

    # normalise to target gross leverage each bar
    gross = np.sum(np.abs(pos), axis=1, keepdims=True)
    scale = np.divide(gross_leverage, gross, out=np.zeros_like(gross),
                      where=gross > 0)
    pos = pos * scale

    # per-bar portfolio gross return
    gross_pnl = np.sum(pos * rets, axis=1)

    # turnover & transaction cost
    dpos = np.zeros_like(pos)
    dpos[1:] = pos[1:] - pos[:-1]
    dpos[0] = pos[0]
    turnover = np.sum(np.abs(dpos), axis=1)
    fee = float(costs["taker_fee"]) + float(costs["slippage_bps"]) * 1e-4
    tc = turnover * fee

    # funding (perp): applied to net notional exposure on funding bars
    fund_pnl = np.zeros(T)
    if funding is not None:
        funding = np.asarray(funding, float)
        fund_pnl = -np.sum(pos * funding, axis=1)  # pay funding when long & rate>0
    elif funding_every:
        # unknown funding -> zero, but mark that it was omitted
        pass

    net_pnl = gross_pnl - tc + fund_pnl
    equity = np.cumprod(1.0 + net_pnl)
    gross_equity = np.cumprod(1.0 + gross_pnl)

    def _sharpe(x):
        x = x[np.isfinite(x)]
        return float(x.mean() / x.std(ddof=1)) if x.size > 1 and x.std(ddof=1) > 0 else 0.0

    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0

    return {
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "cost": tc,
        "funding_pnl": fund_pnl,
        "turnover": turnover,
        "equity": equity,
        "gross_equity": gross_equity,
        "position": pos,
        "sharpe_gross": _sharpe(gross_pnl),
        "sharpe_net": _sharpe(net_pnl),
        "ann_return_net": float(np.prod(1.0 + net_pnl) ** (1.0 / max(T, 1)) - 1.0),
        "total_cost": float(tc.sum()),
        "total_gross": float(gross_pnl.sum()),
        "total_net": float(net_pnl.sum()),
        "max_drawdown": float(dd.min()),
        "n_bars": int(T),
        "n_coins": int(N),
        "fee_used": fee,
    }


def cost_sweep(prices, signal, costs, multipliers=(0.5, 1.0, 2.0, 4.0)) -> list:
    """Directional cost check: raising fees lowers net, leaves gross unchanged."""
    out = []
    base_taker = costs["taker_fee"]
    base_slip = costs["slippage_bps"]
    for m in multipliers:
        c = dict(costs)
        c["taker_fee"] = base_taker * m
        c["slippage_bps"] = base_slip * m
        r = run_backtest(prices, signal, c)
        out.append({"mult": m, "gross": r["total_gross"], "net": r["total_net"],
                    "sharpe_net": r["sharpe_net"]})
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, N = 2000, 8
    px = 100 * np.cumprod(1 + rng.normal(0, 0.01, (T, N)), axis=0)
    sig = np.sign(rng.normal(size=(T, N)))
    costs = {"taker_fee": 0.0004, "maker_fee": 0.0002, "slippage_bps": 1.0}
    r = run_backtest(px, sig, costs)
    print(f"sharpe gross={r['sharpe_gross']:.3f} net={r['sharpe_net']:.3f} "
          f"cost={r['total_cost']:.4f}")
    sweep = cost_sweep(px, sig, costs)
    for s in sweep:
        print(s)
    assert all(abs(s["gross"] - sweep[0]["gross"]) < 1e-9 for s in sweep), "gross moved!"
    assert sweep[0]["net"] > sweep[-1]["net"], "higher cost did not lower net"
    print("backtest selftest OK")
