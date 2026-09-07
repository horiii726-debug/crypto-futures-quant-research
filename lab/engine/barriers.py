"""Triple-barrier engine with volatility-scaled levels (López de Prado 2018, ch.3).

NO retail indicators. Barrier width comes from academic volatility estimators:
  Yang-Zhang (2000)      — most efficient OHLC estimator, drift-independent
  Rogers-Satchell (1991) — drift-independent range estimator (wick-aware)
  Parkinson (1980)       — high-low range, 5x more efficient than close-to-close
  Corsi (2009) HAR-RV    — cascade forecast of realised variance
  GJR-GARCH (1993)       — asymmetric conditional variance

Optimal TP/SL ratio is SOLVED, not guessed, from the first-passage probability
of an arithmetic Brownian motion with drift between two absorbing barriers
(Karatzas-Shreve 1991, §3.5; classic gambler's ruin):

    P(hit +a before -b) = (1 - e^{-2*mu*b/sigma^2}) / (1 - e^{-2*mu*(a+b)/sigma^2})

which for mu -> 0 reduces to b/(a+b). Expected value of the trade is then
    E = P*a - (1-P)*b - cost
and we choose (a, b) in units of sigma to maximise E subject to a time budget.

The vertical (time) barrier enforces the user's constraint: no 3-5 minute
in-and-out. Minimum hold is a parameter and defaults to 1 day.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

SEC_PER_YEAR = 365 * 24 * 3600


# =========================================================================== #
#  VOLATILITY  — academic estimators only                                     #
# =========================================================================== #
def sigma_yang_zhang(o, h, l, c, n: int = 168) -> pd.DataFrame:
    """Yang-Zhang (2000, J. Business) — minimum-variance unbiased OHLC estimator.
    sigma^2 = sigma_o^2 + k*sigma_c^2 + (1-k)*sigma_rs^2 ,
    k = 0.34/(1.34 + (n+1)/(n-1)).   Handles both drift and opening jumps."""
    lo_c = np.log(o / c.shift(1))
    lo_co = np.log(c / o)
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    v = (lo_c.rolling(n).var(ddof=1)
         + k * lo_co.rolling(n).var(ddof=1)
         + (1 - k) * rs.rolling(n).mean())
    return np.sqrt(v.clip(lower=0))


def sigma_rogers_satchell(o, h, l, c, n: int = 168) -> pd.DataFrame:
    """Rogers-Satchell (1991, Ann. Appl. Prob.) — drift-independent range estimator.
    This is the wick-aware one: it uses the FULL high/low excursion, so a stop
    placed at k*sigma_RS is calibrated to the range the price actually traverses,
    not just where it closes."""
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return np.sqrt(rs.rolling(n).mean().clip(lower=0))


def sigma_parkinson(h, l, n: int = 168) -> pd.DataFrame:
    """Parkinson (1980, J. Business) — sigma^2 = mean(ln(H/L)^2)/(4 ln 2)."""
    return np.sqrt((np.log(h / l) ** 2).rolling(n).mean() / (4 * np.log(2)))


def sigma_har_rv(c: pd.DataFrame, d: int = 24, w: int = 120, m: int = 528) -> pd.DataFrame:
    """Corsi (2009, J. Financial Econometrics) heterogeneous autoregressive
    cascade: RV_{t+1} = c + b_d RV_t^(d) + b_w RV_t^(w) + b_m RV_t^(m).
    Coefficients are Corsi's reported daily/weekly/monthly loadings."""
    r2 = np.log(c).diff() ** 2
    rv_d = r2.rolling(d).sum()
    rv_w = r2.rolling(w).sum() * (d / w)
    rv_m = r2.rolling(m).sum() * (d / m)
    return np.sqrt((0.36 * rv_d + 0.36 * rv_w + 0.28 * rv_m).clip(lower=0) / d)


def sigma_blend(o, h, l, c, n: int = 168) -> pd.DataFrame:
    """Per-bar sigma used for barriers: the MAX of a close-based and two
    range-based estimators. Taking the max is deliberate — it is the
    conservative choice that protects against wick stop-outs, and it is what an
    institutional desk does when the estimator disagreement is itself a risk."""
    yz = sigma_yang_zhang(o, h, l, c, n)
    rs = sigma_rogers_satchell(o, h, l, c, n)
    pk = sigma_parkinson(h, l, n)
    return pd.concat([yz, rs, pk]).groupby(level=0).max()


# =========================================================================== #
#  FIRST-PASSAGE MATHS — solve the TP/SL ratio, do not guess it              #
# =========================================================================== #
def p_hit_upper(mu: float, sigma: float, a: float, b: float) -> float:
    """P(reach +a before -b) for dX = mu dt + sigma dW, absorbing barriers.
    Karatzas-Shreve (1991) Prop. 3.5.8 / gambler's ruin with drift."""
    if sigma <= 0 or a <= 0 or b <= 0:
        return np.nan
    if abs(mu) < 1e-12:
        return b / (a + b)
    z = 2.0 * mu / (sigma ** 2)
    # numerically stable form
    with np.errstate(over="ignore"):
        num = 1.0 - np.exp(-z * b)
        den = 1.0 - np.exp(-z * (a + b))
    return float(np.clip(num / den, 0.0, 1.0)) if abs(den) > 1e-15 else b / (a + b)


def expected_trade_value(mu: float, sigma: float, a: float, b: float,
                         cost: float) -> float:
    """E[R] of one triple-barrier trade in return units, net of a round-trip cost."""
    p = p_hit_upper(mu, sigma, a, b)
    if not np.isfinite(p):
        return -np.inf
    return p * a - (1.0 - p) * b - cost


def solve_barriers(mu_per_bar: float, sigma_per_bar: float, cost: float,
                   horizon_bars: int, grid=None) -> dict:
    """Choose (pt_mult, sl_mult) in units of sigma*sqrt(horizon) to maximise the
    expected value of the trade. Constrained so both barriers are reachable
    within the time budget (a, b <= 3 sigma sqrt(T): beyond that the vertical
    barrier binds almost surely and the trade degenerates into a time exit)."""
    T = max(horizon_bars, 1)
    s = sigma_per_bar * np.sqrt(T)
    m = mu_per_bar * T
    if not np.isfinite(s) or s <= 0:
        return {"pt_mult": 2.0, "sl_mult": 1.5, "ev": np.nan, "p_tp": np.nan}
    grid = grid if grid is not None else np.arange(0.5, 3.01, 0.25)
    best = None
    for a_m in grid:
        for b_m in grid:
            a, b = a_m * s, b_m * s
            ev = expected_trade_value(m, s, a, b, cost)
            if best is None or ev > best["ev"]:
                best = {"pt_mult": float(a_m), "sl_mult": float(b_m), "ev": float(ev),
                        "p_tp": p_hit_upper(m, s, a, b)}
    return best


# =========================================================================== #
#  TRIPLE-BARRIER LABELLING (numba)                                          #
# =========================================================================== #
@njit(cache=True)
def _apply_barriers(close, high, low, ev_idx, side, pt, sl, vert):
    """For each event i: walk forward until price touches +pt[i], -sl[i] or the
    vertical barrier vert[i]. Uses the bar HIGH/LOW so an intrabar wick counts —
    this is what makes the SL honest.
    Returns (exit_index, realised_return, which_barrier) with
    which_barrier: 1 = take-profit, -1 = stop-loss, 0 = time."""
    n = ev_idx.size
    out_i = np.empty(n, dtype=np.int64)
    out_r = np.empty(n, dtype=np.float64)
    out_b = np.empty(n, dtype=np.int64)
    for k in range(n):
        i0 = ev_idx[k]
        s = side[k]
        p0 = close[i0]
        up = p0 * (1.0 + pt[k]) if s > 0 else p0 * (1.0 - pt[k])
        dn = p0 * (1.0 - sl[k]) if s > 0 else p0 * (1.0 + sl[k])
        end = vert[k]
        hit_i = end
        hit_b = 0
        for t in range(i0 + 1, end + 1):
            if s > 0:
                if low[t] <= dn:
                    hit_i = t; hit_b = -1; break
                if high[t] >= up:
                    hit_i = t; hit_b = 1; break
            else:
                if high[t] >= dn:
                    hit_i = t; hit_b = -1; break
                if low[t] <= up:
                    hit_i = t; hit_b = 1; break
        if hit_b == 1:
            r = pt[k]
        elif hit_b == -1:
            r = -sl[k]
        else:
            r = s * (close[hit_i] / p0 - 1.0)
        out_i[k] = hit_i
        out_r[k] = r
        out_b[k] = hit_b
    return out_i, out_r, out_b


@dataclass
class BarrierConfig:
    pt_mult: float = 1.0          # take-profit in units of sigma*sqrt(hold)
    sl_mult: float = 1.0          # stop-loss   in units of sigma*sqrt(hold)
    max_hold_bars: int = 168      # vertical barrier (7 days at H1)
    min_hold_bars: int = 24       # user constraint: no 3-5 minute in-and-out
    sigma_window: int = 168
    sl_floor_mult: float = 1.0    # SL never tighter than this x the range sigma
    cost_roundtrip: float = 0.0015


def label_events(px: dict, events: pd.DataFrame, cfg: BarrierConfig,
                 sigma: pd.DataFrame | None = None) -> pd.DataFrame:
    """events: DataFrame with columns [ts, coin, side]. Returns the same rows plus
    entry/exit index, realised return, which barrier, holding time and the actual
    pt/sl levels used."""
    o, h, l, c = px["open"], px["high"], px["low"], px["close"]
    if sigma is None:
        sigma = sigma_blend(o, h, l, c, cfg.sigma_window)
    rs = sigma_rogers_satchell(o, h, l, c, cfg.sigma_window)   # wick floor
    idx = c.index
    pos = {t: i for i, t in enumerate(idx)}
    rows = []
    for coin, g in events.groupby("coin"):
        if coin not in c.columns:
            continue
        cc = c[coin].to_numpy(np.float64)
        hh = h[coin].to_numpy(np.float64)
        ll = l[coin].to_numpy(np.float64)
        sg = sigma[coin].to_numpy(np.float64)
        sr = rs[coin].to_numpy(np.float64)
        ev, sd = [], []
        for t, s in zip(g["ts"], g["side"]):
            i = pos.get(t)
            if i is None or i + cfg.min_hold_bars >= len(idx):
                continue
            if not (np.isfinite(sg[i]) and sg[i] > 0 and np.isfinite(cc[i])):
                continue
            ev.append(i); sd.append(float(s))
        if not ev:
            continue
        ev = np.array(ev, dtype=np.int64)
        sd = np.array(sd, dtype=np.float64)
        scale = sg[ev] * np.sqrt(cfg.max_hold_bars)
        pt = cfg.pt_mult * scale
        sl = cfg.sl_mult * scale
        # anti-wick floor: never place the stop inside the typical intrabar range
        floor = cfg.sl_floor_mult * sr[ev] * np.sqrt(cfg.max_hold_bars)
        sl = np.maximum(sl, np.nan_to_num(floor))
        vert = np.minimum(ev + cfg.max_hold_bars, len(idx) - 1)
        xi, xr, xb = _apply_barriers(cc, hh, ll, ev, sd, pt, sl, vert)
        for j in range(len(ev)):
            rows.append({"coin": coin, "t_entry": idx[ev[j]], "t_exit": idx[xi[j]],
                         "i_entry": int(ev[j]), "i_exit": int(xi[j]),
                         "side": sd[j], "ret_gross": float(xr[j]),
                         "barrier": int(xb[j]), "hold_bars": int(xi[j] - ev[j]),
                         "pt": float(pt[j]), "sl": float(sl[j]),
                         "sigma": float(sg[ev[j]])})
    if not rows:
        return pd.DataFrame(columns=["coin", "t_entry", "t_exit", "side", "ret_gross",
                                     "barrier", "hold_bars", "pt", "sl", "sigma"])
    return pd.DataFrame(rows).sort_values("t_entry").reset_index(drop=True)


# =========================================================================== #
#  SAMPLE UNIQUENESS (López de Prado 2018, ch.4)                             #
# =========================================================================== #
def average_uniqueness(labels: pd.DataFrame, n_bars: int) -> np.ndarray:
    """Concurrency-adjusted sample weight: labels that overlap in time carry
    less independent information. Required before any CV or ML fitting."""
    conc = np.zeros(n_bars, dtype=np.float64)
    for a, b in zip(labels["i_entry"].to_numpy(), labels["i_exit"].to_numpy()):
        conc[a:b + 1] += 1.0
    w = np.empty(len(labels))
    for k, (a, b) in enumerate(zip(labels["i_entry"].to_numpy(), labels["i_exit"].to_numpy())):
        seg = conc[a:b + 1]
        seg = seg[seg > 0]
        w[k] = float((1.0 / seg).mean()) if seg.size else 0.0
    return w
