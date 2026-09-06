#!/usr/bin/env python3
"""Synthetic multi-coin OHLCV generator with crypto microstructure.

Design targets (from the S0 brief):
  * volatility clustering (GARCH-like)
  * jumps (rare large moves)
  * 8-hour funding cycle
  * cross-coin correlation that SURGES during crashes - diversification
    disappears exactly when it is most needed. This MUST be in the calibration
    data or the risk model is being fit to a friendlier world than reality.

  --edge 0.0    control: no predictability. A strategy on this data must lose
                exactly the transaction cost.
  --edge 0.35   planted edge: a CAUSAL, live-computable feature (trailing
                cross-sectional momentum) predicts the next bar's return with
                an information coefficient of ~0.35.

Output: parquet/csv per coin under the chosen directory, plus funding.csv and
a manifest.json describing the planted structure (for calibration only - the
research pipeline never reads the manifest).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _garch_vol(n, rng, omega=2e-6, a=0.08, b=0.90, base=0.02):
    v = np.empty(n)
    v[0] = base ** 2
    eps = rng.normal(size=n)
    for t in range(1, n):
        v[t] = omega + a * (eps[t - 1] ** 2 * v[t - 1]) + b * v[t - 1]
    return np.sqrt(v), eps


def generate(n_coins=12, n_bars=20000, bar_minutes=60, edge=0.0, seed=0,
             mom_lookback=24, jump_prob=0.002, crash_prob=0.0007):
    rng = np.random.default_rng(seed)

    # --- market factor with regime-switching (calm / stressed) --------------
    mkt_vol, mkt_eps = _garch_vol(n_bars, rng, base=0.012)
    regime = np.zeros(n_bars, dtype=int)         # 0 calm, 1 stressed
    p_stay = {0: 0.995, 1: 0.97}
    for t in range(1, n_bars):
        stay = rng.random() < p_stay[regime[t - 1]]
        regime[t] = regime[t - 1] if stay else 1 - regime[t - 1]

    crash = (rng.random(n_bars) < crash_prob) | (regime == 1) & (rng.random(n_bars) < 0.02)
    mkt_ret = mkt_eps * mkt_vol
    mkt_ret[crash] -= rng.uniform(0.03, 0.12, size=crash.sum())   # crash drops

    # --- per-coin construction -------------------------------------------
    betas = rng.uniform(0.6, 1.5, n_coins)
    idio_vol = np.array([_garch_vol(n_bars, rng, base=v)[0]
                         for v in rng.uniform(0.008, 0.02, n_coins)])
    idio = rng.normal(size=(n_coins, n_bars)) * idio_vol

    # correlation surge in crashes: replace idio with common shock
    common_shock = rng.normal(size=n_bars) * 0.04
    surge_w = np.where(crash, 0.85, np.where(regime == 1, 0.35, 0.0))
    idio = (1 - surge_w) * idio + surge_w * common_shock

    ret = betas[:, None] * mkt_ret[None, :] + idio            # (N, T) log-ish returns

    # --- planted edge: trailing x-sec momentum at bar t predicts the return
    #     realised at bar t+2 -----------------------------------------------
    #  This matches the backtest execution model (signal at t, fill at t+1,
    #  first earned bar return is t+2). An edge planted at t+1 instead would be
    #  in the one-bar dead zone and undetectable - which is the point of the
    #  off-by-one test.
    realized_ic = None
    LEAD = 2
    if edge and edge > 0:
        logp0 = np.cumsum(ret, axis=1)
        mom = np.zeros_like(ret)
        mom[:, mom_lookback:] = logp0[:, mom_lookback:] - logp0[:, :-mom_lookback]
        mu = mom.mean(axis=0, keepdims=True)
        sd = mom.std(axis=0, keepdims=True)
        z = (mom - mu) / np.where(sd > 0, sd, 1.0)
        z[:, :mom_lookback] = 0.0
        # ret[:, t+LEAD] += c * z[:, t] * local_sigma ; c set by bisection so the
        # realised information coefficient corr(z_t, ret_{t+LEAD}) == `edge`.
        local_sigma = idio_vol.mean(axis=0)          # (T,)
        base0 = np.zeros_like(ret)
        base0[:, LEAD:] = z[:, :-LEAD] * local_sigma[None, :-LEAD]
        ret_base = ret.copy()
        fsel = z[:, mom_lookback:-LEAD].ravel()

        def ic_for(c):
            rr = ret_base + c * base0
            y = rr[:, mom_lookback + LEAD:].ravel()
            mm = np.isfinite(fsel) & np.isfinite(y)
            return float(np.corrcoef(fsel[mm], y[mm])[0, 1])

        lo, hi = 0.0, 1.0
        while ic_for(hi) < edge and hi < 1e4:
            hi *= 2
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if ic_for(mid) < edge:
                lo = mid
            else:
                hi = mid
        c = 0.5 * (lo + hi)
        ret = ret_base + c * base0
        realized_ic = ic_for(c)

    # --- prices + OHLC from returns -------------------------------------
    price0 = rng.uniform(5, 40000, n_coins)
    close = price0[:, None] * np.exp(np.cumsum(ret, axis=1))
    # intrabar range scaled by vol
    rng_bar = np.abs(rng.normal(size=(n_coins, n_bars))) * (idio_vol + mkt_vol[None, :])
    high = close * (1 + rng_bar)
    low = close * (1 - rng_bar)
    open_ = np.empty_like(close)
    open_[:, 0] = price0
    open_[:, 1:] = close[:, :-1]
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    volume = np.abs(rng.normal(1e6, 3e5, (n_coins, n_bars))) * (1 + 5 * crash[None, :])

    # --- funding: 8h cycle -------------------------------------------------
    bars_per_8h = int(round(8 * 60 / bar_minutes))
    funding = np.zeros((n_coins, n_bars))
    fund_bars = np.arange(0, n_bars, bars_per_8h)
    base_prem = rng.normal(0.0001, 0.00005, n_coins)
    for fb in fund_bars:
        funding[:, fb] = base_prem + 0.0002 * np.tanh(ret[:, max(fb - 1, 0)]) \
            + rng.normal(0, 0.00003, n_coins)

    return {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "funding": funding, "fund_bars": fund_bars,
        "regime": regime, "crash": crash, "betas": betas,
        "edge": edge, "mom_lookback": mom_lookback, "realized_ic": realized_ic,
        "bar_minutes": bar_minutes,
    }


def write_out(data: dict, out_dir: str, symbols=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    N, T = data["close"].shape
    symbols = symbols or [f"SYN{i:02d}USDT" for i in range(N)]
    ts0 = np.datetime64("2021-01-01T00:00:00")
    step = np.timedelta64(data["bar_minutes"], "m")
    ts = (ts0 + np.arange(T) * step).astype("datetime64[s]").astype(str)
    for i, sym in enumerate(symbols):
        rows = np.column_stack([
            ts,
            data["open"][i], data["high"][i], data["low"][i],
            data["close"][i], data["volume"][i], data["funding"][i],
        ])
        hdr = "timestamp,open,high,low,close,volume,funding"
        np.savetxt(out / f"{sym}.csv", rows, fmt="%s", delimiter=",", header=hdr,
                   comments="")
    manifest = {
        "symbols": symbols, "n_bars": int(T), "bar_minutes": data["bar_minutes"],
        "edge": data["edge"], "mom_lookback": data["mom_lookback"],
        "realized_ic": data["realized_ic"],
        "n_crash_bars": int(data["crash"].sum()),
        "pct_stressed": float(np.mean(data["regime"] == 1)),
        "note": "manifest is for calibration only; the research pipeline must "
                "not read it (R1/R4).",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", type=float, default=0.0)
    ap.add_argument("--coins", type=int, default=12)
    ap.add_argument("--bars", type=int, default=20000)
    ap.add_argument("--bar-minutes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = generate(n_coins=args.coins, n_bars=args.bars, bar_minutes=args.bar_minutes,
                 edge=args.edge, seed=args.seed)
    man = write_out(d, args.out)
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
