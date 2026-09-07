"""Event-level tape (aggTrades) -> causal microstructure bar features.

Every feature at bar close time T uses only trades with ts <= T. Bar T covers
[T, T+dt). Streamed day-by-day (a year of BTC tape is ~10^9 rows).

One numba pass per day computes all per-bar aggregates. Bar columns:
  ofi_usd, buy_frac, aggr_imb, n_trades, notional, avg_trade_usd, kyle_lambda,
  vwap, open, high, low, close, ret, intensity, big_trade_imb,
  trade_sign_ac1, roll_spread_bps, realised_vol
plus (cross-day) cum_delta, twap_ret.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from lab.data.binance_vision import CACHE

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed" / "tape"


def _read_day(symbol: str, day: str):
    """Return (ts, price, qty, sell_aggr) numpy arrays or None. Fast path:
    numpy genfromtxt on the raw CSV bytes."""
    key = f"data/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day}.zip"
    p = CACHE / key
    if not p.exists():
        return None
    with zipfile.ZipFile(io.BytesIO(p.read_bytes())) as z:
        raw = z.read(z.namelist()[0])
    first = raw.split(b"\n", 1)[0].split(b",")[0].strip()
    has_hdr = not first.replace(b".", b"").replace(b"-", b"").isdigit()
    from pyarrow import csv as pacsv
    if has_hdr:
        tbl = pacsv.read_csv(io.BytesIO(raw))
        c = tbl.column_names
        px, qt, tt, bmc = c[1], c[2], c[5], c[6]
    else:
        tbl = pacsv.read_csv(io.BytesIO(raw),
                             read_options=pacsv.ReadOptions(autogenerate_column_names=True))
        px, qt, tt, bmc = "f1", "f2", "f5", "f6"
    price = tbl.column(px).to_numpy(zero_copy_only=False).astype(np.float64)
    qty = tbl.column(qt).to_numpy(zero_copy_only=False).astype(np.float64)
    ts = tbl.column(tt).to_numpy(zero_copy_only=False).astype(np.float64)
    bm = tbl.column(bmc).to_numpy(zero_copy_only=False)
    if ts.size == 0:
        return None
    if ts[0] > 2e15:
        ts = ts / 1000.0
    if bm.dtype == bool:
        sell_aggr = bm
    else:
        sell_aggr = np.char.lower(bm.astype("U5")).astype("<U1") == "t"
    order = np.argsort(ts, kind="stable")
    return ts[order], price[order], qty[order], sell_aggr[order]


@njit(cache=False, fastmath=False)
def _day_bars(ts, price, qty, sell_aggr, bar_ms, t0, p90):
    n = ts.shape[0]
    nb = int((ts[-1] - t0) // bar_ms) + 1
    ofi = np.zeros(nb); buy_n = np.zeros(nb); sell_n = np.zeros(nb); cnt = np.zeros(nb)
    fp = np.full(nb, np.nan); lp = np.full(nb, np.nan)
    spv = np.zeros(nb); sv = np.zeros(nb)
    hi = np.full(nb, -1e18); lo = np.full(nb, 1e18)
    # kyle regression sums
    sx = np.zeros(nb); sy = np.zeros(nb); sxx = np.zeros(nb); sxy = np.zeros(nb); kn = np.zeros(nb)
    # big-trade signed notional
    big_s = np.zeros(nb); big_a = np.zeros(nb)
    # trade-sign AC1 sums: sum(s_t), sum(s_t s_{t-1}), count of pairs
    ss = np.zeros(nb); ss1 = np.zeros(nb); sp = np.zeros(nb)
    prev_sign = np.zeros(nb)
    have_prev = np.zeros(nb)
    # roll spread: sum(dp_t dp_{t-1}), count  (dp = log price change)
    rc = np.zeros(nb); rn = np.zeros(nb)
    prev_dp = np.zeros(nb); have_dp = np.zeros(nb)
    prev_lp_in_bar = np.zeros(nb); have_lpb = np.zeros(nb)
    # realised vol: sum dp^2, count
    rv2 = np.zeros(nb); rvn = np.zeros(nb)
    prev_p_global = price[0]
    for i in range(n):
        b = int((ts[i] - t0) // bar_ms)
        if b < 0 or b >= nb:
            continue
        s = -1.0 if sell_aggr[i] else 1.0
        nz = qty[i] * price[i]
        ofi[b] += s * nz
        if s > 0:
            buy_n[b] += nz
        else:
            sell_n[b] += nz
        cnt[b] += 1.0
        if np.isnan(fp[b]):
            fp[b] = price[i]
        lp[b] = price[i]
        spv[b] += price[i] * nz
        sv[b] += nz
        if price[i] > hi[b]:
            hi[b] = price[i]
        if price[i] < lo[b]:
            lo[b] = price[i]
        # kyle
        dpk = price[i] - prev_p_global
        xv = s * np.sqrt(nz)
        sx[b] += xv; sy[b] += dpk; sxx[b] += xv * xv; sxy[b] += xv * dpk; kn[b] += 1.0
        prev_p_global = price[i]
        # big trade
        if nz >= p90:
            big_s[b] += s * nz
            big_a[b] += nz
        # sign AC1
        if have_prev[b] > 0.5:
            ss1[b] += s * prev_sign[b]
            sp[b] += 1.0
        ss[b] += s
        prev_sign[b] = s
        have_prev[b] = 1.0
        # roll spread / rv from within-bar log-price changes
        if have_lpb[b] > 0.5:
            dp = np.log(price[i] / prev_lp_in_bar[b])
            rv2[b] += dp * dp
            rvn[b] += 1.0
            if have_dp[b] > 0.5:
                rc[b] += dp * prev_dp[b]
                rn[b] += 1.0
            prev_dp[b] = dp
            have_dp[b] = 1.0
        prev_lp_in_bar[b] = price[i]
        have_lpb[b] = 1.0

    tot = buy_n + sell_n
    kyle = np.full(nb, np.nan)
    ac1 = np.full(nb, np.nan)
    rollb = np.full(nb, np.nan)
    rv = np.full(nb, np.nan)
    big_imb = np.full(nb, np.nan)
    for b in range(nb):
        d = kn[b] * sxx[b] - sx[b] * sx[b]
        if d != 0.0 and kn[b] > 5:
            kyle[b] = (kn[b] * sxy[b] - sx[b] * sy[b]) / d
        if sp[b] > 4:
            mean_s = ss[b] / cnt[b]
            cov = ss1[b] / sp[b] - mean_s * mean_s
            var = 1.0 - mean_s * mean_s
            if var > 1e-9:
                ac1[b] = cov / var
        if rn[b] > 2:
            c = rc[b] / rn[b]
            rollb[b] = 2.0 * np.sqrt(-c) * 1e4 if c < 0 else 0.0
        if rvn[b] > 2:
            rv[b] = np.sqrt(rv2[b] / rvn[b])
        if big_a[b] > 0:
            big_imb[b] = big_s[b] / big_a[b]
    return (ofi, buy_n, sell_n, cnt, fp, lp, spv, sv, hi, lo, kyle, ac1, rollb,
            rv, big_imb, tot)


def _bars_for_day(day_arrays, bar_min: int) -> pd.DataFrame:
    ts, price, qty, sell_aggr = day_arrays
    notional_all = qty * price
    p90 = np.quantile(notional_all, 0.90)
    bar_ms = bar_min * 60_000.0
    t0 = (ts[0] // bar_ms) * bar_ms
    (ofi, buy_n, sell_n, cnt, fp, lp, spv, sv, hi, lo, kyle, ac1, rollb, rv,
     big_imb, tot) = _day_bars(ts, price, qty, sell_aggr, bar_ms, t0, p90)
    nb = len(ofi)
    idx = pd.to_datetime((t0 + np.arange(nb) * bar_ms).astype("int64"), unit="ms", utc=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        buy_frac = np.where(tot > 0, buy_n / tot, np.nan)
        aggr_imb = np.where(tot > 0, (buy_n - sell_n) / tot, np.nan)
        vwap = np.where(sv > 0, spv / sv, np.nan)
    df = pd.DataFrame({
        "ofi_usd": ofi, "buy_frac": buy_frac, "aggr_imb": aggr_imb, "n_trades": cnt,
        "notional": tot, "avg_trade_usd": np.where(cnt > 0, tot / cnt, np.nan),
        "kyle_lambda": kyle, "vwap": vwap, "open": fp,
        "high": np.where(hi > -1e17, hi, np.nan), "low": np.where(lo < 1e17, lo, np.nan),
        "close": lp, "intensity": cnt / (bar_min * 60.0), "big_trade_imb": big_imb,
        "trade_sign_ac1": ac1, "roll_spread_bps": rollb, "realised_vol": rv,
    }, index=idx)
    df["ret"] = np.log(df["close"] / df["open"])
    return df[df["n_trades"] > 0]


def bar_features(symbol: str, days: list[str], bar_min: int) -> pd.DataFrame:
    out = []
    for d in days:
        da = _read_day(symbol, d)
        if da is None or da[0].size < 100:
            continue
        try:
            out.append(_bars_for_day(da, bar_min))
        except Exception:  # noqa
            continue
        del da
    if not out:
        return pd.DataFrame()
    df = pd.concat(out).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["twap_ret"] = np.log(df["vwap"] / df["vwap"].shift(1))
    df["cum_delta"] = df["ofi_usd"].cumsum()
    df["symbol"] = symbol
    return df


def build_tape_panel(symbols, days, bar_min, cache=True):
    out = {}
    PROC.mkdir(parents=True, exist_ok=True)
    for s in symbols:
        f = PROC / f"{s}_{bar_min}m.parquet"
        if cache and f.exists():
            out[s] = pd.read_parquet(f)
            continue
        df = bar_features(s, days, bar_min)
        if len(df):
            df.to_parquet(f)
            out[s] = df
    return out


if __name__ == "__main__":
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range("2025-03-01", "2025-03-07")]
    import time
    t = time.time()
    df = bar_features("BTCUSDT", days, 5)
    print(f"{df.shape} in {time.time()-t:.1f}s")
    print(df[["close", "ofi_usd", "buy_frac", "kyle_lambda", "roll_spread_bps",
              "trade_sign_ac1", "big_trade_imb"]].tail(5).to_string())
