"""Dynamic per-trade cost model — RESEARCH ROUND 2 · P0.3.

Single API used by EVERY backtest:

    cost_bps(coin, side, notional, ts, mode) -> dict
        {fee, spread, impact, slippage, adverse_sel, p_fill, tier, total_bps}

All components are per-trade, not constants. Calibration inputs are frozen by
`lab/reports/REPRICING_RULE.md`; nothing here is fitted per coin except the
quantities that are directly *measured* per coin (half-spread, adverse
selection, near-touch depth, arrival rate). The impact coefficient Y, the fee
schedule and the tier fallbacks are global.

`cost_oneway_bps_by_coin(coins, notional, mode)` returns a per-coin scalar for
vectorised re-scoring of the existing trials (the trials track turnover as a
fraction of book, not per-fill notional, so a fixed research notional is used —
see CHANGELOG_METHOD 2026-09-07).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma as _gamma

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
_CACHE = PROC / "cost_model_cache.parquet"

# ---- frozen global constants (REPRICING_RULE §2, §3, §4) --------------------
FEE_TAKER_BPS = 4.5          # Binance USDT-M perp VIP-0 taker, post BNB schedule
FEE_MAKER_BPS = 1.8          # VIP-0 maker
Y_IMPACT = 0.5               # square-root-law coefficient, ONE value for the universe
TIF_ENTRY_SEC = 45.0
TIF_EXIT_SEC = 10.0
AS_FLOOR_BPS = 0.75          # ROUND 1 adverse-selection floor
TIER_HALF_SPREAD = {1: 0.4, 2: 1.2, 3: 4.0}
TIER_PFILL = {1: 0.92, 2: 0.85, 3: 0.72}
TIER_ADV_USD = {1: 300e6, 2: 30e6}          # >=300M -> tier1 ; 30-300M -> tier2 ; else tier3
RESEARCH_NOTIONAL_USD = 20_000.0            # ~ $1M market-neutral book / 28 names / 1.8 turns

# --------------------------------------------------------------------------- #
#  calibration load                                                          #
# --------------------------------------------------------------------------- #
def _load_half_spread() -> dict:
    cov: dict[str, float] = {}
    for f in ("exec_calibration.json",):
        for r in json.loads((PROC / f).read_text()):
            rt = r.get("roundtrip", {})
            e = rt.get("entry", {}) if isinstance(rt, dict) else {}
            # entry effective one-way minus fee ~ half-spread proxy; prefer explicit spread files
        # exec_calibration has no clean half-spread field -> skip, use spread files
    for f in ("aggtrades_micro.json", "spread_measured.json", "spread_extra.json"):
        p = PROC / f
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            if r.get("status") == "ok" and r.get("half_spread_bps") is not None:
                cov.setdefault(r["symbol"], float(r["half_spread_bps"]))
    se = json.loads((PROC / "spread_estimates.json").read_text()).get("realised_by_symbol", {})
    for k, v in se.items():
        cov.setdefault(k, float(v))
    return cov


def _load_adverse_sel() -> dict:
    """per-coin round-trip adverse selection (entry + exit) in bps, from the
    5 fully-calibrated coins."""
    out = {}
    for r in json.loads((PROC / "exec_calibration.json").read_text()):
        if r.get("status") != "ok":
            continue
        e = r.get("entry_sim", {}).get("adverse_selection_bps")
        x = r.get("exit_sim", {}).get("adverse_selection_bps")
        if e is not None and x is not None:
            # only the ADVERSE (positive = against us) part matters; entry is often
            # negative (favourable). Round 1 used a 0.75 bps floor per leg.
            out[r["symbol"]] = (max(0.0, float(e)) + max(0.0, float(x)))
    return out


def _load_queue() -> dict:
    out = {}
    for r in json.loads((PROC / "exec_calibration.json").read_text()):
        if r.get("status") != "ok":
            continue
        q = r.get("entry_sim", {}).get("queue_ahead_notional")
        pf_e = r.get("entry_sim", {}).get("p_fill")
        pf_x = r.get("exit_sim", {}).get("p_fill")
        out[r["symbol"]] = {"q_ahead": float(q) if q else None,
                            "p_fill_entry": pf_e, "p_fill_exit": pf_x}
    return out


def _load_near_touch() -> dict:
    out = {}
    p = PROC / "bookdepth_features.json"
    if not p.exists():
        return out
    for r in json.loads(p.read_text()):
        if r.get("status") != "ok":
            continue
        b = r.get("near_touch_notional_bid_usd_median")
        a = r.get("near_touch_notional_ask_usd_median")
        if b and a:
            out[r["symbol"]] = 0.5 * (float(b) + float(a))     # ~ +-1% band notional
    return out


def _build_vol_adv() -> pd.DataFrame:
    """sigma_daily (30d rolling std of daily log return, sample-median) and ADV
    (30d median daily quote-volume) per coin, from 1m klines."""
    if _CACHE.exists():
        return pd.read_parquet(_CACHE)
    rows = []
    kdir = PROC / "klines_1m"
    for f in sorted(kdir.glob("*.parquet")):
        sym = f.stem
        try:
            d = pd.read_parquet(f)
            tcol = "dt" if "dt" in d.columns else ("close_time" if "close_time" in d.columns else "open_time")
            idx = pd.to_datetime(d[tcol], utc=True, unit=("ms" if tcol in ("open_time", "close_time") else None),
                                 errors="coerce")
            c = pd.Series(d["close"].astype(float).values, index=idx)
            qv = pd.Series((d["quote_volume"].astype(float).values if "quote_volume" in d.columns
                            else d["volume"].astype(float).values * d["close"].astype(float).values), index=idx)
            dc = c.resample("1D").last()
            dqv = qv.resample("1D").sum()
            lr = np.log(dc).diff()
            sig = lr.rolling(30, min_periods=15).std()
            adv = dqv.rolling(30, min_periods=15).median()
            rows.append({"symbol": sym,
                         "sigma_daily": float(np.nanmedian(sig)),
                         "adv_usd": float(np.nanmedian(adv))})
        except Exception as e:  # noqa
            rows.append({"symbol": sym, "sigma_daily": np.nan, "adv_usd": np.nan})
    df = pd.DataFrame(rows).set_index("symbol")
    df.to_parquet(_CACHE)
    return df


@lru_cache(maxsize=1)
def _calib() -> dict:
    hs = _load_half_spread()
    va = _build_vol_adv()
    return {
        "half_spread": hs,
        "adverse": _load_adverse_sel(),
        "queue": _load_queue(),
        "near_touch": _load_near_touch(),
        "vol_adv": va,
        "sigma_daily_median": float(np.nanmedian(va["sigma_daily"])),
        "half_spread_median": float(np.nanmedian(list(hs.values()))),
    }


# --------------------------------------------------------------------------- #
#  per-coin static helpers                                                    #
# --------------------------------------------------------------------------- #
def tier_of(coin: str) -> int:
    va = _calib()["vol_adv"]
    adv = va["adv_usd"].get(coin, np.nan) if hasattr(va["adv_usd"], "get") else np.nan
    try:
        adv = float(va.loc[coin, "adv_usd"])
    except Exception:
        adv = np.nan
    if not np.isfinite(adv):
        return 2
    if adv >= TIER_ADV_USD[1]:
        return 1
    if adv >= TIER_ADV_USD[2]:
        return 2
    return 3


def half_spread_bps(coin: str, ts=None) -> tuple[float, bool]:
    hs = _calib()["half_spread"]
    if coin in hs:
        return hs[coin], True
    return TIER_HALF_SPREAD[tier_of(coin)], False


def _sigma_daily(coin: str) -> float:
    c = _calib()
    try:
        v = float(c["vol_adv"].loc[coin, "sigma_daily"])
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return c["sigma_daily_median"]


def _adv_usd(coin: str) -> float:
    c = _calib()
    try:
        v = float(c["vol_adv"].loc[coin, "adv_usd"])
        if np.isfinite(v) and v > 0:
            return v
    except Exception:
        pass
    return TIER_ADV_USD[2]


def impact_bps(coin: str, notional: float) -> float:
    if notional <= 0:
        return 0.0
    sig = _sigma_daily(coin)
    adv = _adv_usd(coin)
    return 1e4 * Y_IMPACT * sig * np.sqrt(max(notional, 0.0) / adv)


def adverse_sel_bps(coin: str) -> float:
    a = _calib()["adverse"]
    if coin in a:
        return max(AS_FLOOR_BPS, a[coin])
    hsb, _ = half_spread_bps(coin)
    return max(AS_FLOOR_BPS, 0.35 * hsb)


def p_fill(coin: str, *, leg: str, notional: float, sigma_1h_ratio: float = 1.0) -> float:
    """queue-depletion fill probability for a passive order at the touch."""
    c = _calib()
    q = c["queue"].get(coin, {})
    tif = TIF_ENTRY_SEC if leg == "entry" else TIF_EXIT_SEC
    # calibrated coins: use their measured entry/exit p_fill, shrunk by size & vol
    key = "p_fill_entry" if leg == "entry" else "p_fill_exit"
    if q.get(key) is not None:
        base = float(q[key])
    else:
        # queue model from near-touch depth + tier arrival proxy
        near = c["near_touch"].get(coin)
        qa = q.get("q_ahead")
        if qa is None and near is not None:
            qa = near * 1e-3                       # L1 ~ 0.1% of the +-1% band
        if qa is None:
            base = TIER_PFILL[tier_of(coin)]
        else:
            adv = _adv_usd(coin)
            lam_notional_per_s = adv / 86400.0     # avg opposing-aggressor $ / s
            avg_trade = max(adv / max(_trade_count(coin), 1.0), 1.0)
            expected_trades = lam_notional_per_s * tif / avg_trade
            need = qa / avg_trade
            base = float(1.0 - _gamma.cdf(need, a=max(expected_trades, 1e-3)))
    # size penalty: a bigger order sits behind more queue
    size_pen = 1.0 / (1.0 + notional / 50_000.0)
    base *= (0.5 + 0.5 * size_pen)
    # volatility penalty (REPRICING_RULE §3)
    base *= np.exp(-2.0 * max(0.0, sigma_1h_ratio - 1.0))
    return float(np.clip(base, 0.05, 0.99))


@lru_cache(maxsize=256)
def _trade_count(coin: str) -> float:
    for f in ("spread_measured.json", "spread_extra.json", "aggtrades_micro.json"):
        p = PROC / f
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            if r.get("symbol") == coin and r.get("n_trades"):
                # normalise to per-day (these are whole-sample counts)
                return float(r["n_trades"]) / 300.0
    return 5000.0


# --------------------------------------------------------------------------- #
#  main API                                                                   #
# --------------------------------------------------------------------------- #
def cost_bps(coin: str, side: str, notional: float, ts=None,
             mode: str = "hybrid", *, leg: str = "entry",
             sigma_1h_ratio: float = 1.0) -> dict:
    """One-way cost of a single fill, in basis points of notional."""
    hsb, hs_measured = half_spread_bps(coin, ts)
    imp = impact_bps(coin, notional)
    tier = tier_of(coin)

    if mode == "taker":
        pf = 0.0
    else:
        pf = p_fill(coin, leg=leg, notional=notional, sigma_1h_ratio=sigma_1h_ratio)

    as_bps = adverse_sel_bps(coin)
    # residual walk-the-book slippage for the part of the order beyond L1 depth
    near = _calib()["near_touch"].get(coin)
    l1 = (near * 1e-3) if near else _adv_usd(coin) * 1e-5
    resid = 0.0 if notional <= l1 else hsb * min(4.0, np.sqrt(notional / max(l1, 1.0)) - 1.0)

    maker_leg = FEE_MAKER_BPS + as_bps - hsb                 # earn the spread if filled
    taker_leg = FEE_TAKER_BPS + hsb + imp + resid
    total = pf * maker_leg + (1.0 - pf) * taker_leg
    total = max(total, FEE_TAKER_BPS * (0.0 if mode != "taker" else 1.0) + 0.1) \
        if mode == "taker" else max(total, FEE_MAKER_BPS + 0.1)

    return {
        "coin": coin, "side": side, "notional": float(notional), "mode": mode,
        "leg": leg, "tier": tier,
        "fee_bps": FEE_MAKER_BPS if (mode != "taker" and pf > 0.5) else FEE_TAKER_BPS,
        "spread_bps": -hsb * pf + hsb * (1 - pf),            # net spread paid
        "impact_bps": float(imp * (1 - pf)),
        "slippage_bps": float(resid * (1 - pf)),
        "adverse_sel_bps": float(as_bps * pf),
        "p_fill": float(pf),
        "half_spread_measured": bool(hs_measured),
        "total_bps": float(total),
    }


def roundtrip_bps(coin: str, notional: float = RESEARCH_NOTIONAL_USD,
                  mode: str = "hybrid", ts=None) -> float:
    e = cost_bps(coin, "buy", notional, ts, mode, leg="entry")["total_bps"]
    x = cost_bps(coin, "sell", notional, ts, mode, leg="exit")["total_bps"]
    return e + x


def cost_oneway_bps_by_coin(coins, notional: float = RESEARCH_NOTIONAL_USD,
                            mode: str = "hybrid") -> pd.Series:
    """per-coin one-way cost (avg of entry & exit legs) for vectorised re-scoring."""
    out = {}
    for c in coins:
        e = cost_bps(c, "buy", notional, None, mode, leg="entry")["total_bps"]
        x = cost_bps(c, "sell", notional, None, mode, leg="exit")["total_bps"]
        out[c] = 0.5 * (e + x)
    return pd.Series(out, dtype=float)


def cost_frac_by_coin(coins, notional: float = RESEARCH_NOTIONAL_USD,
                      mode: str = "hybrid") -> pd.Series:
    """one-way cost as a FRACTION of notional (what backtests multiply turnover by)."""
    return cost_oneway_bps_by_coin(coins, notional, mode) * 1e-4


if __name__ == "__main__":
    import sys
    for c in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "HOOKUSDT", "DARUSDT", "CRVUSDT"):
        print(f"{c:9} tier{tier_of(c)}  hs={half_spread_bps(c)[0]:.2f}  "
              f"hybrid RT @20k = {roundtrip_bps(c):.2f} bps   "
              f"taker RT = {roundtrip_bps(c, mode='taker'):.2f} bps")
    print("\nBTC reconcile (hybrid RT @25k):", round(roundtrip_bps("BTCUSDT", 25_000), 3),
          " target 5.662  band [4.81, 6.51]")
