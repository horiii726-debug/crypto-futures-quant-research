"""S1 ingest: 1m klines -> quality gate -> resampled panels -> chronological
partitions -> hashes in the ledger.

Quality gate (data-warden veto surface):
  - timestamp monotonic, on the 1-minute grid, no duplicate bars
  - gap accounting (missing minutes) with maintenance-window classification
  - OHLC consistency (h >= max(o,c,l), l <= min(o,c,h), all > 0)
  - stale runs (identical OHLCV repeated)
  - return outliers (|1m log-ret| > OUTLIER_SIGMA robust sigma)
  - listing/delisting edges (leading/trailing all-NaN)
"""
from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lab.data.binance_vision as bv  # noqa

PROC = ROOT / "data" / "processed"
FEAT = ROOT / "data" / "features"
OUTLIER_SIGMA = 12.0
HORIZONS = {"1h": 60, "4h": 240, "1d": 1440}


def month_range(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start, end, freq="M")]


# ---- per-symbol quality gate ----------------------------------------------

def quality_gate(sym: str, df1m: pd.DataFrame) -> dict:
    if df1m.empty:
        return {"symbol": sym, "status": "EMPTY", "n": 0}
    d = df1m.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    t = d["open_time"].to_numpy()
    dt = pd.to_datetime(t, unit="ms", utc=True)
    full = pd.date_range(dt.min(), dt.max(), freq="1min", tz="UTC")
    present = pd.DatetimeIndex(dt)
    missing = full.difference(present)
    # gaps -> contiguous runs
    gap_runs = []
    if len(missing):
        mi = missing.astype("int64") // 60_000_000_000
        brk = np.where(np.diff(mi) > 1)[0]
        starts = np.concatenate([[0], brk + 1])
        ends = np.concatenate([brk, [len(mi) - 1]])
        for s, e in zip(starts, ends):
            gap_runs.append((str(missing[s]), int(mi[e] - mi[s] + 1)))
    o, h, l, c = (d[x].to_numpy(float) for x in ["open", "high", "low", "close"])
    v = d["volume"].to_numpy(float)
    ohlc_bad = int(np.sum((h < np.maximum.reduce([o, c, l])) |
                          (l > np.minimum.reduce([o, c, h])) |
                          (o <= 0) | (c <= 0) | (h <= 0) | (l <= 0)))
    logret = np.zeros(len(c))
    logret[1:] = np.log(c[1:] / c[:-1])
    rs = 1.4826 * np.median(np.abs(logret - np.median(logret))) + 1e-12
    outliers = int(np.sum(np.abs(logret) > OUTLIER_SIGMA * rs))
    stale = int(np.sum((np.diff(c) == 0) & (np.diff(v) == 0) & (np.diff(o) == 0)))
    big_gaps = [g for g in gap_runs if g[1] >= 120]  # >= 2h likely maintenance/halt
    return {
        "symbol": sym, "status": "OK", "n": int(len(d)),
        "t_start": str(dt.min()), "t_end": str(dt.max()),
        "missing_minutes": int(len(missing)),
        "missing_pct": round(100 * len(missing) / max(len(full), 1), 4),
        "n_gap_runs": len(gap_runs),
        "n_gap_runs_ge_2h": len(big_gaps),
        "largest_gap_min": max([g[1] for g in gap_runs], default=0),
        "ohlc_violations": ohlc_bad,
        "return_outliers": outliers,
        "stale_bars": stale,
        "sample_big_gaps": big_gaps[:5],
    }


def _load_1m(sym: str, months: list[str]) -> pd.DataFrame:
    import json as _json
    cache = PROC / "klines_1m" / f"{sym}.parquet"
    miss_f = PROC / "klines_1m" / f"{sym}.absent.json"
    absent = set(_json.loads(miss_f.read_text())) if miss_f.exists() else set()
    if cache.exists():
        d = pd.read_parquet(cache)
        have = set(d["month"].unique()) if "month" in d else set()
        need = [m for m in months if m not in have and m not in absent]
        if not need:
            return d[d["month"].isin(months)] if "month" in d else d
        got = bv.klines(sym, need, interval="1m", freq="monthly")
        if not got.empty:
            got["month"] = got["dt"].dt.strftime("%Y-%m")
            new_have = set(got["month"].unique())
        else:
            new_have = set()
        # record months we asked for but did not get (delisted / not yet listed)
        newly_absent = [m for m in need if m not in new_have]
        if newly_absent:
            miss_f.write_text(_json.dumps(sorted(absent | set(newly_absent))))
        if not got.empty:
            keep = ["open_time", "dt", "month", "open", "high", "low", "close", "volume",
                    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
            d = pd.concat([d, got[keep]], ignore_index=True).drop_duplicates("open_time") \
                .sort_values("open_time").reset_index(drop=True)
            d.to_parquet(cache)
        return d[d["month"].isin(months)] if "month" in d else d
    df = bv.klines(sym, [m for m in months if m not in absent], interval="1m", freq="monthly")
    if df.empty:
        return df
    df["month"] = df["dt"].dt.strftime("%Y-%m")
    got_months = set(df["month"].unique())
    newly_absent = [m for m in months if m not in got_months and m not in absent]
    if newly_absent:
        miss_f.write_text(_json.dumps(sorted(absent | set(newly_absent))))
    keep = ["open_time", "dt", "month", "open", "high", "low", "close", "volume",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
    df = df[keep]
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def resample(df1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    g = df1m.set_index("dt")
    rule = f"{minutes}min"
    out = pd.DataFrame({
        "open": g["open"].resample(rule, label="left", closed="left").first(),
        "high": g["high"].resample(rule, label="left", closed="left").max(),
        "low": g["low"].resample(rule, label="left", closed="left").min(),
        "close": g["close"].resample(rule, label="left", closed="left").last(),
        "volume": g["volume"].resample(rule, label="left", closed="left").sum(),
        "quote_volume": g["quote_volume"].resample(rule, label="left", closed="left").sum(),
        "trades": g["trades"].resample(rule, label="left", closed="left").sum(),
        "taker_buy_base": g["taker_buy_base"].resample(rule, label="left", closed="left").sum(),
    })
    return out.dropna(subset=["close"])


def build_panels(symbols: list[str], months: list[str], tag: str,
                 max_missing_pct: float = 5.0) -> dict:
    qc_rows, per_h = [], {h: {} for h in HORIZONS}
    kept, dropped = [], []
    for sym in symbols:
        df = _load_1m(sym, months)
        qc = quality_gate(sym, df)
        qc_rows.append(qc)
        if qc["status"] != "OK" or qc["missing_pct"] > max_missing_pct or qc["ohlc_violations"] > 0:
            dropped.append((sym, qc.get("status"), qc.get("missing_pct"), qc.get("ohlc_violations")))
            continue
        kept.append(sym)
        for h, m in HORIZONS.items():
            per_h[h][sym] = resample(df, m)
    panels = {}
    for h in HORIZONS:
        if not per_h[h]:
            continue
        close = pd.DataFrame({s: v["close"] for s, v in per_h[h].items()}).sort_index()
        qv = pd.DataFrame({s: v["quote_volume"] for s, v in per_h[h].items()}).reindex(close.index)
        tbb = pd.DataFrame({s: v["taker_buy_base"] for s, v in per_h[h].items()}).reindex(close.index)
        vol = pd.DataFrame({s: v["volume"] for s, v in per_h[h].items()}).reindex(close.index)
        hi = pd.DataFrame({s: v["high"] for s, v in per_h[h].items()}).reindex(close.index)
        lo = pd.DataFrame({s: v["low"] for s, v in per_h[h].items()}).reindex(close.index)
        out = PROC / "panels" / tag / h
        out.mkdir(parents=True, exist_ok=True)
        for nm, fr in [("close", close), ("quote_volume", qv), ("taker_buy_base", tbb),
                       ("volume", vol), ("high", hi), ("low", lo)]:
            fr.to_parquet(out / f"{nm}.parquet")
        panels[h] = {"shape": list(close.shape), "path": str(out),
                     "t_start": str(close.index.min()), "t_end": str(close.index.max())}
    (PROC / "panels" / tag / "quality.json").write_text(json.dumps(qc_rows, indent=1))
    return {"tag": tag, "kept": kept, "n_kept": len(kept), "dropped": dropped,
            "panels": panels, "quality_rows": qc_rows}


# ---- chronological partitions + hashing ----------------------------------

def partition_and_hash(tag: str, splits: dict, ledger) -> dict:
    """splits = {'train': ('2023-09-01','2025-02-28'), 'valid': (...), 'test': (...)}"""
    res = {}
    base = PROC / "panels" / tag
    for part, (lo, hi) in splits.items():
        lo_t = pd.Timestamp(lo, tz="UTC")
        hi_t = pd.Timestamp(hi, tz="UTC") + pd.Timedelta(days=1)
        for h in HORIZONS:
            src = base / h / "close.parquet"
            if not src.exists():
                continue
            close = pd.read_parquet(src)
            seg = close[(close.index >= lo_t) & (close.index < hi_t)]
            dst = ROOT / "data" / part / tag / h
            dst.mkdir(parents=True, exist_ok=True)
            for nm in ["close", "quote_volume", "taker_buy_base", "volume", "high", "low"]:
                fr = pd.read_parquet(base / h / f"{nm}.parquet")
                fr[(fr.index >= lo_t) & (fr.index < hi_t)].to_parquet(dst / f"{nm}.parquet")
            blob = pd.util.hash_pandas_object(seg, index=True).values.tobytes()
            sha = hashlib.sha256(blob).hexdigest()
            ledger.add_dataset(part, f"{tag}:{h}", sha, n_rows=int(seg.shape[0] * seg.shape[1]),
                               t_start=str(seg.index.min()), t_end=str(seg.index.max()),
                               payload={"tag": tag, "horizon": h, "n_bars": int(seg.shape[0]),
                                        "n_symbols": int(seg.shape[1])})
            res[f"{part}:{h}"] = {"sha256": sha, "bars": int(seg.shape[0]),
                                  "symbols": int(seg.shape[1]),
                                  "range": [str(seg.index.min()), str(seg.index.max())]}
    return res


if __name__ == "__main__":
    print("ingest module ready")
