"""RESEARCH ROUND 2 · P2.1 — IC-decay curve for imbvol_fade.

MEASUREMENT ONLY. Not a trial. Does not touch the ledger.

imbvol_fade (F_BOOK, the strongest raw signal in the whole project, rank-IC 0.021
at 5m gross Sharpe 4.3, killed only by 5-minute turnover):

    sig = -zscore_xsec( imb_vol * sign(Δlog price over k bars) )

Question: does the predictive content survive to a tradeable horizon (>=1h)?
If IC is gone before 1h  -> close it, F_BOOK stays bounded.
If IC persists at >=1h  -> register F_BOOK_SLOW as a new family with its own budget.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "data" / "processed" / "bookdepth_bars"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT",
        "LINKUSDT", "AVAXUSDT", "ADAUSDT", "LTCUSDT", "BCHUSDT", "INJUSDT"]

# base bar = 5 min. horizons in bars:
HORIZONS = {"5m": 1, "15m": 3, "30m": 6, "1h": 12, "4h": 48, "1d": 288}
K_TREND = 3          # the sign() lookback used in book_study.imbvol_fade


def _zx(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def _load():
    price, imb = {}, {}
    from lab.data.ingest import _load_1m
    months = [str(p) for p in pd.period_range("2023-09", "2024-05", freq="M")]
    for s in SYMS:
        p = BOOK / f"{s}_5m.parquet"
        if not p.exists():
            continue
        b = pd.read_parquet(p)
        d = _load_1m(s, months)
        if d.empty:
            continue
        px = d.set_index("dt")["close"].resample("5min", label="left", closed="left").last()
        idx = b.index
        price[s] = px.reindex(idx)
        imb[s] = b["imb_vol"].reindex(idx)
    C = pd.DataFrame(price).sort_index()
    V = pd.DataFrame(imb).reindex(C.index)
    return C, V


def rank_ic(sig, fwd):
    a = sig.rank(axis=1)
    b = fwd.rank(axis=1)
    return float(a.corrwith(b, axis=1).mean(skipna=True))


def run() -> dict:
    C, V = _load()
    sgn = np.sign(np.log(C).diff(K_TREND))
    sig = -_zx(V.rolling(K_TREND, min_periods=1).mean() * sgn)
    valid = C.notna().sum(axis=1) >= 6
    sig = sig.where(valid)

    out = {}
    for name, h in HORIZONS.items():
        fwd = C.shift(-(h + 1)) / C.shift(-1) - 1.0     # R9-aligned: enter t+1, exit t+1+h
        ic = rank_ic(sig, fwd)
        # per-bar IC series for a t-stat (Newey-West-ish via block std)
        s_ic = sig.rank(axis=1).corrwith(fwd.rank(axis=1), axis=1)
        s_ic = s_ic.dropna()
        neff = max(len(s_ic) / h, 5)                    # overlap deflation
        tstat = float(ic / (s_ic.std() / np.sqrt(neff))) if s_ic.std() > 0 else 0.0
        out[name] = {"horizon_bars": h, "rank_ic": ic, "t_stat_overlap_adj": tstat,
                     "n_obs": int(len(s_ic))}
        print(f"  {name:4} (h={h:3} bars)  rank-IC = {ic:+.4f}   t≈{tstat:+.2f}   n={len(s_ic)}",
              flush=True)

    ic_1h = out["1h"]["rank_ic"]
    ic_4h = out["4h"]["rank_ic"]
    verdict = ("PERSISTS_>=1h — register F_BOOK_SLOW as a new family (needs budget)"
               if (abs(ic_1h) >= 0.010 and abs(out["1h"]["t_stat_overlap_adj"]) >= 2.0)
               else "DECAYS_<1h — imbvol_fade is a sub-1h microstructure effect; F_BOOK stays bounded")
    out["_verdict"] = verdict
    print(f"\n  1h IC {ic_1h:+.4f}  4h IC {ic_4h:+.4f}")
    print(f"  VERDICT: {verdict}")
    return out


if __name__ == "__main__":
    r = run()
    json.dump(r, open(ROOT / "lab" / "reports" / "IMBVOL_DECAY.json", "w"), indent=1, default=float)
    print("\nwrote lab/reports/IMBVOL_DECAY.json")
