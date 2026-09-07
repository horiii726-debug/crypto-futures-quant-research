"""Spec G — regime attribution for oi_leverage_state.

NOT a trial. Splits the existing P&L by regime bucket. If the profit is even
across buckets the edge is regime-robust; if it concentrates in one bucket it is
a regime bet, not an edge — recorded, not hidden.

Buckets (evaluated each 1h bar, state from a long window, all causal):
  btc_trend  = sign(200-bar slope of BTC daily close)  -> up / down / flat
  vol_bucket = tercile of BTC 30d realised vol          -> low / mid / high
  fund_sign  = sign(7d mean cross-sectional funding)    -> pos / neg
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
from lab.research.oi_study import _panel                         # noqa: E402
from lab.research.oi_leverage_hardened import LIQUID, _run_cfg, ANN  # noqa: E402

CFG = dict(n=168, q=0.30, hold=24)


def _sr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1)) if x.size > 5 and x.std(ddof=1) > 0 else 0.0


def _mdd(pnl: pd.Series) -> float:
    eq = (1 + pnl).cumprod()
    return float((eq / eq.cummax() - 1).min())


def run() -> dict:
    close, qv, oi = _panel("1h")
    liq = [c for c in LIQUID if c in close.columns]
    r = _run_cfg(close[liq], qv[liq], oi[liq], CFG["n"], CFG["q"], CFG["hold"])
    pnl = r["net_mk"].copy()
    pnl.index = pd.to_datetime(pnl.index)
    turn = r["turn"].reindex(pnl.index).fillna(0.0)

    # --- regime states, all trailing / causal --------------------------------
    btc = close["BTCUSDT"].copy()
    btc.index = pd.to_datetime(btc.index)
    btc_d = btc.resample("1D").last()
    slope = btc_d.rolling(200).apply(lambda w: np.polyfit(np.arange(len(w)), np.log(w), 1)[0]
                                     if w.notna().all() else np.nan, raw=False)
    thr = slope.abs().median()
    btc_trend_d = slope.apply(lambda s: "flat" if not np.isfinite(s) or abs(s) < 0.3 * thr
                              else ("up" if s > 0 else "down"))
    rv_d = np.log(btc_d).diff().rolling(30).std() * np.sqrt(365)
    q1, q2 = rv_d.quantile([1 / 3, 2 / 3])
    vol_bucket_d = rv_d.apply(lambda v: "low" if v <= q1 else ("high" if v > q2 else "mid"))

    bdir = ROOT / "data" / "processed" / "basis"
    fcols = {}
    for c in liq:
        p = bdir / f"{c}.parquet"
        if p.exists():
            s = pd.read_parquet(p)["funding"]
            s.index = pd.to_datetime(s.index)
            fcols[c] = s
    if fcols:
        fx = pd.DataFrame(fcols).sort_index()
        xmean = fx.mean(axis=1).rolling(24 * 7, min_periods=24).mean()   # 7d mean, hourly grid
        fund_sign_h = np.sign(xmean).reindex(pnl.index, method="ffill")
    else:
        fund_sign_h = pd.Series(np.nan, index=pnl.index)

    day = pnl.index.floor("D")
    bt = btc_trend_d.reindex(day).values
    vb = vol_bucket_d.reindex(day).values
    fs = pd.Series(fund_sign_h.values, index=pnl.index)
    fs = fs.apply(lambda x: "pos" if x > 0 else ("neg" if x < 0 else "na")).values

    df = pd.DataFrame({"pnl": pnl.values, "turn": turn.values,
                       "btc_trend": bt, "vol_bucket": vb, "fund_sign": fs}, index=pnl.index)

    out = {"overall": {"n_bar": len(df), "sharpe_ann": round(_sr(df.pnl) * ANN, 2),
                       "pnl": round(float(df.pnl.sum()), 4), "max_dd": round(_mdd(df.pnl), 4),
                       "turnover": round(float(df.turn.mean()), 4)}}
    tables = {}
    for dim in ("btc_trend", "vol_bucket", "fund_sign"):
        rows = []
        for b, g in df.groupby(dim):
            rows.append({"bucket": b, "n_bar": len(g), "frac": round(len(g) / len(df), 2),
                         "sharpe_ann": round(_sr(g.pnl) * ANN, 2),
                         "pnl": round(float(g.pnl.sum()), 4),
                         "pnl_share": round(float(g.pnl.sum() / df.pnl.sum()), 2) if df.pnl.sum() else None,
                         "max_dd": round(_mdd(g.pnl), 4),
                         "turnover": round(float(g.turn.mean()), 4)})
        tables[dim] = sorted(rows, key=lambda r: -r["pnl"])
    out["by_regime"] = tables

    # verdict per dimension
    verdicts = {}
    for dim, rows in tables.items():
        rr = [r for r in rows if r["bucket"] != "na" and r["frac"] >= 0.03]
        # concentration = P&L share relative to time share; ~1 means even
        conc = [(r["bucket"], (r["pnl_share"] or 0) / max(r["frac"], 1e-6)) for r in rr]
        weak = [b for b, c in conc if c < 0.35]                  # <35% of its fair share
        neg = [r["bucket"] for r in rr if r["sharpe_ann"] < 0]
        hot = [(b, c) for b, c in conc if c > 2.5]
        if neg:
            v = f"NEGATIVE Sharpe in {neg} — candidate regime FILTER (a filter = new pre-registered trial)"
        elif hot and len(rr) > 1:
            v = f"CONCENTRATED — {hot[0][0]} earns {hot[0][1]:.1f}x its time-share of P&L (leans on this regime)"
        elif weak:
            v = f"WEAK in {weak} (Sharpe there ~0) — edge needs the other regimes; still net-positive everywhere"
        else:
            v = "EVEN — P&L tracks time spent in each bucket; regime-robust on this dimension"
        verdicts[dim] = v
    out["verdict"] = verdicts
    return out


if __name__ == "__main__":
    r = run()
    md = ["# REGIME_ATTRIB — oi_leverage_state (spec G)", "",
          "_Not a trial. Existing P&L split by regime bucket (all states causal)._", "",
          f"**Overall:** {r['overall']['n_bar']} bars, Sharpe {r['overall']['sharpe_ann']}, "
          f"P&L {r['overall']['pnl']}, maxDD {r['overall']['max_dd']}, turnover {r['overall']['turnover']}", ""]
    for dim, rows in r["by_regime"].items():
        md += [f"## {dim}", "", "| bucket | n_bar | frac | Sharpe(ann) | P&L | P&L share | maxDD | turnover |",
               "|---|---|---|---|---|---|---|---|"]
        for x in rows:
            md.append(f"| {x['bucket']} | {x['n_bar']} | {x['frac']} | {x['sharpe_ann']} | "
                      f"{x['pnl']} | {x['pnl_share']} | {x['max_dd']} | {x['turnover']} |")
        md += ["", f"→ **{r['verdict'][dim]}**", ""]
    (ROOT / "lab" / "reports" / "REGIME_ATTRIB.md").write_text("\n".join(md) + "\n")
    json.dump(r, open(ROOT / "lab" / "reports" / "REGIME_ATTRIB.json", "w"), indent=1, default=float)
    print("\n".join(md))
