"""Spec J4 — re-run oi_leverage_state on Binance OI data, 2021-12 .. 2023-09.

NOT a trial. Same signal, same params (n=168, q=0.30, hold=24h). Only the data
source and the time window change:
  - OI from Binance metrics (`data/processed/binance_oi_hist`), USD, hourly
  - period 2021-12 .. 2023-09 = the LUNA (2022-05) + FTX (2022-11) bear, a
    regime the Bybit 2023-2025 sample never saw
  - universe = HIST_UNIVERSE ∩ frozen liquidity filter at each bar

If the signal holds here → cross-venue + cross-regime out-of-sample evidence,
DSR rises legitimately. If it fails → oi_leverage_state is Bybit/regime-specific;
that is a finding, recorded, not hidden.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MV = ROOT / "data" / "processed" / "multi_venue"
from lab.exec import universe as U                              # noqa: E402
from lab.exec.cost_model import cost_frac_by_coin               # noqa: E402
from lab.stats.dsr import deflated_sharpe                        # noqa: E402

CFG = dict(n=168, q=0.30, hold=24)
ANN = np.sqrt(365 * 24)


def _zx(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def _sr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1)) if x.size > 5 and x.std(ddof=1) > 0 else 0.0


def _positions(sig, q):
    r = sig.rank(axis=1, pct=True)
    lo = (r >= 1 - q).astype(float); sh = (r <= q).astype(float)
    nl = lo.sum(axis=1).replace(0, np.nan); ns = sh.sum(axis=1).replace(0, np.nan)
    return (lo.div(nl, axis=0) - sh.div(ns, axis=0)).fillna(0.0) / 2.0


def _elig_mask(index, cols):
    idx = pd.to_datetime(index)
    days = pd.DatetimeIndex(sorted({d.floor("D") for d in idx}))
    daily = pd.DataFrame(False, index=days, columns=cols)
    for d in days:
        e = set(U.tradeable_at(d, list(cols)))
        daily.loc[d] = [c in e for c in cols]
    return daily.reindex(idx.floor("D")).set_index(idx)


def run(use_filter=True, oi_validated_only=True) -> dict:
    C = pd.read_parquet(MV / "close_1h.parquet")
    QV = pd.read_parquet(MV / "quote_volume_1h.parquet")
    OI = pd.read_parquet(MV / "oi_usd_1h.parquet")
    common = [c for c in C.columns if c in OI.columns and OI[c].notna().sum() > 2000]
    if oi_validated_only:
        try:
            val = json.loads((MV / "_validation.json").read_text())
            ok = {c for c, r in val.items() if r.get("pass")}
            common = [c for c in common if c in ok]
        except Exception:
            pass
    C, QV, OI = C[common], QV[common], OI[common]

    lev = OI * C / QV.rolling(CFG["n"], min_periods=CFG["n"] // 2).mean().replace(0, np.nan)
    sig = -_zx(np.log(lev))
    sig = sig.where(C.notna().sum(axis=1) >= 6)
    if use_filter:
        m = _elig_mask(C.index, common)
        sig = sig.where(m.values)

    pos = _positions(sig, CFG["q"]).rolling(CFG["hold"], min_periods=1).mean()
    bwd = C / C.shift(1) - 1.0
    held = pos.shift(2)
    gross = (held * bwd).sum(axis=1)
    turn_coin = held.diff().abs().fillna(held.abs())
    mk = cost_frac_by_coin(list(C.columns), mode="hybrid").reindex(C.columns)
    tk = cost_frac_by_coin(list(C.columns), mode="taker").reindex(C.columns)
    net_mk = (gross - turn_coin.mul(mk.fillna(mk.median()), axis=1).sum(axis=1)).dropna()
    net_tk = (gross - turn_coin.mul(tk.fillna(tk.median()), axis=1).sum(axis=1)).dropna()

    idx = pd.to_datetime(net_mk.index)
    # sub-regime windows
    def win(a, b):
        s = net_mk[(idx >= a) & (idx < b)]
        return round(_sr(s) * ANN, 2), round(float(s.sum()), 4), int(len(s))

    rng = np.random.default_rng(12345)
    v = net_mk.values
    plc = np.array([_sr(v * rng.choice([-1.0, 1.0], size=len(v))) for _ in range(300)]) * ANN
    placebo_p = float((np.sum(plc >= _sr(net_mk) * ANN) + 1) / 301)
    T = len(net_mk)
    sur = np.array([_sr(v[np.concatenate([np.arange(s, s + 48) % T
                    for s in rng.integers(0, T, T // 48 + 1)])[:T]]) for _ in range(300)]) * ANN
    surrogate_p = float((np.sum(sur >= _sr(net_mk) * ANN) + 1) / 301)

    n4 = T // 4
    wf = [round(_sr(net_mk.iloc[k * n4:(k + 1) * n4]) * ANN, 2) for k in range(4)]

    out = {
        "period": [str(idx[0]), str(idx[-1])], "n_bars": T, "n_coins": len(common),
        "coins": common,
        "gross_sr_ann": round(_sr(gross) * ANN, 2),
        "maker_sr_ann": round(_sr(net_mk) * ANN, 2),
        "taker_sr_ann": round(_sr(net_tk) * ANN, 2),
        "turnover_per_bar": round(float(turn_coin.sum(axis=1).mean()), 4),
        "walk_forward": wf, "placebo_p": placebo_p, "surrogate_p": surrogate_p,
        "luna_2022H1": win("2022-01-01", "2022-07-01"),
        "ftx_2022H2": win("2022-07-01", "2023-01-01"),
        "recovery_2023": win("2023-01-01", "2023-10-01"),
        "max_drawdown": round(float(((1 + net_mk).cumprod() / (1 + net_mk).cumprod().cummax() - 1).min()), 4),
    }
    return out


if __name__ == "__main__":
    # the frozen liquidity filter has no data before 2023-09 (klines_1m starts
    # then), so for the 2021-2022 window the HIST_UNIVERSE (hand-picked liquid
    # majors) is used directly — use_filter=False.
    r = run(use_filter=False, oi_validated_only=True)
    r_all = run(use_filter=False, oi_validated_only=False)
    r["also_all23_coins"] = {k: r_all[k] for k in
                             ("maker_sr_ann", "taker_sr_ann", "gross_sr_ann", "n_coins",
                              "walk_forward", "surrogate_p")}
    verdict = ("HOLDS cross-venue + cross-regime — evidence strengthened"
               if (r["maker_sr_ann"] > 0.5 and r["surrogate_p"] < 0.10
                   and all(x > -0.5 for x in r["walk_forward"]))
               else "FAILS on Binance 2021-2023 — oi_leverage_state is Bybit/regime-specific (a finding)")
    r["verdict"] = verdict
    md = ["# oi_leverage_state — cross-venue / cross-regime re-score (spec J4)", "",
          f"_Binance metrics OI, {r['period'][0][:10]} .. {r['period'][1][:10]}, "
          f"{r['n_coins']} coins. Same signal, same params. Not a trial._", "",
          f"| | Sharpe(ann) | P&L | n_bar |", "|---|---|---|---|",
          f"| **full window** | maker **{r['maker_sr_ann']}** / taker {r['taker_sr_ann']} / gross {r['gross_sr_ann']} | | {r['n_bars']} |",
          f"| LUNA 2022 H1 | {r['luna_2022H1'][0]} | {r['luna_2022H1'][1]} | {r['luna_2022H1'][2]} |",
          f"| FTX 2022 H2 | {r['ftx_2022H2'][0]} | {r['ftx_2022H2'][1]} | {r['ftx_2022H2'][2]} |",
          f"| recovery 2023 | {r['recovery_2023'][0]} | {r['recovery_2023'][1]} | {r['recovery_2023'][2]} |",
          "",
          f"- walk-forward (4 windows): {r['walk_forward']}",
          f"- turnover/bar: {r['turnover_per_bar']}   max DD: {r['max_drawdown']}",
          f"- placebo p: {r['placebo_p']:.3f}   surrogate p: {r['surrogate_p']:.3f}",
          "", f"## → {verdict}", ""]
    (ROOT / "lab" / "reports" / "OI_CROSSVENUE.md").write_text("\n".join(md) + "\n")
    json.dump(r, open(ROOT / "lab" / "reports" / "OI_CROSSVENUE.json", "w"), indent=1, default=float)
    print("\n".join(md))
