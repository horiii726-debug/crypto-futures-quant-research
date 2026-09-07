"""Spec H3 — re-score oi_leverage_state on the frozen tradeable universe.

NOT a trial. Same signal, same params (n=168, q=0.30, hold=24h), same dates.
Only the coin set changes: instead of a static hand-picked liquid-28, each bar
uses `lab.exec.universe.tradeable_at(ts)` (rolling, in-sample, no lookahead).

Reports Δ net Sharpe (dynamic universe − static-28) on maker and taker cost.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
from lab.exec import universe as U                              # noqa: E402
from lab.exec.cost_model import cost_frac_by_coin               # noqa: E402
from lab.research.oi_study import _panel, _zx                   # noqa: E402
from lab.research.oi_leverage_hardened import LIQUID as STATIC28, _lev_signal, ANN  # noqa: E402

CFG = dict(n=168, q=0.30, hold=24)


def _sr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1)) if x.size > 5 and x.std(ddof=1) > 0 else 0.0


def _positions(sig, q):
    r = sig.rank(axis=1, pct=True)
    lo = (r >= 1 - q).astype(float); sh = (r <= q).astype(float)
    nl = lo.sum(axis=1).replace(0, np.nan); ns = sh.sum(axis=1).replace(0, np.nan)
    return (lo.div(nl, axis=0) - sh.div(ns, axis=0)).fillna(0.0) / 2.0


def _elig_mask(index, cols) -> pd.DataFrame:
    """bool (T x N): is coin c tradeable as of bar t (checked daily, ffwd)."""
    idx = pd.to_datetime(index)
    days = pd.DatetimeIndex(sorted({d.floor("D") for d in idx}))
    daily = pd.DataFrame(False, index=days, columns=cols)
    for d in days:
        elig = set(U.tradeable_at(d, list(cols)))
        for c in cols:
            daily.loc[d, c] = c in elig
    return daily.reindex(idx.floor("D")).set_index(idx)


def _run(close, qv, oi, mask=None):
    sig = _lev_signal(close, qv, oi, n=CFG["n"])
    sig = sig.where(close.notna().sum(axis=1) >= 8)
    if mask is not None:
        sig = sig.where(mask.values)
    pos = _positions(sig, CFG["q"])
    pos = pos.rolling(CFG["hold"], min_periods=1).mean()
    bwd = close / close.shift(1) - 1.0
    held = pos.shift(2)
    gross = (held * bwd).sum(axis=1)
    turn_coin = held.diff().abs().fillna(held.abs())
    mk = cost_frac_by_coin(list(close.columns), mode="hybrid").reindex(close.columns)
    tk = cost_frac_by_coin(list(close.columns), mode="taker").reindex(close.columns)
    net_mk = (gross - turn_coin.mul(mk.fillna(mk.median()), axis=1).sum(axis=1)).dropna()
    net_tk = (gross - turn_coin.mul(tk.fillna(tk.median()), axis=1).sum(axis=1)).dropna()
    return dict(gross_sr=_sr(gross) * ANN, mk_sr=_sr(net_mk) * ANN, tk_sr=_sr(net_tk) * ANN,
                turn=float(turn_coin.sum(axis=1).mean()),
                n_names=float((held.abs() > 1e-9).sum(axis=1).mean()))


def run() -> dict:
    close, qv, oi = _panel("1h")
    cols = list(close.columns)

    static_cols = [c for c in STATIC28 if c in cols]
    r_static = _run(close[static_cols], qv[static_cols], oi[static_cols])

    mask = _elig_mask(close.index, cols)
    r_dyn = _run(close, qv, oi, mask=mask)

    T = len(close)
    cut = int(T * 0.70)
    m2 = mask.iloc[cut:]
    r_oos_dyn = _run(close.iloc[cut:], qv.iloc[cut:], oi.iloc[cut:], mask=m2)
    r_oos_static = _run(close[static_cols].iloc[cut:], qv[static_cols].iloc[cut:], oi[static_cols].iloc[cut:])

    out = {
        "static_28": r_static, "dynamic_universe": r_dyn,
        "held_out_static": r_oos_static, "held_out_dynamic": r_oos_dyn,
        "delta_full_maker_sr": r_dyn["mk_sr"] - r_static["mk_sr"],
        "delta_full_taker_sr": r_dyn["tk_sr"] - r_static["tk_sr"],
        "delta_heldout_maker_sr": r_oos_dyn["mk_sr"] - r_oos_static["mk_sr"],
        "delta_heldout_taker_sr": r_oos_dyn["tk_sr"] - r_oos_static["tk_sr"],
        "avg_names_static": r_static["n_names"], "avg_names_dynamic": r_dyn["n_names"],
    }
    return out


if __name__ == "__main__":
    r = run()
    md = ["# OI universe re-score (spec H3)", "",
          "_Not a trial. oi_leverage_state, same signal/params/dates. "
          "Coin set: static liquid-28 vs frozen `universe.tradeable_at()` (rolling)._", "",
          "| variant | gross SR | maker net SR | taker net SR | turnover | avg names |",
          "|---|---|---|---|---|---|"]
    for k in ("static_28", "dynamic_universe", "held_out_static", "held_out_dynamic"):
        x = r[k]
        md.append(f"| {k} | {x['gross_sr']:+.2f} | {x['mk_sr']:+.2f} | {x['tk_sr']:+.2f} | "
                  f"{x['turn']:.4f} | {x['n_names']:.1f} |")
    md += ["",
           f"**Δ net Sharpe (dynamic − static):**",
           f"- full sample : maker {r['delta_full_maker_sr']:+.2f}, taker {r['delta_full_taker_sr']:+.2f}",
           f"- held-out 30% : maker {r['delta_heldout_maker_sr']:+.2f}, taker {r['delta_heldout_taker_sr']:+.2f}",
           ""]
    (ROOT / "lab" / "reports" / "OI_UNIVERSE_RESCORE.md").write_text("\n".join(md) + "\n")
    json.dump(r, open(ROOT / "lab" / "reports" / "OI_UNIVERSE_RESCORE.json", "w"), indent=1, default=float)
    print("\n".join(md))
