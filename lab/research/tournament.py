"""Division tournament — every formula run through the triple-barrier
event backtest with per-coin cost. Ranks within a division, then across.

Ranking metric is the per-trade net expectancy with its t-statistic; a formula
must clear BOTH an economic bar (expectancy > 0 net of cost) and a statistical
bar (t > 2) before it is a division champion.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
from lab.divisions.base import DIVISIONS, load_panel, load_micro   # noqa: E402
from lab.divisions import (d1_trend, d2_vol, d3_dom, d4_tape,      # noqa: E402,F401
                           d5_oi, d6_fund, d7_volume, d8_regime)
from lab.engine.barriers import BarrierConfig, sigma_blend          # noqa: E402
from lab.engine.event_backtest import (EventConfig, run_event_backtest,  # noqa: E402
                                       trade_stats)
from lab.exec.cost_model import cost_frac_by_coin                   # noqa: E402

MICRO_DIVS = {"D3_DOM", "D4_TAPE"}


def _tstat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 20 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / x.std(ddof=1) * np.sqrt(x.size))


def _panel_for(div: str, bar_min: int = 15):
    """Micro divisions run on 15m bars but with a MULTI-DAY barrier, so the
    holding period constraint still holds (min 24h = 96 bars at 15m)."""
    if div in MICRO_DIVS:
        m = load_micro(bar_min)
        if not m:
            return None, None
        px = {"close": m["close"], "high": m["high"], "low": m["low"],
              "open": m["close"].shift(1)}
        return {**m, **px}, bar_min / 60.0
    return load_panel("H1"), 1.0


def run_division(div: str, *, entry_z=1.5, cost_mode="hybrid",
                 pt=1.0, sl=1.0, hold_hours=168, min_hold_hours=24,
                 max_concurrent=8, verbose=True) -> list[dict]:
    panel, bar_h = _panel_for(div)
    if panel is None:
        return []
    hold_bars = max(int(round(hold_hours / bar_h)), 4)
    min_bars = max(int(round(min_hold_hours / bar_h)), 2)
    sig_win = max(int(round(168 / bar_h)), 20)

    sg = sigma_blend(panel["open"], panel["high"], panel["low"], panel["close"], sig_win)
    cols = list(panel["close"].columns)
    cost = cost_frac_by_coin(cols, mode=cost_mode) * 2.0        # ROUND trip
    cfg = EventConfig(entry_z=entry_z, max_concurrent=max_concurrent,
                      barrier=BarrierConfig(pt_mult=pt, sl_mult=sl,
                                            max_hold_bars=hold_bars,
                                            min_hold_bars=min_bars,
                                            sigma_window=sig_win))
    out = []
    for fid, meta in DIVISIONS.get(div, {}).items():
        t0 = time.time()
        try:
            sig = meta["fn"](panel, **meta["params"])
        except Exception as e:
            out.append({"division": div, "formula": fid, "paper": meta["paper"],
                        "status": f"ERR {e!r}"[:80]})
            continue
        if sig is None or not isinstance(sig, pd.DataFrame) or sig.dropna(how="all").empty:
            out.append({"division": div, "formula": fid, "paper": meta["paper"],
                        "status": "empty_signal"})
            continue
        r = run_event_backtest(panel, sig, cfg, cost_frac=cost, sigma=sg)
        if r["n_trades"] < 30:
            out.append({"division": div, "formula": fid, "paper": meta["paper"],
                        "status": f"too_few_trades({r['n_trades']})",
                        "n_trades": r["n_trades"]})
            if verbose:
                print(f"  {fid:22} {r['status']:22} n={r['n_trades']}", flush=True)
            continue
        st = trade_stats(r["trades"], bar_h, capital_slots=max_concurrent)
        t = _tstat(r["trades"]["ret_net"].to_numpy())
        rec = {"division": div, "formula": fid, "paper": meta["paper"],
               "params": meta["params"], "status": "ok", "t_stat": round(t, 3),
               "secs": round(time.time() - t0, 1), **st}
        out.append(rec)
        if verbose:
            print(f"  {fid:22} n={st['n_trades']:5} win={st['win_rate']:.3f} "
                  f"exp={st['expectancy_bps']:+8.1f}bps t={t:+6.2f} SR={st['sharpe_ann']} "
                  f"hold={st['avg_hold_hours']:.0f}h "
                  f"TP/SL/T={st['tp_hits']}/{st['sl_hits']}/{st['time_exits']}", flush=True)
    return out


def rank(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([r for r in results if r.get("status") == "ok"])
    if df.empty:
        return df
    df["champion"] = (df["expectancy_bps"] > 0) & (df["t_stat"] > 2.0) & (df["n_trades"] >= 50)
    return df.sort_values(["division", "expectancy_bps"], ascending=[True, False])


ALL_DIVISIONS = ["D1_TREND", "D2_VOL", "D5_OI", "D6_FUND", "D7_VOLUME",
                 "D8_REGIME", "D3_DOM", "D4_TAPE"]


if __name__ == "__main__":
    import sys
    divs = sys.argv[1:] or ALL_DIVISIONS
    allr = []
    for d in divs:
        print(f"\n=== {d} ({len(DIVISIONS.get(d, {}))} formulas) ===", flush=True)
        allr += run_division(d)
    df = rank(allr)
    (ROOT / "lab" / "reports").mkdir(parents=True, exist_ok=True)
    json.dump(allr, open(ROOT / "lab" / "reports" / "TOURNAMENT_RAW.json", "w"),
              indent=1, default=float)
    if not df.empty:
        df.to_csv(ROOT / "lab" / "reports" / "TOURNAMENT.csv", index=False)
        print("\n=== CHAMPIONS (expectancy>0, t>2, n>=50) ===")
        ch = df[df["champion"]]
        if ch.empty:
            print("  none")
        for r in ch.itertuples():
            print(f"  {r.division:10} {r.formula:22} exp={r.expectancy_bps:+8.1f}bps "
                  f"t={r.t_stat:+.2f} win={r.win_rate:.3f} n={r.n_trades} SR={r.sharpe_ann}")
        print(f"\n{len(df)} formulas scored, {int(df['champion'].sum())} champions")
