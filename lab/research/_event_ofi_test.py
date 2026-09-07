"""Event-level OFI test: does trailing order-flow imbalance over a short
window predict the price move over the next window, held for exactly the
signal half-life (not a full bar)?

Resamples the real tape to a 10s grid, computes OFI over trailing W seconds,
predicts the forward H-second return. Cost: maker (rest, adverse-selected) and
taker cross-check. If there is no edge here either, F_MICRO_OFI is bounded at
every accessible timescale.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from numba import njit
from lab.ledger import Ledger
from lab.data.tape import _read_day
from lab.stats.dsr import deflated_sharpe


@njit(cache=False)
def _grid_ofi(g, price, sign, notion, t0, nb):
    ofi = np.zeros(nb); last = np.full(nb, np.nan); cnt = np.zeros(nb)
    for i in range(g.shape[0]):
        b = int(g[i] - t0)
        if 0 <= b < nb:
            ofi[b] += sign[i] * notion[i]; last[b] = price[i]; cnt[b] += 1.0
    return ofi, last, cnt

GRID_S = 10                          # 10-second bars
MAKER_ONEWAY = 3.0e-4               # ~3 bps one-way (touch maker + adverse sel floor)
TAKER_ONEWAY = 5.0e-4 + 1.0e-4     # fee + half spread


def grid_bars(sym, days):
    rows = []
    for d in days:
        da = _read_day(sym, d)
        if da is None:
            continue
        ts, price, qty, sell_aggr = da
        notion = qty * price
        sign = np.where(sell_aggr, -1.0, 1.0)
        g = (ts // (GRID_S * 1000)).astype(np.int64)
        nb = int(g[-1] - g[0]) + 1
        t0 = g[0]
        ofi, last, cnt = _grid_ofi(np.ascontiguousarray(g), np.ascontiguousarray(price),
                                   np.ascontiguousarray(sign), np.ascontiguousarray(notion),
                                   int(t0), int(nb))
        idx = pd.to_datetime((t0 + np.arange(nb)) * GRID_S * 1000, unit="ms", utc=True)
        rows.append(pd.DataFrame({"ofi": ofi, "px": last, "n": cnt}, index=idx))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["px"] = df["px"].ffill()
    return df


def test(sym, days, W_bars, H_bars, ledger, exp_id):
    df = grid_bars(sym, days)
    if len(df) < 5000:
        return None
    o = df["ofi"].rolling(W_bars, min_periods=W_bars // 2).sum()
    oz = (o - o.rolling(2000, min_periods=200).mean()) / o.rolling(2000, min_periods=200).std()
    r = np.log(df["px"]).diff()
    fwd = np.log(df["px"].shift(-H_bars) / df["px"])            # forward H-bar return
    sig = np.tanh(0.7 * oz).shift(2)                             # decide t -> act t+2 grid bars (~20s)
    m = np.isfinite(sig) & np.isfinite(fwd)
    ic = float(np.corrcoef(sig[m], fwd[m])[0, 1])
    # backtest: hold sig for H bars, pnl per bar = sig_lagged * r ; turnover per bar
    pos = sig / max(H_bars, 1)                                   # spread the hold
    pos = pos.rolling(H_bars, min_periods=1).sum().clip(-1, 1)
    held = pos.shift(1)
    gross = (held * r).dropna()
    turn = held.diff().abs().dropna()
    net_mk = (gross - turn.reindex_like(gross) * MAKER_ONEWAY)
    net_tk = (gross - turn.reindex_like(gross) * TAKER_ONEWAY)
    bpy = 365 * 24 * 3600 / GRID_S
    sr = lambda x: float(x.mean() / x.std(ddof=1) * np.sqrt(bpy)) if x.std(ddof=1) > 0 else 0.0
    dsr = deflated_sharpe(net_mk.values, ledger=ledger, family="F_MICRO_OFI_EVENT")
    return {"sym": sym, "W_s": W_bars * GRID_S, "H_s": H_bars * GRID_S,
            "ic": ic, "gross_sr": sr(gross), "maker_sr": sr(net_mk), "taker_sr": sr(net_tk),
            "gross_edge_bps_per_bar": float(gross.mean() * 1e4),
            "cost_bps_per_bar": float((turn.mean() * MAKER_ONEWAY) * 1e4),
            "turn_per_bar": float(turn.mean()), "dsr": dsr["dsr"], "n_bars": int(m.sum())}


def main():
    L = Ledger()
    hyp = L.add_hypothesis({
        "claim": "trailing OFI over W seconds predicts the next H-second return, "
                 "held for the signal half-life",
        "mechanism": "aggressive flow pushes price and continues briefly (Cont-Kukanov-Stoikov)",
        "math_form": "sig = tanh(z(sum OFI over W)) ; hold H", "target": "fwd_Hs_return",
        "horizon": "seconds", "null_hypothesis": "IC=0 ; maker net Sharpe <= 0",
        "family": "F_MICRO_OFI_EVENT", "lessons_reviewed": True})
    exp = L.new_experiment(hyp, "F_MICRO_OFI_EVENT",
                           {"grid_s": GRID_S, "note": "3 coins x 30 sampled days"}, stage="discovery")
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range("2025-01-06", "2025-02-04")]
    rows = []
    for sym in ["BTCUSDT", "SOLUSDT", "DOGEUSDT"]:
        for W, H in [(3, 3), (6, 3), (6, 6), (12, 6), (30, 12)]:
            tid = L.log_trial(exp, {"sym": sym, "W": W, "H": H}, stage="discovery", status="running")
            r = test(sym, days, W, H, L, exp)
            L.mark_trial(tid, "done" if r else "failed")
            if r:
                rows.append(r)
                print(f"  {sym:9} W{r['W_s']}s H{r['H_s']}s: IC={r['ic']:+.4f} grossSR={r['gross_sr']:+.1f} "
                      f"makerSR={r['maker_sr']:+.1f} takerSR={r['taker_sr']:+.1f} "
                      f"grossEdge={r['gross_edge_bps_per_bar']:+.3f}bps cost={r['cost_bps_per_bar']:.3f}bps "
                      f"DSR={r['dsr']:.3f}", flush=True)
    import json
    (ROOT / "reports" / "EVENT_OFI_RESULT.json").write_text(json.dumps(rows, indent=1, default=float))
    best_mk = max((r["maker_sr"] for r in rows), default=0)
    best_gross = max((r["gross_edge_bps_per_bar"] for r in rows), default=0)
    verdict = "PASS" if best_mk > 1.0 and any(r["dsr"] > 0.9 for r in rows) else "FAIL"
    L.add_verdict(hyp, "final", verdict, "validator", [exp],
                  rationale=f"event-level OFI, 3 coins, best maker Sharpe {best_mk:.1f}, "
                            f"best gross edge {best_gross:.3f} bps/bar",
                  strongest_surviving_objection=f"gross per-bar OFI edge <= {best_gross:.3f} bps at the "
                            f"10s-2min scale from aggregate trades; real book-event OFI needs L2 data")
    if verdict != "PASS":
        L.add_bound("F_MICRO_OFI_EVENT", "event-level OFI, 10s grid, 3 liquid coins, maker+taker",
                    f"F_MICRO_OFI_EVENT: {len(rows)} configs, 10s-2min windows. Best gross OFI edge "
                    f"<= {best_gross:.3f} bps/bar, best maker net Sharpe <= {best_mk:.1f}. Order-flow "
                    f"imbalance from AGGREGATE trades (not L2 book events) carries no tradeable "
                    f"directional edge at any accessible intraday timescale. Real OFI/microprice work "
                    f"requires a live L2 order-book feed.", [exp])
    print(f"EVENT OFI verdict: {verdict}")


if __name__ == "__main__":
    main()
