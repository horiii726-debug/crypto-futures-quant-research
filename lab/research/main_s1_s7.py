"""S1 -> S7 autonomous run for the 4 priority families. Stops before S8.

Idempotent-ish: panels/partitions cached; the ledger is append-only so a
re-run adds trials (raising the DSR bar further, by design).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lab.ledger import Ledger  # noqa
from lab.data import ingest, spread, f1 as f1mod  # noqa
from lab.data.spread import corwin_schultz, abdi_ranaldo  # noqa
from lab.research.panel import Panel  # noqa
from lab.research.campaign import run_family, write_bound_if_empty, GATES  # noqa
from lab.research import gates as gatemod  # noqa
from lab.research.study import run_study  # noqa

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
UNIVERSE = json.load(open(ROOT / "data/raw/_spec/universe_pool.json"))
UNIVERSE = [s for s in UNIVERSE if s != "SLERFUSDT"]

MONTHS = ingest.month_range("2023-09", "2025-08")
SPLITS = {
    "train": ("2023-09-01", "2025-02-28"),
    "valid": ("2025-03-01", "2025-05-31"),
    "test":  ("2025-06-01", "2025-08-31"),
}
TAG = "prio50_2y"


def step(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}", flush=True)


# ---------------- S1 ----------------

def s1_data(L: Ledger) -> dict:
    step("S1  DATA: quality gate, panels, partitions, spread, F-1")
    bp = ingest.build_panels(UNIVERSE, MONTHS, TAG, max_missing_pct=8.0)
    print(f"  quality gate: kept {bp['n_kept']}/{len(UNIVERSE)} symbols")
    for s, st, mp, ov in bp["dropped"]:
        print(f"    dropped {s}: status={st} missing%={mp} ohlc_viol={ov}")

    parts = ingest.partition_and_hash(TAG, SPLITS, L)
    for k, v in sorted(parts.items()):
        print(f"  {k:16} sha={v['sha256'][:16]} bars={v['bars']} symbols={v['symbols']} {v['range']}")

    # ---- spread: REALISED from aggTrades is authoritative -----------------
    # OHLC estimators (Corwin-Schultz / Abdi-Ranaldo) are unreliable on 1h
    # crypto bars (range is volatility-dominated, not spread-dominated), so
    # they are computed only as a cross-check, NOT used to set the number.
    step("S1  SPREAD MEASUREMENT (realised half-spread from aggTrades)")
    c1 = pd.read_parquet(PROC / "panels" / TAG / "1h" / "close.parquet")
    h1 = pd.read_parquet(PROC / "panels" / TAG / "1h" / "high.parquet")
    l1 = pd.read_parquet(PROC / "panels" / TAG / "1h" / "low.parquet")
    cs_med = float(np.nanmedian([corwin_schultz(h1[s].dropna().values,
                    l1[s].reindex(h1[s].dropna().index).values) for s in bp["kept"]]))
    ar_med = float(np.nanmedian([abdi_ranaldo(c1[s].dropna().values,
                    h1[s].reindex(c1[s].dropna().index).values,
                    l1[s].reindex(c1[s].dropna().index).values) for s in bp["kept"]]))

    realised = []
    for jf in ["spread_measured.json", "spread_extra.json"]:
        p = PROC / jf
        if p.exists():
            realised += [r for r in json.loads(p.read_text())
                         if r.get("status") == "ok"
                         and np.isfinite(r.get("half_spread_bps", np.nan))]
    # de-dup by symbol (keep first)
    seen, uniq = set(), []
    for r in realised:
        if r["symbol"] not in seen:
            seen.add(r["symbol"]); uniq.append(r)
    realised = uniq
    hs = np.array(sorted(r["half_spread_bps"] for r in realised))
    # drop obvious estimator blow-ups (> 5 bps half-spread on a liquid perp is
    # a bad sample day, not the real quoted spread)
    hs_clean = hs[hs <= 5.0]
    realised_half_bps = float(np.percentile(hs_clean, 60))   # conservative-ish
    print(f"  aggTrades symbols measured     = {len(realised)}  ({', '.join(sorted(seen))})")
    print(f"  half-spread bps: p25={np.percentile(hs_clean,25):.2f} "
          f"median={np.median(hs_clean):.2f} p60={realised_half_bps:.2f} "
          f"p75={np.percentile(hs_clean,75):.2f} max(clean)={hs_clean.max():.2f}  "
          f"(dropped {len(hs)-len(hs_clean)} >5bps outliers)")
    print(f"  cross-check (unreliable at 1h): Corwin-Schultz {cs_med*1e4:.1f}bps, "
          f"Abdi-Ranaldo {ar_med*1e4:.1f}bps")

    chosen_half = realised_half_bps * 1e-4

    # ---- fill venue.yaml slippage_bps (measured) ----
    venue = yaml.safe_load((ROOT / "config" / "venue.yaml").read_text())
    impact_bps = 0.5
    slippage_bps = round(realised_half_bps + impact_bps, 2)
    txt = (ROOT / "config" / "venue.yaml").read_text()
    import re as _re
    txt = _re.sub(r"slippage_bps: [\d.]+", f"slippage_bps: {slippage_bps}", txt)
    txt = txt.replace("slippage_bps: null", f"slippage_bps: {slippage_bps}")
    txt = txt.replace("status: PARTIAL", "status: CONFIGURED_FOR_RESEARCH")
    (ROOT / "config" / "venue.yaml").write_text(txt)
    print(f"  -> venue.yaml slippage_bps = {slippage_bps} "
          f"(realised half-spread p60 {realised_half_bps:.2f} + impact {impact_bps})")
    (PROC / "spread_estimates.json").write_text(json.dumps(
        {"cs_median_bps": cs_med * 1e4, "ar_median_bps": ar_med * 1e4,
         "realised_half_spread_bps_p60": realised_half_bps,
         "realised_by_symbol": {r["symbol"]: r["half_spread_bps"] for r in realised},
         "slippage_bps_written": slippage_bps,
         "note": "aggTrades realised half-spread is authoritative; OHLC estimators "
                 "shown only as an (unreliable at 1h) cross-check"}, indent=1, default=float))

    # ---- funding: mean |funding| for F-1 ----
    import lab.data.binance_vision as bv
    fund_abs = []
    for s in bp["kept"][:20]:
        fd = bv.funding(s, MONTHS)
        if len(fd):
            fund_abs.append(fd["fundingRate"].abs().mean())
    mean_abs_funding_8h = float(np.nanmean(fund_abs)) if fund_abs else 0.0002
    print(f"  mean |funding| per interval    = {mean_abs_funding_8h*1e4:.2f} bps (n={len(fund_abs)})")

    # ---- F-1 ----
    step("S1  F-1 FEASIBILITY (measured costs)")
    f1res = f1mod.f1_table(TAG, 1.0, chosen_half, mean_abs_funding_8h, venue)
    print(json.dumps(f1res["rows"], indent=1))
    print(f"  FEASIBLE horizons  : {f1res['feasible_horizons']}")
    print(f"  INFEASIBLE (closed): {f1res['infeasible_horizons']}")
    for h in f1res["infeasible_horizons"]:
        L.add_bound("ALL", f"horizon {h}", f"required IC "
                    f"{[r['required_IC'] for r in f1res['rows'] if r['horizon']==h][0]} "
                    f"exceeds the {f1mod.IC_CEILING} admissible ceiling at measured cost floor "
                    f"{[r['cost_floor_bps'] for r in f1res['rows'] if r['horizon']==h][0]}bps", [])
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "F1_CRYPTO.md").write_text(_f1_md(f1res, parts, bp))
    return {"panels": bp, "partitions": parts, "f1": f1res,
            "slippage_bps": slippage_bps, "kept": bp["kept"]}


def _f1_md(f1res, parts, bp) -> str:
    L = ["# F-1 — crypto feasibility (measured costs)\n",
         f"[SPEC] Universe tag `{f1res['tag']}`, {len(bp['kept'])} symbols "
         f"(survivorship: delisted names retained). 1h base panel.\n",
         f"Half-spread used: {f1res['half_spread_bps']} bps (Corwin-Schultz / Abdi-Ranaldo "
         f"on real OHLC, cross-checked vs aggTrades). Mean |funding|/8h: "
         f"{f1res['mean_abs_funding_8h_bps']} bps. Assumed round-trip turnover "
         f"{f1res['turnover_assumed']}x per rebalance.\n",
         "| horizon | cost floor (bps) | dispersion D (bps) | typ move R (bps) | required IC | required hit-rate | FEASIBLE |",
         "|---|---|---|---|---|---|---|"]
    for r in f1res["rows"]:
        L.append(f"| {r['horizon']} | {r['cost_floor_bps']} | {r['dispersion_D_bps']} | "
                 f"{r['typ_move_R_bps']} | {r['required_IC']} | {r['required_hit_rate']} | "
                 f"{'yes' if r['FEASIBLE'] else 'NO (closed)'} |")
    L.append(f"\n**Feasible horizons:** {', '.join(f1res['feasible_horizons']) or 'NONE'}")
    L.append(f"\n**Closed (infeasible) horizons:** {', '.join(f1res['infeasible_horizons']) or 'none'}")
    L.append(f"\nIC ceiling {f1res['ic_ceiling']} (generous upper bound on admissible crypto "
             f"cross-sectional IC); hit-rate ceiling {f1res['hit_rate_ceiling']}.")
    L.append("\n## Partition hashes (ledger `datasets`)\n")
    for k, v in sorted(parts.items()):
        L.append(f"- `{k}` sha256 `{v['sha256'][:24]}…` — {v['bars']} bars × {v['symbols']} symbols, {v['range'][0]} … {v['range'][1]}")
    return "\n".join(L)


# ---------------- panels for research ----------------

_FUND_CACHE = {}


def _funding_panel(index, symbols) -> pd.DataFrame:
    import lab.data.binance_vision as bv
    fund = pd.DataFrame(index=index, columns=symbols, dtype=float)
    for s in symbols:
        if s not in _FUND_CACHE:
            fd = bv.funding(s, MONTHS)
            _FUND_CACHE[s] = fd.set_index("dt")["fundingRate"] if len(fd) else None
        ser = _FUND_CACHE[s]
        if ser is not None:
            fund[s] = ser.reindex(index, method="ffill")
    return fund


BAR_HOURS = {"1h": 1.0, "4h": 4.0, "1d": 24.0, "2d": 48.0, "3d": 72.0}
_MULTIDAY = {"2d": 2, "3d": 3}


def _resample_multiday(frames: dict, n_days: int) -> dict:
    out = {}
    rule = f"{n_days}D"
    for k, v in frames.items():
        if v is None:
            out[k] = None
            continue
        if k in ("high",):
            out[k] = v.resample(rule, label="left", closed="left").max()
        elif k in ("low",):
            out[k] = v.resample(rule, label="left", closed="left").min()
        elif k in ("close",):
            out[k] = v.resample(rule, label="left", closed="left").last()
        elif k in ("quote_volume", "volume", "taker_buy_base"):
            out[k] = v.resample(rule, label="left", closed="left").sum()
        else:
            out[k] = v.resample(rule, label="left", closed="left").last()
    return out


def _load_research_panel(parts: list[str], bar: str = "1h") -> Panel:
    src_bar = "1d" if bar in _MULTIDAY else bar
    frames = {}
    for n in ["close", "quote_volume", "taker_buy_base", "volume", "high", "low"]:
        segs = []
        for part in parts:
            p = ROOT / "data" / part / TAG / src_bar / f"{n}.parquet"
            if p.exists():
                segs.append(pd.read_parquet(p))
        frames[n] = pd.concat(segs).sort_index() if segs else None
    if bar in _MULTIDAY:
        frames = _resample_multiday(frames, _MULTIDAY[bar])
    close = frames["close"]
    fund = _funding_panel(close.index, list(close.columns))
    if bar in _MULTIDAY:
        # funding over the multi-day bar = sum of interval funding in the window
        import lab.data.binance_vision as bv
        fsum = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        for s in close.columns:
            if s in _FUND_CACHE and _FUND_CACHE[s] is not None:
                fsum[s] = _FUND_CACHE[s].resample(f"{_MULTIDAY[bar]}D", label="left",
                                                  closed="left").sum().reindex(close.index)
        fund = fsum
    return Panel(close=close, quote_volume=frames["quote_volume"],
                 taker_buy_base=frames["taker_buy_base"], volume=frames["volume"],
                 high=frames["high"], low=frames["low"], funding=fund,
                 bar_hours=BAR_HOURS[bar])


# ---------------- S3-S6 ----------------

def s3_s6(L: Ledger, feasible_horizons: list[str]) -> dict:
    step("S3-S6  ALPHA RESEARCH (cross-sectional, 4 families)")
    research_bars = [h for h in feasible_horizons if h in ("1h", "4h", "1d", "2d", "3d")]
    # always include 1d as the reference even if only 3d is "feasible", so the
    # ladder produces a bound at the near horizon too.
    if "3d" in feasible_horizons and "3d" not in research_bars:
        research_bars.append("3d")
    if not research_bars:
        research_bars = ["1d"]
    if "1d" not in research_bars:
        research_bars = ["1d"] + research_bars
    print(f"  research bar frequencies: {research_bars}")

    out = {fam: {"family": fam, "n_configs": 0, "verdicts": [], "survivors": [],
                 "results": [], "frozen": False}
           for fam in ["F_XSEC", "F_FUND", "F_FLOW", "F_XCOIN"]}

    for bar in research_bars:
        pdisc = _load_research_panel(["train", "valid"], bar=bar)
        print(f"  [{bar}] discovery panel {pdisc.close.shape} "
              f"{pdisc.close.index.min()}..{pdisc.close.index.max()}")
        for fam in out:
            step(f"S3  family {fam}  @ {bar}")
            t0 = time.time()
            r = run_family(fam, pdisc, {bar: 1}, L, q=0.2)
            secs = round(time.time() - t0, 1)
            out[fam]["n_configs"] += r["n_configs"]
            out[fam]["results"] += r["results"]
            out[fam]["frozen"] = out[fam]["frozen"] or r["frozen"]
            for v in r["verdicts"]:
                v["bar"] = bar
                out[fam]["verdicts"].append(v)
                if v["status"] == "PASS":
                    out[fam]["survivors"].append(v)
            print(f"  {fam}@{bar}: {r['n_configs']} configs, "
                  f"{len([v for v in r['verdicts'] if v['status']=='PASS'])} survivors, {secs}s")
            for v in r["verdicts"]:
                b = v["best"]
                print(f"    {v['feature']:26} -> {v['status']:12} "
                      f"IC={b['ic_1bar']:+.4f} netSR_ann={b['sharpe_net_ann']:+.2f} "
                      f"DSR={b['dsr']:.3f} placebo_p={b['placebo_p_value']:.3f}")
    return out


# ---------------- S7 ----------------

def s7_oos(L: Ledger, s36: dict) -> dict:
    step("S7  OUT-OF-SAMPLE (single sealed test look per family with a survivor)")
    _ptest_cache = {}
    results = {}
    for fam, r in s36.items():
        if not r["survivors"]:
            print(f"  {fam}: no survivor -> no test look taken (partition stays sealed)")
            continue
        for surv in r["survivors"]:
            b = surv["best"]
            bar = surv.get("bar", "1d")
            if bar not in _ptest_cache:
                _ptest_cache[bar] = _load_research_panel(["test"], bar=bar)
            ptest = _ptest_cache[bar]
            rec = next(x for x in r["results"] if x["feature"] == surv["feature"])
            exp = L.new_experiment(
                rec["hyp_id"], fam,
                {"stage": "test", "feature": surv["feature"], "params": b["params"], "bar": bar},
                stage="validation")
            try:
                L.record_test_look(fam, exp, "auto", "AUTONOMOUS_RUN_S7",
                                   note=f"{surv['feature']} single OOS look")
            except PermissionError as e:
                print(f"  {fam}: {e}")
                continue
            st = run_study(f"TEST|{fam}|{surv['feature']}", surv["feature"], b["params"],
                           1, ptest, family=fam, hyp_id="TEST",
                           exp_id=exp, ledger=L, q=0.2,
                           mode="zscore" if surv["feature"] in ("xcoin_btc_leadlag", "xcoin_dispersion_switch") else "decile",
                           n_placebo=200, n_surrogate=200, seed=999)
            g = gatemod.evaluate(st, GATES)
            print(f"  {fam} {surv['feature']}: TEST net_sharpe_ann={st['sharpe_net_ann']:+.2f} "
                  f"IC={st['ic_1bar']:+.4f} DSR={st['dsr']:.3f} -> {g['status']}")
            results[f"{fam}:{surv['feature']}"] = {"study": {k: st[k] for k in st if k != "net_pnl_daily"},
                                                   "gate": g}
    return results


def main():
    L = Ledger()
    t0 = time.time()
    s1 = s1_data(L)
    s36 = s3_s6(L, s1["f1"]["feasible_horizons"])
    for fam, r in s36.items():
        if not r["survivors"]:
            r["bound_id"] = write_bound_if_empty(fam, r["results"], L)
    s7 = s7_oos(L, s36)
    summary = {"s1": {"kept": s1["kept"], "slippage_bps": s1["slippage_bps"],
                      "feasible_horizons": s1["f1"]["feasible_horizons"],
                      "infeasible_horizons": s1["f1"]["infeasible_horizons"]},
               "s3_s6": {f: {"n_configs": r["n_configs"], "survivors": r["survivors"],
                             "verdicts": r["verdicts"]} for f, r in s36.items()},
               "s7": s7,
               "total_trials_ledger": L.trial_count(),
               "seconds": round(time.time() - t0, 1)}
    (REPORTS / "S1_S7_RESULT.json").write_text(json.dumps(summary, indent=1, default=float))
    step(f"DONE in {summary['seconds']}s — cumulative lab trials: {summary['total_trials_ledger']}")
    print(json.dumps({"feasible": s1["f1"]["feasible_horizons"],
                      "survivors_by_family": {f: [v["feature"] for v in r["survivors"]]
                                              for f, r in s36.items()}}, indent=1))
    return summary


if __name__ == "__main__":
    main()
