"""Batch 4 — crypto-native formulas (M4 / F060-F070) through the N3-N9 pipeline.

Family F_CN4 (crypto-native, batch 4), pre-declared. 1 formula = 1 trial at the
author's default params, no grid. Every TF screen result is logged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("CRYPTO_LAB_LEDGER_SQLITE", str(ROOT / "ledger" / "round3.sqlite"))

from lab.features import crypto_native as CN                    # noqa: E402
from lab.ledger import Ledger                                   # noqa: E402
from lab.research import pipeline as P                          # noqa: E402

FAMILY = "F_CN4"
BUDGET = 12
BAR_H = {"H1": 1, "H4": 4, "D1": 24}
HOLD_HINT = {"H1": 24, "H4": 12, "D1": 5}


def _price_1h() -> pd.DataFrame:
    parts = []
    for part in ("train", "valid"):
        p = ROOT / "data" / part / "prio50_2y" / "1h" / "close.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    return pd.concat(parts).sort_index()


def _qv_1h() -> pd.DataFrame:
    parts = []
    for part in ("train", "valid"):
        p = ROOT / "data" / part / "prio50_2y" / "1h" / "quote_volume.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    return pd.concat(parts).sort_index()


def _oi_1h(index, cols) -> pd.DataFrame:
    d = {}
    for c in cols:
        p = ROOT / "data" / "processed" / "bybit_oi" / f"{c}.parquet"
        if p.exists():
            s = pd.read_parquet(p).set_index("dt")["oi"]
            s.index = pd.to_datetime(s.index, utc=True)
            d[c] = s
    return pd.DataFrame(d).reindex(index).ffill(limit=6)


def _funding_1h(index, cols) -> pd.DataFrame:
    d = {}
    for c in cols:
        p = ROOT / "data" / "processed" / "basis" / f"{c}.parquet"
        if p.exists():
            s = pd.read_parquet(p)["funding"]
            s.index = pd.to_datetime(s.index, utc=True)
            d[c] = s
    return pd.DataFrame(d).reindex(index).ffill(limit=8)


def _resample(df, tf):
    h = BAR_H[tf]
    if h == 1:
        return df
    return df.resample(f"{h}h", label="right", closed="right").last()


def build_panels():
    C1 = _price_1h()
    QV1 = _qv_1h()
    cols = list(C1.columns)
    OI1 = _oi_1h(C1.index, cols) * C1                     # contracts -> USD
    F1 = _funding_1h(C1.index, cols)
    OI1 = OI1.where(C1.notna())
    panels = {}
    for tf in BAR_H:
        C = _resample(C1, tf)
        panels[tf] = dict(close=C,
                          quote_volume=_resample(QV1, tf).reindex_like(C),
                          oi_usd=_resample(OI1, tf).reindex_like(C),
                          funding=_resample(F1, tf).reindex_like(C))
    return panels


RUNNABLE = ["F060_leverage_ratio", "F061_oi_divergence", "F063_funding_momentum",
            "F068_liq_magnet", "F062_funding_level", "F070_basis_carry"]
DATA_LIMITED = {
    "F064_term_structure": "needs perp+quarterly futures; only BTC/ETH have quarterly -> 2-asset TS, not XS. Queue to a term-structure study.",
    "F065_xvenue_funding": "needs OKX funding (not yet ingested). Queue to module J.",
    "F066_liq_cascade": "coin-M liquidationSnapshot is BTC/ETH only -> F_LIQD, already bounded (UNDERPOWERED).",
    "F067_liq_imbalance": "same as F066 — F_LIQD, bounded.",
}
COLLAPSED = {
    "F060_leverage_ratio": "= oi_leverage_state (F_OI). Bounded + failed cross-venue (Round 3 J4). Re-run here for the registry record only.",
    "F062_funding_level": "= F_CARRY / F_FUND fund_carry. Bounded.",
    "F070_basis_carry": "= F_CARRY basis carry. Bounded (risk premium, surrogate p=1).",
}


def run() -> dict:
    L = Ledger()
    panels = build_panels()
    close_by_tf = {tf: panels[tf]["close"] for tf in BAR_H}
    spent = L.trial_count(family=FAMILY)
    results = []

    for fid in RUNNABLE:
        if spent >= BUDGET:
            break
        fn, params, native_tf, need = CN.REGISTRY[fid]
        hyp = L.add_hypothesis({
            "claim": f"{fid}: crypto-native cross-sectional signal, net-positive after per-coin cost",
            "mechanism": COLLAPSED.get(fid, f"see lab/features/crypto_native.py::{fid}"),
            "math_form": fid, "target": "xsec_fwd_return", "horizon": native_tf,
            "null_hypothesis": "rank IC = 0 ; net Sharpe <= 0 after cost",
            "family": FAMILY, "lessons_reviewed": True})
        exp = L.new_experiment(hyp, FAMILY, {"formula": fid, "params": params,
                                             "tfs": list(BAR_H)}, stage="discovery")
        # --- N3 cheap screen across TFs ---
        feat_by_tf = {}
        for tf in BAR_H:
            try:
                s = fn(panels[tf], **params)
            except Exception as e:
                s = None
            feat_by_tf[tf] = s
        scr = P.cheap_screen(feat_by_tf, close_by_tf)
        for tf, m in scr["by_tf"].items():
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    L.log_metric(exp, f"screen_{tf}_{k}", float(v), context={"formula": fid})

        tid = L.log_trial(exp, {"formula": fid, "config": "default"}, stage="discovery", status="running")
        spent += 1

        gate = None
        if scr["passes_screen"]:
            tf = scr["best_tf"]
            try:
                gate = P.full_gate(feat_by_tf[tf], close_by_tf[tf],
                                   family=FAMILY, ledger=L, tf=tf,
                                   hold_bars=HOLD_HINT[tf], q=0.25,
                                   seed=abs(hash(fid)) % (2**31))
                for k, v in gate.items():
                    if isinstance(v, (int, float)) and v == v:
                        L.log_metric(exp, k, float(v), trial_id=tid)
            except Exception as e:
                L.log_metric(exp, "gate_error", None, trial_id=tid, text_value=repr(e)[:200])
        L.mark_trial(tid, "done")

        v = P.verdict_of(gate)
        rc, why = P.diagnose(scr, gate)
        results.append({"formula": fid, "screen": scr, "gate": gate,
                        "verdict": v, "root_cause": rc, "why": why,
                        "exp_id": exp, "hyp_id": hyp})
        L.add_verdict(hyp, "final", ("PASS" if v == "PASS" else "FAIL"), "validator", [exp],
                      rationale=(f"{fid}: screen best ratio {scr['best_ratio']:.2f} @ {scr['best_tf']}; "
                                 + (f"gate net SR {gate['net_sr_ann']:.2f}, surrogate p {gate['surrogate_p']:.3f}, "
                                    f"DSR {gate['dsr_family']:.2f}" if gate else "did not pass screen")),
                      strongest_surviving_objection=f"{rc}: {why}")
        if v != "PASS":
            L.add_lesson(root_cause=f"{rc}: {why}",
                         lesson=f"F_CN4 {fid}: {why}. screen@{scr['best_tf']} ratio {scr['best_ratio']:.2f}"
                                + (f"; gate net SR {gate['net_sr_ann']:.2f}" if gate else ""),
                         family=FAMILY, subject_id=hyp, ladder_level="L4")
        b = scr["by_tf"].get(scr["best_tf"], {})
        print(f"  {fid:24} screen@{scr['best_tf'] or '-':3} ratio={scr['best_ratio']:5.2f} "
              f"IC={b.get('ic_raw', 0):+.4f}  -> {v:11}  [{rc}: {why[:60]}]", flush=True)

    survivors = [r for r in results if r["verdict"] == "PASS"]
    if not survivors:
        L.add_bound(FAMILY, "crypto-native cross-sectional (F060-F070) net of per-coin cost, H1/H4/D1",
                    f"{len(results)} formulas run (leverage ratio, OI divergence, funding momentum, "
                    f"liq magnet, funding level, basis carry). None clears the N3-N9 ladder. "
                    f"Root causes: {json.dumps({r['root_cause']: sum(1 for x in results if x['root_cause']==r['root_cause']) for r in results})}. "
                    f"Plus DATA_LIMITED: term-structure (no XS quarterly), cross-venue funding (no OKX yet), "
                    f"liquidation (BTC/ETH only). Crypto-native positioning/funding signals are bounded "
                    f"below the cost floor at feasible horizons for this universe/period.",
                    [r["exp_id"] for r in results])
    return {"results": results, "survivors": survivors, "spent": spent,
            "data_limited": DATA_LIMITED}


if __name__ == "__main__":
    r = run()
    rc_dist = {}
    for x in r["results"]:
        rc_dist[x["root_cause"]] = rc_dist.get(x["root_cause"], 0) + 1
    md = ["# BATCH 4 — crypto-native (M4 / F060-F070)", "",
          f"_Family F_CN4, ledger `ledger/round3.sqlite`. {len(r['results'])} formulas run, "
          f"{len(r['survivors'])} survivors, {r['spent']}/{12} budget._", "",
          "| formula | screen best TF | ratio | IC | verdict | root cause |",
          "|---|---|---|---|---|---|"]
    for x in r["results"]:
        b = x["screen"]["by_tf"].get(x["screen"]["best_tf"], {})
        md.append(f"| {x['formula']} | {x['screen']['best_tf']} | {x['screen']['best_ratio']:.2f} | "
                  f"{b.get('ic_raw', 0):+.4f} | {x['verdict']} | {x['root_cause']} |")
    md += ["", "## Per-TF screen detail", ""]
    for x in r["results"]:
        md.append(f"**{x['formula']}** — {x['why']}")
        md.append("| TF | IC | edge_bps | turnover | cost/bar bps | ratio |")
        md.append("|---|---|---|---|---|---|")
        for tf, m in x["screen"]["by_tf"].items():
            md.append(f"| {tf} | {m['ic_raw']:+.4f} | {m['edge_bps_est']:.2f} | {m['turnover_est']:.3f} "
                      f"| {m['cost_per_bar_bps']:.3f} | {m['ratio']:.2f} |")
        if x["gate"]:
            g = x["gate"]
            md.append(f"\ngate @ {g['tf']}: net SR {g['net_sr_ann']:.2f}, gross {g['gross_sr_ann']:.2f}, "
                      f"turnover {g['turnover']:.4f}, CV {g['cv_folds_positive']}/{g['cv_n']}, "
                      f"placebo {g['placebo_p']:.3f}, surrogate {g['surrogate_p']:.3f}, "
                      f"PBO {g['pbo']:.2f}, DSR {g['dsr_family']:.2f}, WF {g['walk_forward']}")
        md.append("")
    md += ["## Not run (recorded, not discarded)", ""]
    for fid, why in r["data_limited"].items():
        md.append(f"- **{fid}** — DATA_LIMITED: {why}")
    md += ["", f"## Root-cause distribution: {rc_dist}", ""]
    (ROOT / "lab" / "reports" / "BATCH_4.md").write_text("\n".join(md) + "\n")
    json.dump(r, open(ROOT / "lab" / "reports" / "BATCH_4.json", "w"), indent=1, default=float)
    print(f"\nBATCH 4 done — {len(r['survivors'])} survivors. root causes: {rc_dist}")
