"""P3: F_BOOK / F_LIQ test on the bookDepth window (2023-09 .. 2024-05).
bookDepth is the only order-book product with history on data.binance.vision,
ending 2024-05. ~8 months of daily features -> expected UNDERPOWERED (R5)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab.ledger import Ledger
from lab.research.panel import zscore_x
from lab.stats.power import required_n_for_ic

BD = ROOT / "data" / "processed" / "bookdepth_daily"
PROC = ROOT / "data" / "processed"


def load_book_panel():
    syms = [p.stem for p in BD.glob("*.parquet")]
    imb, slope, near = {}, {}, {}
    for s in syms:
        g = pd.read_parquet(BD / f"{s}.parquet")
        g.index = pd.to_datetime(g.index, utc=True)
        imb[s], slope[s], near[s] = g["depth_imb"], g["book_slope"], g["near_notional"]
    imb = pd.DataFrame(imb).sort_index()
    slope = pd.DataFrame(slope).reindex(imb.index)
    near = pd.DataFrame(near).reindex(imb.index)
    closes = []
    for part in ["train", "valid"]:
        p = ROOT / "data" / part / "prio50_2y" / "1d" / "close.parquet"
        if p.exists():
            closes.append(pd.read_parquet(p))
    close = pd.concat(closes).sort_index()
    common = [c for c in imb.columns if c in close.columns]
    idx = imb.index.intersection(close.index)
    return {"close": close.loc[idx, common], "depth_imb": imb.loc[idx, common],
            "book_slope": slope.loc[idx, common],
            "n_days": len(idx), "n_syms": len(common),
            "range": [str(idx.min()), str(idx.max())]}


def main():
    L = Ledger()
    bp = load_book_panel()
    print(f"book panel: {bp['n_days']} days x {bp['n_syms']} syms  {bp['range']}")
    close = bp["close"]
    hyp = L.add_hypothesis({
        "claim": "F_BOOK: near-touch depth imbalance predicts next-day XS return",
        "mechanism": "resting-liquidity asymmetry reflects near-term pressure",
        "math_form": "depth_imb = (bid_notional_1pct - ask_notional_1pct)/sum",
        "target": "xsec_fwd_return_1d", "horizon": "1d", "null_hypothesis": "IC = 0",
        "family": "F_BOOK", "lessons_reviewed": True})
    exp = L.new_experiment(hyp, "F_BOOK", {"window": bp["range"], "n_days": bp["n_days"]},
                           stage="discovery")
    feats = {
        "depth_imb_mom": zscore_x(bp["depth_imb"].rolling(3, min_periods=1).mean()),
        "depth_imb_level": zscore_x(bp["depth_imb"]),
        "book_slope_level": zscore_x(bp["book_slope"]),
        "depth_imb_rev": -zscore_x(bp["depth_imb"])}
    fwd = close.shift(-1) / close - 1.0
    b = fwd.rank(axis=1)
    rng = np.random.default_rng(0)
    T = len(close); arr = fwd.values
    results = []
    for name, sig in feats.items():
        tid = L.log_trial(exp, {"feature": name}, stage="discovery", status="running")
        a = sig.rank(axis=1)
        ic = float(a.corrwith(b, axis=1).mean(skipna=True))
        plc = np.array([float(pd.DataFrame(rng.normal(size=sig.shape), index=sig.index,
                        columns=sig.columns).where(sig.notna()).rank(axis=1)
                        .corrwith(b, axis=1).mean(skipna=True)) for _ in range(300)])
        p_plc = float((np.sum(np.abs(plc) >= abs(ic)) + 1) / 301)
        surr = []
        for _ in range(300):
            st = rng.integers(0, T, int(np.ceil(T / 10)))
            order = np.concatenate([np.arange(s, s + 10) % T for s in st])[:T]
            sh = pd.DataFrame(arr[order], index=fwd.index, columns=fwd.columns).rank(axis=1)
            surr.append(abs(float(a.corrwith(sh, axis=1).mean(skipna=True))))
        surr = np.array(surr)
        p_surr = float((np.sum(surr >= abs(ic)) + 1) / 301)
        for k, v in [("ic_1bar", ic), ("placebo_p_value", p_plc), ("surrogate_p_value", p_surr)]:
            L.log_metric(exp, k, v, trial_id=tid)
        L.mark_trial(tid, "done")
        results.append({"feature": name, "ic": ic, "placebo_p": p_plc, "surrogate_p": p_surr,
                        "surrogate_q99": float(np.quantile(surr, 0.99))})
        print(f"  {name:18} IC={ic:+.4f} placebo_p={p_plc:.3f} surrogate_p={p_surr:.3f}")
    req = required_n_for_ic(0.03)
    best_ic = max(abs(r["ic"]) for r in results)
    verdict = "UNDERPOWERED" if bp["n_days"] < req else (
        "FAIL" if all(r["surrogate_p"] > 0.05 or abs(r["ic"]) < 0.03 for r in results) else "REVIEW")
    print(f"\n  required N for IC 0.03 ~ {req} days; have {bp['n_days']} -> {verdict}")
    L.add_verdict(hyp, "final", verdict, "validator", [exp],
                  rationale=f"F_BOOK on {bp['n_days']}d ({bp['range'][0][:10]}..{bp['range'][1][:10]}); best |IC|={best_ic:.4f}; bookDepth history ends 2024-05.",
                  strongest_surviving_objection=f"only ~8 months of order-book history exist; cannot resolve IC 0.03 daily (need ~{req} obs).")
    L.add_bound("F_BOOK", "cross-sectional daily, bookDepth window 2023-09..2024-05",
                f"F_BOOK/F_LIQ (depth): {bp['n_days']} trading days is all the order-book history that exists (bookDepth ends 2024-05). Best |IC| {best_ic:.4f}, verdict {verdict}. A real microprice / book-imbalance strategy needs a live L2 feed.", [exp])
    (PROC / "book_test_result.json").write_text(json.dumps(
        {"panel": {k: bp[k] for k in ("n_days", "n_syms", "range")}, "results": results,
         "required_n_ic03": req, "verdict": verdict}, indent=1, default=float))
    print("\nverdict:", verdict)


if __name__ == "__main__":
    main()
