"""Generate BATCH_{1,2,3,5,6}.md from the formula registry + prior ledgers.

Batch 4 is written by batch4.py (it actually ran). Batches 1/2/3/5/6 are
registry-level dispositions: their formulas are variants of mechanisms already
bounded in Rounds 1-3 or are sizing/gating tools, so they are recorded here with
their family, gate status and root cause rather than re-run.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from lab.research.formula_registry import ROWS

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "lab" / "reports"

BATCHES = {
    1: ("microstructure / order flow", ("F001", "F002", "F003", "F004", "F005", "F006",
                                        "F007", "F008", "F009", "F010", "F011", "F012", "F013", "F014")),
    2: ("volatility (sizing + gating, not direction)", tuple(f"F0{n}" for n in range(20, 31))),
    3: ("trend / momentum / reversal", tuple(f"F0{n}" for n in range(40, 49))),
    5: ("state / filtering (wrappers)", tuple(f"F0{n}" for n in range(80, 88))),
    6: ("cross-asset / stat", tuple(f"F0{n}" for n in range(90, 95))),
}


def write():
    by_id = {r[0]: r for r in ROWS}
    for n, (title, ids) in BATCHES.items():
        rows = [by_id[i] for i in ids if i in by_id]
        sc = Counter(r[6] for r in rows)
        rc = Counter(r[7].split(":")[0].split(";")[0].split("(")[0].strip()
                     for r in rows if r[6] in ("BOUNDED", "RUN_BATCH1"))
        n_survivor = sum(1 for r in rows if r[6] == "SURVIVOR")
        md = [f"# BATCH {n} — {title}", "",
              f"_{len(rows)} formulas. Status: {dict(sc)}._", "",
              "These are registry-level dispositions: every formula here is either "
              "(a) a variant of a mechanism already tested and bounded in Rounds 1-3, "
              "(b) a sizing / gating tool used inside `lab/exec/` rather than a directional "
              "signal, or (c) blocked by a data source we do not have. Where a fresh run "
              "was warranted it is noted.", "",
              "| id | family | paper | status | disposition |",
              "|---|---|---|---|---|"]
        for r in rows:
            md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[6]} | {r[7]} |")
        md += ["",
               f"- formulas screened: {len(rows)}",
               f"- passed cheap screen: {sc.get('RUN_BATCH1', 0) + sc.get('BOUNDED', 0)} "
               f"(all previously; none newly)",
               f"- passed full backtest: 0",
               f"- **survivors: {n_survivor}**",
               f"- status distribution: {dict(sc)}", ""]
        if n == 1:
            md += ["## Note on the 'new' microstructure formulas", "",
                   "- **F002 multi-level OFI** — needs a live L2 order-book feed. Binance bookDepth "
                   "is ±1%…±5% cumulative bands, not per-level. DATA_LIMITED.",
                   "- **F011 Hawkes self-excitation** — the intensity difference λ^buy − λ^sell is a "
                   "decay-weighted signed-flow series; that is exactly `ofi_momentum` in F_MICRO_OFI "
                   "(bounded). The branching ratio n = α/β is captured by `trade_sign_ac1` in the tape "
                   "bars. No incremental directional content over the bounded F_MICRO family.",
                   "- **F012 propagator / transient impact** — the propagator G(l)~l^-0.5 is a model of "
                   "*impact decay*, used in `cost_model` (square-root law, F013). As a *signal* it "
                   "reduces to lagged signed order flow — F_MICRO_OFI, bounded.",
                   "- **F001/F003/F004/F007/F010/F014** — F_MICRO_OFI / F_MICRO_BOOK / F_BOOK, all "
                   "bounded: gross microstructure edge exists (up to 4.3 gross Sharpe for F_BOOK "
                   "`imbvol_fade`) but is destroyed by turnover — COST_BOUND.", ""]
        if n == 2:
            md += ["## Note", "",
                   "Per spec D, volatility is for **sizing and gating, not direction**. "
                   "`lab/exec/vol.py` implements EWMA (F020), GARCH (F021), GJR-GARCH (F023), "
                   "Yang-Zhang (F027), and vol-targeting (F048 sizing). The directional variants "
                   "(low-vol tilt, semivariance skew, vol-of-vol) were tested as F_VOL in Round 1 "
                   "— best net Sharpe −0.08, DSR 0.002, NO_EDGE.", ""]
        if n == 5:
            md += ["## Note", "",
                   "Per spec, state/filtering formulas are **wrappers** for batches 1-4. "
                   "`lab/exec/kalman.py` implements the local-level smoother (F080), dynamic "
                   "hedge ratio (F081) and normalised innovation z (F082). F083 HMM regime "
                   "gating is a filter — there is currently no directional signal left that "
                   "clears the ladder for it to gate, so it is queued, not run. "
                   "F087 Johansen was run as F_PAIRS: cointegration does not persist OOS.", ""]
        (REP / f"BATCH_{n}.md").write_text("\n".join(md) + "\n")
        print(f"wrote BATCH_{n}.md  ({len(rows)} formulas, {n_survivor} survivors)")


if __name__ == "__main__":
    write()
