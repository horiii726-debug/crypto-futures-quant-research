# ROADMAP

## S0 — Foundation  ✅ (this stage)
Framework + guards + stats + engine + synthetic calibration. No research, no
data download. Verification must be green:
- `python3 tests/test_lab.py` — all pass
- `python3 lab/engine/kernels.py` — numba IDENTICAL
- `test_seal.py` blocks `data/test/` access (exit 2)
- calibration: `--edge 0` is NOT a candidate, `--edge 0.35` IS a candidate
- `config/venue.yaml` still null (blocks pipeline on purpose)

## S1 — Data  (awaiting confirmation)
Fill `config/venue.yaml` from official Binance USDT-M spec. Staged download
from data.binance.vision (klines, aggTrades, bookTicker, funding, OI,
long/short, liquidations). Survivorship: every symbol ever listed, delisted
included. Data quality gate (data-warden veto). Lock `config/universe.yaml`.
Chronological partitions, test sealed, hashes to ledger. F-1 feasibility table
→ `reports/F1_CRYPTO.md`; close INFEASIBLE horizons.

## S2 — Formula corpus  (awaiting confirmation)
`lab/rag/triage.py` + `extract_formula.py`. Tiered scan (title → abstract →
full text only if ADMISSIBLE). Sources: arXiv, SSRN, OpenAlex, S2, RePEc,
CORE, NBER/BIS/ECB/Fed, 1980–now. Admissibility filter
(ADMISSIBLE / REJECT_DATA / REJECT_HORIZON / PARKED). Extract formulas, tag to
one of 23 divisions (F_VOL … F_XCOIN). Target 150–300 per division. Store to
`lab/rag/formulas.sqlite` + vector index. Priority: F_XSEC, F_FUND, F_FLOW,
F_XCOIN.

## S3+ — Research cycles  (not started)
`/run-cycle` per objective. Gate ladder G0→G10. Discovery on train/valid;
single sealed test look per family on sign-off. Paper trading → live (R8).

## Division families (23)
F_VOL F_DIR F_ENTRY F_EXIT F_STATE F_SIZE F_FILTER F_TRANSFORM F_DIST F_DEP
F_PATH F_MULTI F_FORMULA F_META F_EXEC F_FLOW F_BOOK F_LIQ F_FUND F_OI F_LIQD
F_XSEC F_XCOIN
