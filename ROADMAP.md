# ROADMAP

## S0 — Foundation  ✅ (this stage)
Framework + guards + stats + engine + synthetic calibration. No research, no
data download. Verification must be green:
- `python3 tests/test_lab.py` — all pass
- `python3 lab/engine/kernels.py` — numba IDENTICAL
- `test_seal.py` blocks `data/test/` access (exit 2)
- calibration: `--edge 0` is NOT a candidate, `--edge 0.35` IS a candidate
- `config/venue.yaml` still null (blocks pipeline on purpose)

## S1 — Data  ✅ (Option A run, tag `prio50_2y`)
`config/venue.yaml` filled from official Binance USDT-M spec (fees, funding,
per-symbol instrument map). 55-symbol universe, 2y 1m klines from
data.binance.vision, **survivorship intact** (20 delisted/migrated names
retained). Quality gate 55/55. Chronological partitions, **test sealed**,
hashes in ledger `datasets`. Transaction costs **measured from aggTrades**
(half-spread p60 0.85 bps) → `slippage_bps` 1.35. F-1 → `reports/F1_CRYPTO.md`:
**only 3d feasible**; 5m/15m/1h/4h/1d closed (bounds in ledger).
Not done: aggTrades full tape, bookTicker, depth (Tahap 3/4), Tahap 1c (100c/3y).

## S2 — Formula corpus  ✅ (limited: 4 priority families)
`lab/rag/store.py` + `_seed_corpus*.py`. **50 papers, 66 formulas** for
F_XSEC / F_FUND / F_FLOW / F_XCOIN (40 ADMISSIBLE, 9 PARKED, 1 REJECT_DATA).
`lab/rag/formulas.sqlite`. Full 23-division / 150–300-per corpus not attempted.

## S3–S7 — Alpha research  ✅ (4 priority families) — **ZERO survivors**
`lab/research/{panel,study,gates,campaign,main_s1_s7}.py`. 107 pre-registered
cross-sectional configs, purged CV + embargo, placebo (random ranking),
surrogate (block-shuffle), DSR on cumulative machine trial count. All FAIL.
Bounds written per family. **Test partition never unsealed** (`test_looks`=0).
See `reports/S1_S7_REPORT.md`.

## S8–S10 — not started (S8 needs 60d wall-clock; S10 needs human sign-off)
`lab/research/montecarlo.py` ($25k risk sim) built, unused — no candidate.

## Division families (23)
F_VOL F_DIR F_ENTRY F_EXIT F_STATE F_SIZE F_FILTER F_TRANSFORM F_DIST F_DEP
F_PATH F_MULTI F_FORMULA F_META F_EXEC F_FLOW F_BOOK F_LIQ F_FUND F_OI F_LIQD
F_XSEC F_XCOIN
