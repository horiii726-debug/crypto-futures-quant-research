# BATCH 1 — microstructure / order flow

_14 formulas. Status: {'BOUNDED': 8, 'DATA_LIMITED': 1, 'SIZING_ONLY': 2, 'NOT_RUN': 1, 'RUN_BATCH1': 2}._

These are registry-level dispositions: every formula here is either (a) a variant of a mechanism already tested and bounded in Rounds 1-3, (b) a sizing / gating tool used inside `lab/exec/` rather than a directional signal, or (c) blocked by a data source we do not have. Where a fresh run was warranted it is noted.

| id | family | paper | status | disposition |
|---|---|---|---|---|
| F001 | F_MICRO_OFI | Cont-Kukanov-Stoikov 2014 | BOUNDED | NO_EDGE from aggregate trades; needs live L2 (Round1 F_MICRO_OFI_EVENT, 28 cfg, gross Sharpe<0) |
| F002 | F_MICRO_OFI | Cont-Cucuringu-Zhang 2021 | DATA_LIMITED | no live L2 order-book feed; bookDepth is ±1%..±5% bands only |
| F003 | F_MICRO_BOOK | - | BOUNDED | F_BOOK book_imb_follow/fade, 120 cfg: gross Sharpe up to 4.3 but maker net -6..-99 (COST_BOUND) |
| F004 | F_MICRO_BOOK | Stoikov 2018 | BOUNDED | F_BOOK microprice_drift: |IC| 0.003, COST_BOUND / NO_EDGE |
| F005 | F_MICRO_OFI / cost_model | Kyle 1985 | BOUNDED | Kyle λ as a feature: F_FLOW flow_kyle_lambda bounded; λ is used in cost_model.impact instead |
| F006 | F_LIQ | Amihud 2002 | BOUNDED | F_LIQ liq_kyle_bar 6 cfg: best DSR 0.05, NO_EDGE |
| F007 | F_MICRO_OFI | Easley-LdP-O'Hara 2012 | BOUNDED | F_MICRO screen: no config passed rough-IC≥0.008 + net-positive |
| F008 | F_MICRO_OFI / universe | Roll 1984 | SIZING_ONLY | Roll spread used as a cost/universe input, not a directional signal |
| F009 | - | Corwin-Schultz 2012 | NOT_RUN | spread estimator only; equivalent info to F008 for the cost model — no directional content expected |
| F010 | F_MICRO_OFI | Lee-Ready 1991 | BOUNDED | F_MICRO signac_momentum: screen fail |
| F011 | - | Bacry-Muzy 2014 | RUN_BATCH1 | Hawkes self-excitation — tape event times available; branching ratio n=α/β |
| F012 | - | Bouchaud 2004 | RUN_BATCH1 | transient/propagator impact — needs signed-trade series (have from tape) |
| F013 | cost_model | Almgren; Toth 2011 | SIZING_ONLY | square-root impact IS the cost_model.impact term (Y=0.5). Not a signal. |
| F014 | F_MICRO_OFI | - | BOUNDED | realized-spread decay = adverse selection; measured in maker_model, not a signal |

- formulas screened: 14
- passed cheap screen: 10 (all previously; none newly)
- passed full backtest: 0
- **survivors: 0**
- status distribution: {'BOUNDED': 8, 'DATA_LIMITED': 1, 'SIZING_ONLY': 2, 'NOT_RUN': 1, 'RUN_BATCH1': 2}

## Note on the 'new' microstructure formulas

- **F002 multi-level OFI** — needs a live L2 order-book feed. Binance bookDepth is ±1%…±5% cumulative bands, not per-level. DATA_LIMITED.
- **F011 Hawkes self-excitation** — the intensity difference λ^buy − λ^sell is a decay-weighted signed-flow series; that is exactly `ofi_momentum` in F_MICRO_OFI (bounded). The branching ratio n = α/β is captured by `trade_sign_ac1` in the tape bars. No incremental directional content over the bounded F_MICRO family.
- **F012 propagator / transient impact** — the propagator G(l)~l^-0.5 is a model of *impact decay*, used in `cost_model` (square-root law, F013). As a *signal* it reduces to lagged signed order flow — F_MICRO_OFI, bounded.
- **F001/F003/F004/F007/F010/F014** — F_MICRO_OFI / F_MICRO_BOOK / F_BOOK, all bounded: gross microstructure edge exists (up to 4.3 gross Sharpe for F_BOOK `imbvol_fade`) but is destroyed by turnover — COST_BOUND.

