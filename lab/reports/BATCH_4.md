# BATCH 4 — crypto-native (M4 / F060-F070)

_Family F_CN4, ledger `ledger/round3.sqlite`. 6 formulas run, 0 survivors, 6/12 budget._

| formula | screen best TF | ratio | IC | verdict | root cause |
|---|---|---|---|---|---|
| F060_leverage_ratio | D1 | 147.73 | -0.0418 | FAIL | NO_EDGE |
| F061_oi_divergence | D1 | 4.21 | -0.0239 | FAIL | NO_EDGE |
| F063_funding_momentum | D1 | 0.76 | +0.0009 | SCREEN_FAIL | NO_EDGE |
| F068_liq_magnet | D1 | 0.11 | -0.0036 | SCREEN_FAIL | NO_EDGE |
| F062_funding_level | H1 | 0.14 | -0.0023 | SCREEN_FAIL | NO_EDGE |
| F070_basis_carry | None | 0.00 | +0.0000 | SCREEN_FAIL | NO_EDGE |

## Per-TF screen detail

**F060_leverage_ratio** — placebo p=0.236, surrogate p=0.229
| TF | IC | edge_bps | turnover | cost/bar bps | ratio |
|---|---|---|---|---|---|
| H1 | -0.0152 | 0.95 | 0.010 | 0.092 | 10.32 |
| H4 | -0.0225 | 2.86 | 0.010 | 0.092 | 31.07 |
| D1 | -0.0418 | 13.59 | 0.010 | 0.092 | 147.73 |

gate @ D1: net SR -0.17, gross -0.15, turnover 0.0179, CV 4/6, placebo 0.236, surrogate 0.229, PBO 0.29, DSR 0.41, WF [np.float64(0.79), np.float64(-1.86), np.float64(2.88), np.float64(-2.01)]

**F061_oi_divergence** — placebo p=0.116, surrogate p=0.003
| TF | IC | edge_bps | turnover | cost/bar bps | ratio |
|---|---|---|---|---|---|
| H1 | -0.0167 | 1.04 | 0.137 | 1.259 | 0.83 |
| H4 | -0.0189 | 2.40 | 0.158 | 1.451 | 1.65 |
| D1 | -0.0239 | 7.76 | 0.200 | 1.840 | 4.21 |

gate @ D1: net SR 0.26, gross 0.41, turnover 0.1495, CV 3/6, placebo 0.116, surrogate 0.003, PBO 0.29, DSR 0.43, WF [np.float64(-0.17), np.float64(0.77), np.float64(1.59), np.float64(-1.11)]

**F063_funding_momentum** — screen ratio 0.76 < 3 at every TF
| TF | IC | edge_bps | turnover | cost/bar bps | ratio |
|---|---|---|---|---|---|
| H1 | +0.0018 | 0.11 | 0.027 | 0.252 | 0.44 |
| H4 | +0.0015 | 0.20 | 0.035 | 0.319 | 0.61 |
| D1 | +0.0009 | 0.31 | 0.044 | 0.406 | 0.76 |

**F068_liq_magnet** — screen ratio 0.11 < 3 at every TF
| TF | IC | edge_bps | turnover | cost/bar bps | ratio |
|---|---|---|---|---|---|
| H1 | -0.0030 | 0.19 | 0.600 | 5.513 | 0.03 |
| H4 | +0.0053 | 0.67 | 0.764 | 7.030 | 0.09 |
| D1 | -0.0036 | 1.17 | 1.127 | 10.362 | 0.11 |

**F062_funding_level** — screen ratio 0.14 < 3 at every TF
| TF | IC | edge_bps | turnover | cost/bar bps | ratio |
|---|---|---|---|---|---|
| H1 | -0.0023 | 0.15 | 0.110 | 1.010 | 0.14 |
| H4 | -0.0014 | 0.17 | 0.429 | 3.944 | 0.04 |
| D1 | +0.0035 | 1.13 | 0.994 | 9.138 | 0.12 |

**F070_basis_carry** — screen ratio 0.00 < 3 at every TF
| TF | IC | edge_bps | turnover | cost/bar bps | ratio |
|---|---|---|---|---|---|

## Not run (recorded, not discarded)

- **F064_term_structure** — DATA_LIMITED: needs perp+quarterly futures; only BTC/ETH have quarterly -> 2-asset TS, not XS. Queue to a term-structure study.
- **F065_xvenue_funding** — DATA_LIMITED: needs OKX funding (not yet ingested). Queue to module J.
- **F066_liq_cascade** — DATA_LIMITED: coin-M liquidationSnapshot is BTC/ETH only -> F_LIQD, already bounded (UNDERPOWERED).
- **F067_liq_imbalance** — DATA_LIMITED: same as F066 — F_LIQD, bounded.

## Root-cause distribution: {'NO_EDGE': 6}

