# BATCH 5 — state / filtering (wrappers)

_8 formulas. Status: {'SIZING_ONLY': 2, 'BOUNDED': 2, 'RUN_BATCH5': 1, 'NOT_RUN': 3}._

These are registry-level dispositions: every formula here is either (a) a variant of a mechanism already tested and bounded in Rounds 1-3, (b) a sizing / gating tool used inside `lab/exec/` rather than a directional signal, or (c) blocked by a data source we do not have. Where a fresh run was warranted it is noted.

| id | family | paper | status | disposition |
|---|---|---|---|---|
| F080 | kalman.py | - | SIZING_ONLY | lab/exec/kalman.py local_level — adaptive-lag smoother |
| F081 | kalman.py / F_PAIRS | Elliott-vd-Hoek-Malcolm 2005 | BOUNDED | kalman.py hedge_ratio; F_PAIRS pairs_kalman_ou catastrophic (turnover 2.6, Sharpe -36) |
| F082 | kalman.py | - | SIZING_ONLY | lab/exec/kalman.py innovation_z — gating, not direction |
| F083 | F_STATE | Hamilton 1989 | RUN_BATCH5 | regime-gating wrapper. Nothing left that works to gate; run as a filter on F060. |
| F084 | - | Haas-Mittnik-Paolella | NOT_RUN | regime-dependent vol; sizing refinement, not a directional signal |
| F085 | - | Gordon-Salmond-Smith 1993 | NOT_RUN | non-linear state estimation; sizing input. Kalman covers the linear case. |
| F086 | - | Page 1954 | NOT_RUN | structural-break detector; a gate input, not a signal. Regime attribution (spec G) covers the need. |
| F087 | F_PAIRS | Johansen 1991 | BOUNDED | F_PAIRS: Johansen selects pairs correctly but cointegration does not persist OOS |

- formulas screened: 8
- passed cheap screen: 2 (all previously; none newly)
- passed full backtest: 0
- **survivors: 0**
- status distribution: {'SIZING_ONLY': 2, 'BOUNDED': 2, 'RUN_BATCH5': 1, 'NOT_RUN': 3}

## Note

Per spec, state/filtering formulas are **wrappers** for batches 1-4. `lab/exec/kalman.py` implements the local-level smoother (F080), dynamic hedge ratio (F081) and normalised innovation z (F082). F083 HMM regime gating is a filter — there is currently no directional signal left that clears the ladder for it to gate, so it is queued, not run. F087 Johansen was run as F_PAIRS: cointegration does not persist OOS.

