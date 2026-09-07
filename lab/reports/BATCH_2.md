# BATCH 2 — volatility (sizing + gating, not direction)

_11 formulas. Status: {'SIZING_ONLY': 5, 'NOT_RUN': 5, 'BOUNDED': 1}._

These are registry-level dispositions: every formula here is either (a) a variant of a mechanism already tested and bounded in Rounds 1-3, (b) a sizing / gating tool used inside `lab/exec/` rather than a directional signal, or (c) blocked by a data source we do not have. Where a fresh run was warranted it is noted.

| id | family | paper | status | disposition |
|---|---|---|---|---|
| F020 | vol.py | RiskMetrics | SIZING_ONLY | lab/exec/vol.py ewma_var, λ=0.94 — position sizing |
| F021 | F_VOL / vol.py | Bollerslev 1986 | SIZING_ONLY | vol.py gjr_garch_sigma (GARCH baseline). F_VOL vol_* directional variants bounded. |
| F022 | - | Nelson 1991 | NOT_RUN | asymmetric-vol variant of F023; sizing only. GJR (F023) already implemented and covers the asymmetry. |
| F023 | vol.py | Glosten-Jagannathan-Runkle 1993 | SIZING_ONLY | lab/exec/vol.py gjr_garch_sigma — MLE, asymmetric. Sizing only per spec D. |
| F024 | F_VOL | Corsi 2009 | NOT_RUN | RV forecaster; feeds sizing. Directional use would be a vol-timing trial — deferred. |
| F025 | F_VOL | Barndorff-Nielsen 2010 | BOUNDED | F_VOL vol_semivar_skew 8 cfg: best net Sharpe -0.08, DSR 0.002, NO_EDGE |
| F026 | - | BNS 2004 | NOT_RUN | jump detector; sizing/gating input, not directional |
| F027 | vol.py | Yang-Zhang 2000 | SIZING_ONLY | lab/exec/vol.py yang_zhang — most efficient OHLC estimator. F_VOL vol_yang_zhang directional bounded. |
| F028 | F_VOL | Parkinson/Garman-Klass/Rogers-Satchell | SIZING_ONLY | RS term is inside Yang-Zhang. Directional low-vol tilt = F_VOL, bounded. |
| F029 | - | - | NOT_RUN | second-moment feature; gating input. Standalone directional use not expected to clear cost. |
| F030 | - | Barndorff-Nielsen 2008 | NOT_RUN | noise-robust RV estimator; sizing input only |

- formulas screened: 11
- passed cheap screen: 1 (all previously; none newly)
- passed full backtest: 0
- **survivors: 0**
- status distribution: {'SIZING_ONLY': 5, 'NOT_RUN': 5, 'BOUNDED': 1}

## Note

Per spec D, volatility is for **sizing and gating, not direction**. `lab/exec/vol.py` implements EWMA (F020), GARCH (F021), GJR-GARCH (F023), Yang-Zhang (F027), and vol-targeting (F048 sizing). The directional variants (low-vol tilt, semivariance skew, vol-of-vol) were tested as F_VOL in Round 1 — best net Sharpe −0.08, DSR 0.002, NO_EDGE.

