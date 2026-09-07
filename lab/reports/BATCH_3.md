# BATCH 3 — trend / momentum / reversal

_9 formulas. Status: {'BOUNDED': 9}._

These are registry-level dispositions: every formula here is either (a) a variant of a mechanism already tested and bounded in Rounds 1-3, (b) a sizing / gating tool used inside `lab/exec/` rather than a directional signal, or (c) blocked by a data source we do not have. Where a fresh run was warranted it is noted.

| id | family | paper | status | disposition |
|---|---|---|---|---|
| F040 | F_DIR | Moskowitz-Ooi-Pedersen 2012 | BOUNDED | TSMOM: F_XSEC xsec_momentum / F_DIR dir_ma_cross, 58+30 cfg, best DSR 0.14, COST/NO_EDGE |
| F041 | F_XSEC | Jegadeesh-Titman 1993 | BOUNDED | F_XSEC xsec_momentum 58 cfg: best |IC| 0.057, net Sharpe ≤1.3, DSR 0.14 |
| F042 | F_XSEC | Lehmann 1990 | BOUNDED | F_XSEC xsec_st_reversal: bounded (part of the 58) |
| F043 | F_XCOIN | Blitz 2011 | BOUNDED | F_XCOIN xcoin_pca_residual / xsec_residual_momentum 20 cfg, best DSR 0.09 |
| F044 | F_DEP | Lo-MacKinlay 1988 | BOUNDED | F_DEP dep_variance_ratio: best |IC| 0.014, DSR 0.085, NO_EDGE |
| F045 | F_TRANSFORM | Mandelbrot; Peng DFA | BOUNDED | F_TRANSFORM transform_hurst: bounded (part of the 10) |
| F046 | F_PAIRS | Uhlenbeck-Ornstein | BOUNDED | F_PAIRS pairs_johansen_ou: OOS s-score→fwd-spread corr +0.004 (wrong sign). NO_EDGE. |
| F047 | F_PAIRS | Avellaneda-Lee 2010 | BOUNDED | F_PAIRS pairs_avellaneda: gross Sharpe -0.83, surrogate p 0.52, NO_EDGE |
| F048 | F_DIR | Baltas-Kosowski | BOUNDED | vol-scaled TSMOM = F_DIR with vol.py sizing; bounded like F040 |

- formulas screened: 9
- passed cheap screen: 9 (all previously; none newly)
- passed full backtest: 0
- **survivors: 0**
- status distribution: {'BOUNDED': 9}

