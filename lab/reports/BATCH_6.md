# BATCH 6 — cross-asset / stat

_5 formulas. Status: {'BOUNDED': 5}._

These are registry-level dispositions: every formula here is either (a) a variant of a mechanism already tested and bounded in Rounds 1-3, (b) a sizing / gating tool used inside `lab/exec/` rather than a directional signal, or (c) blocked by a data source we do not have. Where a fresh run was warranted it is noted.

| id | family | paper | status | disposition |
|---|---|---|---|---|
| F090 | F_XCOIN | Granger 1969 | BOUNDED | F_XCOIN xcoin_btc_leadlag: bounded (part of the 20) |
| F091 | F_XCOIN | - | BOUNDED | F_XCOIN xcoin_pca_residual: best DSR 0.09 |
| F092 | F_STATE | - | BOUNDED | F_STATE / F_XCOIN xcoin_dispersion_switch: bounded |
| F093 | F_XCOIN | - | BOUNDED | F_XCOIN xcoin_dispersion_switch 5 cfg: bounded |
| F094 | F_VOL / F_XCOIN | Ang et al 2006 | BOUNDED | F_XSEC xsec_idio_vol: bounded (Ang et al low-vol anomaly, part of the 58) |

- formulas screened: 5
- passed cheap screen: 5 (all previously; none newly)
- passed full backtest: 0
- **survivors: 0**
- status distribution: {'BOUNDED': 5}

