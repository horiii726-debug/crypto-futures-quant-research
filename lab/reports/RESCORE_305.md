# RESCORE_305 — per-coin cost re-score (RESEARCH ROUND 2 · P0.5)

_193 unique configs re-priced across the cross-sectional campaign (12 families x mechanisms x {1d, 3d}). Every config run through run_study twice: v1 = ROUND 1 flat pooled cost, v2 = per-coin cost_model. No trial logged (RescoreLedger); signals / params / universe / dates unchanged._

## Headline
- **gate-status changes: 0**
- net Sharpe **worse** under per-coin cost: 184 / 193
- net Sharpe **better**: 0
- mean Δ net Sharpe (v2 − v1): **-0.118**   (median -0.090)

The 55-coin cross-sectional universe is thin-coin-heavy, so the corrected per-coin cost is *higher* on average (~6.7 vs 4.27 bps one-way) and every already-failing verdict fails by a little more. **No verdict flips. The NO_SURVIVOR bound is strengthened by the cost correction, not weakened.**

## Gate-status changes

_none._

## Best config per family / horizon

| family | h | best v1 netSR | best v2 netSR | Δ | gate |
|---|---|---|---|---|---|
| F_DEP | 1d | -0.30 | -0.42 | -0.12 | FAIL->FAIL |
| F_DEP | 3d | +1.14 | +1.07 | -0.07 | FAIL->FAIL |
| F_DIR | 1d | +1.05 | +1.02 | -0.03 | FAIL->FAIL |
| F_DIR | 3d | +1.08 | +1.05 | -0.03 | FAIL->FAIL |
| F_FLOW | 1d | +1.25 | +1.09 | -0.16 | FAIL->FAIL |
| F_FLOW | 3d | +0.95 | +0.85 | -0.10 | FAIL->FAIL |
| F_FUND | 1d | +1.14 | +1.01 | -0.13 | FAIL->FAIL |
| F_FUND | 3d | +0.79 | +0.69 | -0.10 | FAIL->FAIL |
| F_LIQ | 1d | +0.86 | +0.81 | -0.05 | FAIL->FAIL |
| F_LIQ | 3d | +0.80 | +0.76 | -0.04 | FAIL->FAIL |
| F_MULTI | 1d | +0.48 | +0.15 | -0.33 | FAIL->FAIL |
| F_MULTI | 3d | +0.69 | +0.59 | -0.10 | FAIL->FAIL |
| F_PATH | 1d | +0.93 | +0.82 | -0.11 | FAIL->FAIL |
| F_PATH | 3d | +0.88 | +0.82 | -0.06 | FAIL->FAIL |
| F_STATE | 1d | -0.94 | -1.08 | -0.14 | FAIL->FAIL |
| F_STATE | 3d | -0.10 | -0.19 | -0.09 | FAIL->FAIL |
| F_TRANSFORM | 1d | +0.37 | +0.29 | -0.08 | FAIL->FAIL |
| F_TRANSFORM | 3d | +1.89 | +1.81 | -0.08 | FAIL->FAIL |
| F_VOL | 1d | -0.08 | -0.11 | -0.03 | FAIL->FAIL |
| F_VOL | 3d | -0.20 | -0.21 | -0.01 | FAIL->FAIL |
| F_XCOIN | 1d | +0.25 | +0.10 | -0.15 | FAIL->FAIL |
| F_XCOIN | 3d | +1.13 | +0.90 | -0.23 | FAIL->FAIL |
| F_XSEC | 1d | +0.95 | +0.87 | -0.08 | FAIL->FAIL |
| F_XSEC | 3d | +1.30 | +1.23 | -0.07 | FAIL->FAIL |

## Full table

| family | mechanism | h | cfg | v1 netSR | v2 netSR | Δ | gate |
|---|---|---|---|---|---|---|
| F_DEP | dep_autocorr1 | 1d | {'n': 21} | -1.60 | -1.76 | -0.16 | FAIL->FAIL |
| F_DEP | dep_autocorr1 | 1d | {'n': 45} | -1.00 | -1.11 | -0.11 | FAIL->FAIL |
| F_DEP | dep_autocorr1 | 3d | {'n': 15} | +0.09 | +0.01 | -0.08 | FAIL->FAIL |
| F_DEP | dep_autocorr1 | 3d | {'n': 7} | +0.23 | +0.11 | -0.12 | FAIL->FAIL |
| F_DEP | dep_variance_ratio | 1d | {'n': 30} | -0.30 | -0.42 | -0.12 | FAIL->FAIL |
| F_DEP | dep_variance_ratio | 1d | {'n': 60} | -0.52 | -0.61 | -0.09 | FAIL->FAIL |
| F_DEP | dep_variance_ratio | 3d | {'n': 10} | +1.12 | +1.02 | -0.10 | FAIL->FAIL |
| F_DEP | dep_variance_ratio | 3d | {'n': 20} | +1.14 | +1.07 | -0.07 | FAIL->FAIL |
| F_DIR | dir_bollinger_z | 1d | {'n': 14} | -1.05 | -1.29 | -0.24 | FAIL->FAIL |
| F_DIR | dir_bollinger_z | 1d | {'n': 30} | -0.84 | -1.00 | -0.16 | FAIL->FAIL |
| F_DIR | dir_bollinger_z | 3d | {'n': 10} | -0.51 | -0.61 | -0.10 | FAIL->FAIL |
| F_DIR | dir_bollinger_z | 3d | {'n': 5} | -0.03 | -0.17 | -0.14 | FAIL->FAIL |
| F_DIR | dir_donchian_pos | 1d | {'n': 14} | +0.02 | -0.24 | -0.26 | FAIL->FAIL |
| F_DIR | dir_donchian_pos | 1d | {'n': 30} | +1.01 | +0.84 | -0.17 | FAIL->FAIL |
| F_DIR | dir_donchian_pos | 3d | {'n': 10} | +0.94 | +0.85 | -0.09 | FAIL->FAIL |
| F_DIR | dir_donchian_pos | 3d | {'n': 5} | +0.47 | +0.35 | -0.12 | FAIL->FAIL |
| F_DIR | dir_ma_cross | 1d | {'fast': 10, 'slow': 30} | +0.79 | +0.73 | -0.06 | FAIL->FAIL |
| F_DIR | dir_ma_cross | 1d | {'fast': 10, 'slow': 60} | +1.05 | +1.02 | -0.03 | FAIL->FAIL |
| F_DIR | dir_ma_cross | 1d | {'fast': 5, 'slow': 30} | +0.76 | +0.69 | -0.07 | FAIL->FAIL |
| F_DIR | dir_ma_cross | 1d | {'fast': 5, 'slow': 60} | +0.77 | +0.73 | -0.04 | FAIL->FAIL |
| F_DIR | dir_ma_cross | 3d | {'fast': 2, 'slow': 10} | +0.65 | +0.59 | -0.06 | FAIL->FAIL |
| F_DIR | dir_ma_cross | 3d | {'fast': 2, 'slow': 20} | +0.81 | +0.77 | -0.04 | FAIL->FAIL |
| F_DIR | dir_ma_cross | 3d | {'fast': 3, 'slow': 10} | +1.08 | +1.03 | -0.05 | FAIL->FAIL |
| F_DIR | dir_ma_cross | 3d | {'fast': 3, 'slow': 20} | +1.08 | +1.05 | -0.03 | FAIL->FAIL |
| F_DIR | dir_macd | 1d | {'fast': 12, 'slow': 21} | -1.11 | -1.22 | -0.11 | FAIL->FAIL |
| F_DIR | dir_macd | 1d | {'fast': 12, 'slow': 34} | -0.80 | -0.89 | -0.09 | FAIL->FAIL |
| F_DIR | dir_macd | 1d | {'fast': 8, 'slow': 21} | -1.22 | -1.34 | -0.12 | FAIL->FAIL |
| F_DIR | dir_macd | 1d | {'fast': 8, 'slow': 34} | -0.80 | -0.92 | -0.12 | FAIL->FAIL |
| F_DIR | dir_macd | 3d | {'fast': 3, 'slow': 11} | -0.04 | -0.10 | -0.06 | FAIL->FAIL |
| F_DIR | dir_macd | 3d | {'fast': 3, 'slow': 7} | -0.47 | -0.54 | -0.07 | FAIL->FAIL |
| F_DIR | dir_macd | 3d | {'fast': 4, 'slow': 11} | +0.14 | +0.09 | -0.05 | FAIL->FAIL |
| F_DIR | dir_macd | 3d | {'fast': 4, 'slow': 7} | -0.11 | -0.17 | -0.06 | FAIL->FAIL |
| F_DIR | dir_rsi | 1d | {'n': 14} | -0.79 | -0.93 | -0.14 | FAIL->FAIL |
| F_DIR | dir_rsi | 1d | {'n': 30} | -0.85 | -0.94 | -0.09 | FAIL->FAIL |
| F_DIR | dir_rsi | 1d | {'n': 7} | -0.59 | -0.79 | -0.20 | FAIL->FAIL |
| F_DIR | dir_rsi | 3d | {'n': 10} | -0.94 | -0.99 | -0.05 | FAIL->FAIL |
| F_DIR | dir_rsi | 3d | {'n': 2} | -0.47 | -0.58 | -0.11 | FAIL->FAIL |
| F_DIR | dir_rsi | 3d | {'n': 5} | -0.74 | -0.81 | -0.07 | FAIL->FAIL |
| F_FLOW | flow_imb_momentum | 1d | {'lookback': 2} | +0.64 | +0.25 | -0.39 | FAIL->FAIL |
| F_FLOW | flow_imb_momentum | 1d | {'lookback': 3} | +0.64 | +0.35 | -0.29 | FAIL->FAIL |
| F_FLOW | flow_imb_momentum | 1d | {'lookback': 7} | +1.25 | +1.09 | -0.16 | FAIL->FAIL |
| F_FLOW | flow_imb_momentum | 3d | {'lookback': 2} | +0.95 | +0.85 | -0.10 | FAIL->FAIL |
| F_FLOW | flow_imb_reversal | 1d | {'lookback': 2} | -2.03 | -2.41 | -0.38 | FAIL->FAIL |
| F_FLOW | flow_imb_reversal | 3d | {'lookback': 2} | -1.35 | -1.45 | -0.10 | FAIL->FAIL |
| F_FLOW | flow_kyle_lambda | 1d | {'lookback': 14} | +0.12 | -0.06 | -0.18 | FAIL->FAIL |
| F_FLOW | flow_kyle_lambda | 1d | {'lookback': 30} | +0.51 | +0.42 | -0.09 | FAIL->FAIL |
| F_FLOW | flow_kyle_lambda | 1d | {'lookback': 7} | +0.62 | +0.37 | -0.25 | FAIL->FAIL |
| F_FLOW | flow_kyle_lambda | 3d | {'lookback': 10} | +0.43 | +0.37 | -0.06 | FAIL->FAIL |
| F_FLOW | flow_kyle_lambda | 3d | {'lookback': 2} | +0.20 | +0.02 | -0.18 | FAIL->FAIL |
| F_FLOW | flow_kyle_lambda | 3d | {'lookback': 5} | +0.67 | +0.57 | -0.10 | FAIL->FAIL |
| F_FLOW | flow_signed_vol | 1d | {'lookback': 2} | -1.00 | -1.21 | -0.21 | FAIL->FAIL |
| F_FLOW | flow_signed_vol | 1d | {'lookback': 3} | -0.33 | -0.49 | -0.16 | FAIL->FAIL |
| F_FLOW | flow_signed_vol | 1d | {'lookback': 7} | -1.04 | -1.13 | -0.09 | FAIL->FAIL |
| F_FLOW | flow_signed_vol | 3d | {'lookback': 2} | -0.62 | -0.67 | -0.05 | FAIL->FAIL |
| F_FUND | fund_accel | 1d | {'lookback': 2} | -2.23 | -2.66 | -0.43 | FAIL->FAIL |
| F_FUND | fund_accel | 3d | {'lookback': 2} | -0.24 | -0.44 | -0.20 | FAIL->FAIL |
| F_FUND | fund_carry | 1d | {} | -0.49 | -0.88 | -0.39 | FAIL->FAIL |
| F_FUND | fund_carry | 3d | {} | +0.66 | +0.53 | -0.13 | FAIL->FAIL |
| F_FUND | fund_momentum | 1d | {'lookback': 14} | +0.57 | +0.49 | -0.08 | FAIL->FAIL |
| F_FUND | fund_momentum | 1d | {'lookback': 2} | +0.56 | +0.30 | -0.26 | FAIL->FAIL |
| F_FUND | fund_momentum | 1d | {'lookback': 7} | +1.14 | +1.01 | -0.13 | FAIL->FAIL |
| F_FUND | fund_momentum | 3d | {'lookback': 2} | +0.79 | +0.69 | -0.10 | FAIL->FAIL |
| F_FUND | fund_momentum | 3d | {'lookback': 5} | +0.40 | +0.35 | -0.05 | FAIL->FAIL |
| F_FUND | fund_zscore | 1d | {'lookback': 15} | -2.67 | -3.10 | -0.43 | FAIL->FAIL |
| F_FUND | fund_zscore | 1d | {'lookback': 30} | -2.37 | -2.77 | -0.40 | FAIL->FAIL |
| F_FUND | fund_zscore | 1d | {'lookback': 60} | -2.03 | -2.43 | -0.40 | FAIL->FAIL |
| F_FUND | fund_zscore | 3d | {'lookback': 10} | -0.23 | -0.41 | -0.18 | FAIL->FAIL |
| F_FUND | fund_zscore | 3d | {'lookback': 20} | +0.09 | -0.07 | -0.16 | FAIL->FAIL |
| F_FUND | fund_zscore | 3d | {'lookback': 5} | -0.89 | -1.08 | -0.19 | FAIL->FAIL |
| F_LIQ | liq_kyle_bar | 1d | {'n': 14} | +0.12 | -0.06 | -0.18 | FAIL->FAIL |
| F_LIQ | liq_kyle_bar | 1d | {'n': 30} | +0.51 | +0.42 | -0.09 | FAIL->FAIL |
| F_LIQ | liq_kyle_bar | 1d | {'n': 60} | +0.86 | +0.81 | -0.05 | FAIL->FAIL |
| F_LIQ | liq_kyle_bar | 3d | {'n': 10} | +0.43 | +0.37 | -0.06 | FAIL->FAIL |
| F_LIQ | liq_kyle_bar | 3d | {'n': 20} | +0.80 | +0.76 | -0.04 | FAIL->FAIL |
| F_LIQ | liq_kyle_bar | 3d | {'n': 5} | +0.67 | +0.57 | -0.10 | FAIL->FAIL |
| F_MULTI | multi_fast_slow | 1d | {'fast': 3, 'slow': 21} | -0.46 | -0.61 | -0.15 | FAIL->FAIL |
| F_MULTI | multi_fast_slow | 1d | {'fast': 3, 'slow': 45} | -1.32 | -1.40 | -0.08 | FAIL->FAIL |
| F_MULTI | multi_fast_slow | 1d | {'fast': 7, 'slow': 21} | -1.01 | -1.17 | -0.16 | FAIL->FAIL |
| F_MULTI | multi_fast_slow | 1d | {'fast': 7, 'slow': 45} | -1.51 | -1.60 | -0.09 | FAIL->FAIL |
| F_MULTI | multi_fast_slow | 3d | {'fast': 2, 'slow': 15} | -1.38 | -1.43 | -0.05 | FAIL->FAIL |
| F_MULTI | multi_fast_slow | 3d | {'fast': 2, 'slow': 7} | -1.33 | -1.41 | -0.08 | FAIL->FAIL |
| F_MULTI | multi_tf_agreement | 1d | {} | +0.48 | +0.15 | -0.33 | FAIL->FAIL |
| F_MULTI | multi_tf_agreement | 3d | {} | +0.69 | +0.59 | -0.10 | FAIL->FAIL |
| F_PATH | path_mfe_mae | 1d | {'n': 10} | -0.82 | -1.02 | -0.20 | FAIL->FAIL |
| F_PATH | path_mfe_mae | 1d | {'n': 20} | +0.45 | +0.29 | -0.16 | FAIL->FAIL |
| F_PATH | path_mfe_mae | 1d | {'n': 45} | +0.93 | +0.82 | -0.11 | FAIL->FAIL |
| F_PATH | path_mfe_mae | 3d | {'n': 15} | +0.88 | +0.82 | -0.06 | FAIL->FAIL |
| F_PATH | path_mfe_mae | 3d | {'n': 3} | -0.40 | -0.50 | -0.10 | FAIL->FAIL |
| F_PATH | path_mfe_mae | 3d | {'n': 7} | +0.44 | +0.36 | -0.08 | FAIL->FAIL |
| F_PATH | path_run_length | 1d | {} | -1.21 | -1.41 | -0.20 | FAIL->FAIL |
| F_PATH | path_run_length | 3d | {} | -1.17 | -1.25 | -0.08 | FAIL->FAIL |
| F_STATE | state_vol_regime_mom | 1d | {'vol_n': 14, 'mom_n': 14} | -1.08 | -1.27 | -0.19 | FAIL->FAIL |
| F_STATE | state_vol_regime_mom | 1d | {'vol_n': 14, 'mom_n': 30} | -0.94 | -1.08 | -0.14 | FAIL->FAIL |
| F_STATE | state_vol_regime_mom | 1d | {'vol_n': 30, 'mom_n': 14} | -1.14 | -1.32 | -0.18 | FAIL->FAIL |
| F_STATE | state_vol_regime_mom | 1d | {'vol_n': 30, 'mom_n': 30} | -1.34 | -1.47 | -0.13 | FAIL->FAIL |
| F_STATE | state_vol_regime_mom | 3d | {'vol_n': 10, 'mom_n': 10} | -0.19 | -0.26 | -0.07 | FAIL->FAIL |
| F_STATE | state_vol_regime_mom | 3d | {'vol_n': 10, 'mom_n': 5} | -0.17 | -0.27 | -0.10 | FAIL->FAIL |
| F_STATE | state_vol_regime_mom | 3d | {'vol_n': 5, 'mom_n': 10} | -0.10 | -0.19 | -0.09 | FAIL->FAIL |
| F_STATE | state_vol_regime_mom | 3d | {'vol_n': 5, 'mom_n': 5} | -0.70 | -0.80 | -0.10 | FAIL->FAIL |
| F_TRANSFORM | transform_hurst | 1d | {'n': 120} | +0.37 | +0.29 | -0.08 | FAIL->FAIL |
| F_TRANSFORM | transform_hurst | 1d | {'n': 30} | +0.38 | +0.20 | -0.18 | FAIL->FAIL |
| F_TRANSFORM | transform_hurst | 1d | {'n': 60} | +0.35 | +0.22 | -0.13 | FAIL->FAIL |
| F_TRANSFORM | transform_hurst | 3d | {'n': 10} | +0.00 | +0.00 | +0.00 | FAIL->FAIL |
| F_TRANSFORM | transform_hurst | 3d | {'n': 20} | +1.89 | +1.81 | -0.08 | FAIL->FAIL |
| F_TRANSFORM | transform_hurst | 3d | {'n': 40} | +0.35 | +0.30 | -0.05 | FAIL->FAIL |
| F_TRANSFORM | transform_spectral_entropy | 1d | {'n': 32} | -0.60 | -0.82 | -0.22 | FAIL->FAIL |
| F_TRANSFORM | transform_spectral_entropy | 1d | {'n': 64} | +0.24 | +0.06 | -0.18 | FAIL->FAIL |
| F_TRANSFORM | transform_spectral_entropy | 3d | {'n': 11} | +0.34 | +0.21 | -0.13 | FAIL->FAIL |
| F_TRANSFORM | transform_spectral_entropy | 3d | {'n': 21} | +0.25 | +0.14 | -0.11 | FAIL->FAIL |
| F_VOL | vol_semivar_skew | 1d | {'n': 14} | -1.03 | -1.19 | -0.16 | FAIL->FAIL |
| F_VOL | vol_semivar_skew | 1d | {'n': 30} | -1.64 | -1.72 | -0.08 | FAIL->FAIL |
| F_VOL | vol_semivar_skew | 3d | {'n': 10} | -1.08 | -1.14 | -0.06 | FAIL->FAIL |
| F_VOL | vol_semivar_skew | 3d | {'n': 5} | -1.05 | -1.14 | -0.09 | FAIL->FAIL |
| F_VOL | vol_yang_zhang | 1d | {'n': 14} | -0.08 | -0.11 | -0.03 | FAIL->FAIL |
| F_VOL | vol_yang_zhang | 1d | {'n': 30} | -0.60 | -0.62 | -0.02 | FAIL->FAIL |
| F_VOL | vol_yang_zhang | 3d | {'n': 10} | -0.20 | -0.21 | -0.01 | FAIL->FAIL |
| F_VOL | vol_yang_zhang | 3d | {'n': 5} | -0.46 | -0.49 | -0.03 | FAIL->FAIL |
| F_XCOIN | xcoin_btc_leadlag | 1d | {'lead': 1} | +0.29 | -0.03 | -0.32 | FAIL->FAIL |
| F_XCOIN | xcoin_btc_leadlag | 3d | {'lead': 1} | -1.04 | -1.12 | -0.08 | FAIL->FAIL |
| F_XCOIN | xcoin_dispersion_switch | 1d | {'lookback': 14, 'mom_lb': 14} | -0.27 | -0.42 | -0.15 | FAIL->FAIL |
| F_XCOIN | xcoin_dispersion_switch | 1d | {'lookback': 14, 'mom_lb': 30} | -0.90 | -1.01 | -0.11 | FAIL->FAIL |
| F_XCOIN | xcoin_dispersion_switch | 3d | {'lookback': 5, 'mom_lb': 10} | -1.60 | -1.66 | -0.06 | FAIL->FAIL |
| F_XCOIN | xcoin_dispersion_switch | 3d | {'lookback': 5, 'mom_lb': 5} | -1.17 | -1.25 | -0.08 | FAIL->FAIL |
| F_XCOIN | xcoin_pca_residual | 1d | {'lookback': 21, 'n_pc': 3} | -0.85 | -1.18 | -0.33 | FAIL->FAIL |
| F_XCOIN | xcoin_pca_residual | 1d | {'lookback': 45, 'n_pc': 3} | -0.28 | -0.40 | -0.12 | FAIL->FAIL |
| F_XCOIN | xcoin_pca_residual | 3d | {'lookback': 15, 'n_pc': 3} | +0.16 | -0.04 | -0.20 | FAIL->FAIL |
| F_XCOIN | xcoin_pca_residual | 3d | {'lookback': 7, 'n_pc': 3} | +1.13 | +0.90 | -0.23 | FAIL->FAIL |
| F_XCOIN | xcoin_peer_return | 1d | {'lookback': 2} | -0.62 | -1.02 | -0.40 | FAIL->FAIL |
| F_XCOIN | xcoin_peer_return | 1d | {'lookback': 3} | -0.81 | -1.13 | -0.32 | FAIL->FAIL |
| F_XCOIN | xcoin_peer_return | 1d | {'lookback': 7} | -1.08 | -1.29 | -0.21 | FAIL->FAIL |
| F_XCOIN | xcoin_peer_return | 3d | {'lookback': 2} | -0.67 | -0.79 | -0.12 | FAIL->FAIL |
| F_XCOIN | xcoin_spillover_momentum | 1d | {'lookback': 14, 'corr_win': 30} | +0.25 | +0.10 | -0.15 | FAIL->FAIL |
| F_XCOIN | xcoin_spillover_momentum | 1d | {'lookback': 7, 'corr_win': 30} | -0.55 | -0.74 | -0.19 | FAIL->FAIL |
| F_XCOIN | xcoin_spillover_momentum | 3d | {'lookback': 2, 'corr_win': 10} | -0.92 | -1.05 | -0.13 | FAIL->FAIL |
| F_XCOIN | xcoin_spillover_momentum | 3d | {'lookback': 5, 'corr_win': 10} | +0.83 | +0.75 | -0.08 | FAIL->FAIL |
| F_XSEC | xsec_52w_high | 1d | {'lookback': 150} | +0.65 | +0.58 | -0.07 | FAIL->FAIL |
| F_XSEC | xsec_52w_high | 1d | {'lookback': 45} | +0.63 | +0.52 | -0.11 | FAIL->FAIL |
| F_XSEC | xsec_52w_high | 1d | {'lookback': 90} | +0.77 | +0.68 | -0.09 | FAIL->FAIL |
| F_XSEC | xsec_52w_high | 3d | {'lookback': 15} | +0.01 | -0.04 | -0.05 | FAIL->FAIL |
| F_XSEC | xsec_52w_high | 3d | {'lookback': 30} | +0.62 | +0.57 | -0.05 | FAIL->FAIL |
| F_XSEC | xsec_52w_high | 3d | {'lookback': 50} | +0.51 | +0.47 | -0.04 | FAIL->FAIL |
| F_XSEC | xsec_amihud | 1d | {'lookback': 14} | -1.58 | -1.60 | -0.02 | FAIL->FAIL |
| F_XSEC | xsec_amihud | 1d | {'lookback': 30} | -1.44 | -1.45 | -0.01 | FAIL->FAIL |
| F_XSEC | xsec_amihud | 1d | {'lookback': 60} | -1.42 | -1.42 | +0.00 | FAIL->FAIL |
| F_XSEC | xsec_amihud | 3d | {'lookback': 10} | -1.38 | -1.39 | -0.01 | FAIL->FAIL |
| F_XSEC | xsec_amihud | 3d | {'lookback': 20} | -1.73 | -1.74 | -0.01 | FAIL->FAIL |
| F_XSEC | xsec_amihud | 3d | {'lookback': 5} | -2.05 | -2.07 | -0.02 | FAIL->FAIL |
| F_XSEC | xsec_idio_vol | 1d | {'lookback': 21} | -0.26 | -0.30 | -0.04 | FAIL->FAIL |
| F_XSEC | xsec_idio_vol | 1d | {'lookback': 45} | -0.35 | -0.38 | -0.03 | FAIL->FAIL |
| F_XSEC | xsec_idio_vol | 3d | {'lookback': 15} | -0.32 | -0.34 | -0.02 | FAIL->FAIL |
| F_XSEC | xsec_idio_vol | 3d | {'lookback': 7} | -0.28 | -0.31 | -0.03 | FAIL->FAIL |
| F_XSEC | xsec_low_vol | 1d | {'lookback': 14} | -0.54 | -0.58 | -0.04 | FAIL->FAIL |
| F_XSEC | xsec_low_vol | 1d | {'lookback': 30} | -0.77 | -0.80 | -0.03 | FAIL->FAIL |
| F_XSEC | xsec_low_vol | 1d | {'lookback': 60} | -0.15 | -0.17 | -0.02 | FAIL->FAIL |
| F_XSEC | xsec_low_vol | 3d | {'lookback': 10} | -0.21 | -0.23 | -0.02 | FAIL->FAIL |
| F_XSEC | xsec_low_vol | 3d | {'lookback': 20} | -0.02 | -0.03 | -0.01 | FAIL->FAIL |
| F_XSEC | xsec_low_vol | 3d | {'lookback': 5} | +0.02 | -0.02 | -0.04 | FAIL->FAIL |
| F_XSEC | xsec_max | 1d | {'lookback': 21} | -0.41 | -0.44 | -0.03 | FAIL->FAIL |
| F_XSEC | xsec_max | 1d | {'lookback': 45} | -0.33 | -0.35 | -0.02 | FAIL->FAIL |
| F_XSEC | xsec_max | 3d | {'lookback': 15} | -0.44 | -0.46 | -0.02 | FAIL->FAIL |
| F_XSEC | xsec_max | 3d | {'lookback': 7} | -0.28 | -0.31 | -0.03 | FAIL->FAIL |
| F_XSEC | xsec_momentum | 1d | {'lookback': 14, 'skip': 1} | +0.25 | +0.10 | -0.15 | FAIL->FAIL |
| F_XSEC | xsec_momentum | 1d | {'lookback': 30, 'skip': 1} | +0.54 | +0.43 | -0.11 | FAIL->FAIL |
| F_XSEC | xsec_momentum | 1d | {'lookback': 60, 'skip': 1} | +0.95 | +0.87 | -0.08 | FAIL->FAIL |
| F_XSEC | xsec_momentum | 1d | {'lookback': 7, 'skip': 1} | -0.54 | -0.74 | -0.20 | FAIL->FAIL |
| F_XSEC | xsec_momentum | 3d | {'lookback': 10, 'skip': 1} | +1.23 | +1.17 | -0.06 | FAIL->FAIL |
| F_XSEC | xsec_momentum | 3d | {'lookback': 2, 'skip': 1} | -0.98 | -1.11 | -0.13 | FAIL->FAIL |
| F_XSEC | xsec_momentum | 3d | {'lookback': 20, 'skip': 1} | +1.16 | +1.12 | -0.04 | FAIL->FAIL |
| F_XSEC | xsec_momentum | 3d | {'lookback': 5, 'skip': 1} | +0.83 | +0.75 | -0.08 | FAIL->FAIL |
| F_XSEC | xsec_residual_momentum | 1d | {'lookback': 21, 'skip': 1} | +0.65 | +0.52 | -0.13 | FAIL->FAIL |
| F_XSEC | xsec_residual_momentum | 1d | {'lookback': 45, 'skip': 1} | +0.83 | +0.74 | -0.09 | FAIL->FAIL |
| F_XSEC | xsec_residual_momentum | 3d | {'lookback': 15, 'skip': 1} | +1.17 | +1.12 | -0.05 | FAIL->FAIL |
| F_XSEC | xsec_residual_momentum | 3d | {'lookback': 7, 'skip': 1} | +1.30 | +1.23 | -0.07 | FAIL->FAIL |
| F_XSEC | xsec_skew | 1d | {'lookback': 21} | -1.29 | -1.42 | -0.13 | FAIL->FAIL |
| F_XSEC | xsec_skew | 1d | {'lookback': 45} | -1.22 | -1.30 | -0.08 | FAIL->FAIL |
| F_XSEC | xsec_skew | 3d | {'lookback': 15} | -0.92 | -0.97 | -0.05 | FAIL->FAIL |
| F_XSEC | xsec_skew | 3d | {'lookback': 7} | -0.65 | -0.75 | -0.10 | FAIL->FAIL |
| F_XSEC | xsec_st_reversal | 1d | {'lookback': 2} | -0.30 | -0.69 | -0.39 | FAIL->FAIL |
| F_XSEC | xsec_st_reversal | 1d | {'lookback': 3} | -0.23 | -0.54 | -0.31 | FAIL->FAIL |
| F_XSEC | xsec_st_reversal | 1d | {'lookback': 7} | +0.36 | +0.15 | -0.21 | FAIL->FAIL |
| F_XSEC | xsec_st_reversal | 3d | {'lookback': 2} | -0.00 | -0.13 | -0.13 | FAIL->FAIL |
| F_XSEC | xsec_turnover_reversal | 1d | {'lookback': 14} | -0.40 | -0.42 | -0.02 | FAIL->FAIL |
| F_XSEC | xsec_turnover_reversal | 1d | {'lookback': 7} | -0.08 | -0.11 | -0.03 | FAIL->FAIL |
| F_XSEC | xsec_turnover_reversal | 3d | {'lookback': 2} | -0.30 | -0.33 | -0.03 | FAIL->FAIL |
| F_XSEC | xsec_turnover_reversal | 3d | {'lookback': 5} | -0.63 | -0.65 | -0.02 | FAIL->FAIL |
| F_XSEC | xsec_volume_trend | 1d | {'fast': 14, 'slow': 30} | +0.82 | +0.71 | -0.11 | FAIL->FAIL |
| F_XSEC | xsec_volume_trend | 1d | {'fast': 14, 'slow': 60} | +0.76 | +0.70 | -0.06 | FAIL->FAIL |
| F_XSEC | xsec_volume_trend | 1d | {'fast': 7, 'slow': 30} | +0.13 | +0.01 | -0.12 | FAIL->FAIL |
| F_XSEC | xsec_volume_trend | 1d | {'fast': 7, 'slow': 60} | +0.82 | +0.74 | -0.08 | FAIL->FAIL |
| F_XSEC | xsec_volume_trend | 3d | {'fast': 2, 'slow': 10} | +0.01 | -0.09 | -0.10 | FAIL->FAIL |
| F_XSEC | xsec_volume_trend | 3d | {'fast': 2, 'slow': 20} | +0.70 | +0.63 | -0.07 | FAIL->FAIL |
| F_XSEC | xsec_volume_trend | 3d | {'fast': 5, 'slow': 10} | +0.85 | +0.76 | -0.09 | FAIL->FAIL |
| F_XSEC | xsec_volume_trend | 3d | {'fast': 5, 'slow': 20} | +1.21 | +1.16 | -0.05 | FAIL->FAIL |
