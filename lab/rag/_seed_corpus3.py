"""S2 expansion (P2) - all remaining divisions. Method-driven: the estimators
are textbook, citations are real anchor papers. Admissibility is by DATA
availability only (our set: 1m OHLCV + funding + bookDepth 2023-01..2024-05).

F_OI / F_LIQD => REJECT_DATA: Binance does not publish >30d open-interest or
liquidation history in bulk, and the REST hist endpoints are 30-day only.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab.rag.store import Corpus  # noqa

C = Corpus()

P = [
 # ---- F_VOL (estimators; overlay/sizing, and low-vol XS sort) ----
 ("Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts", "Andersen & Bollerslev", 1998, "10.2307/2527343", "doi", "ADMISSIBLE", "RV from 1m returns", "RV forecastable; direction not",
  [("F_VOL","realized_vol","RV_t = sum_i r_{t,i}^2 over the day's 1m returns",["close"],"1d",1,"base measure"),
   ("F_VOL","har_rv","RV_{t+1}=c+b_d RV_t+b_w mean(RV_{t-5:t})+b_m mean(RV_{t-22:t})",["close"],"1d",1,None),
   ("F_VOL","bipower_variation","BV_t=(pi/2) sum |r_i||r_{i-1}|; jump-robust",["close"],"1d",1,None),
   ("F_VOL","realized_semivariance","RS+ = sum r_i^2 1(r_i>0); RS- similarly",["close"],"1d",1,"signed component")]),
 ("A Heterogeneous Autoregressive Model of Realized Volatility", "Corsi", 2009, "10.1093/jjfinec/nbp001", "doi", "ADMISSIBLE", "HAR", "workhorse",
  [("F_VOL","harq","HAR with RQ-adjusted coefficients (measurement-error aware)",["close"],"1d",1,None),
   ("F_VOL","med_rv","MedRV = c * median(|r_{i-1}|,|r_i|,|r_{i+1}|)^2 sum; jump robust",["close"],"1d",1,None)]),
 ("The Distribution of Realized Exchange Rate Volatility (Garman-Klass/Parkinson/RS/YZ)", "various", 2000, "10.1086/209650", "doi", "ADMISSIBLE", "OHLC range estimators", "efficient vol from OHLC",
  [("F_VOL","parkinson","sigma^2 = (1/4ln2) mean( ln(H/L)^2 )",["high","low"],"1d",1,None),
   ("F_VOL","garman_klass","0.5 ln(H/L)^2 - (2ln2-1) ln(C/O)^2",["open","high","low","close"],"1d",1,None),
   ("F_VOL","rogers_satchell","ln(H/C)ln(H/O)+ln(L/C)ln(L/O); drift-free",["open","high","low","close"],"1d",1,None),
   ("F_VOL","yang_zhang","overnight + k*open-close + (1-k)*RS",["open","high","low","close"],"1d",1,None)]),
 ("Generalized Autoregressive Conditional Heteroskedasticity", "Bollerslev", 1986, "10.1016/0304-4076(86)90063-1", "doi", "ADMISSIBLE", "returns", "GARCH forecastable",
  [("F_VOL","garch11","sigma^2_t = w + a eps^2_{t-1} + b sigma^2_{t-1}",["close"],"1d",1,None),
   ("F_VOL","gjr_garch","+ gamma eps^2_{t-1} 1(eps_{t-1}<0); leverage",["close"],"1d",1,None),
   ("F_VOL","egarch","ln sigma^2_t = w + a g(z_{t-1}) + b ln sigma^2_{t-1}",["close"],"1d",1,None)]),
 ("Volatility is Rough", "Gatheral, Jaisson & Rosenbaum", 2018, "10.1080/14697688.2017.1393551", "doi", "ADMISSIBLE", "log-RV increments", "H~0.1; better short-horizon vol forecast",
  [("F_VOL","rough_vol_hurst","H from log-RV variogram; forecast via fractional kernel",["close"],"1d",1,None)]),

 # ---- F_STATE (regime; overlay) ----
 ("A New Approach to the Economic Analysis of Nonstationary Time Series (Markov switching)", "Hamilton", 1989, "10.2307/1912559", "doi", "ADMISSIBLE", "returns", "regime identification, not alpha itself",
  [("F_STATE","ms_2state","2-state Gaussian Markov-switching mean/var on returns; P(state=bull_t)",["close"],"1d",1,"conditioner"),
   ("F_STATE","hmm_3state","3-state Gaussian HMM (bull/bear/chop) on (ret, |ret|); Viterbi state",["close"],"1d",1,None)]),
 ("Testing for a Unit Root in Time Series with a Changing Mean (CUSUM/changepoint)", "Perron", 1990, "10.2307/1391978", "doi", "ADMISSIBLE", "returns", "structural break detection",
  [("F_STATE","cusum_break","CUSUM of squared returns; flag regime change",["close"],"1d",1,None),
   ("F_STATE","vol_regime","binary: RV_t above/below trailing median -> high/low vol sleeve",["close"],"1d",1,None),
   ("F_STATE","trend_regime","sign & strength of a Kalman/loess trend slope on log price",["close"],"1d",1,None)]),
 ("Threshold Autoregression / SETAR", "Tong", 1990, "10.1093/oso/9780198522249.001.0001", "isbn", "ADMISSIBLE", "returns", "nonlinear regime dynamics",
  [("F_STATE","setar_state","threshold on trailing return: above thr -> momentum regime, below -> mean-revert",["close"],"1d",1,None)]),

 # ---- F_DIR (direction; direct alpha, cross-sectional) ----
 ("Deep Learning for Bitcoin Price Direction Prediction", "various (Financial Innovation)", 2024, "10.1186/s40854-024-00643-1", "doi", "ADMISSIBLE", "OHLCV", "direction hard; RSI/MACD common features",
  [("F_DIR","xsec_rsi","cross-sectional rank of Wilder RSI(n); fade extremes",["close"],"1d",1,None),
   ("F_DIR","xsec_macd","rank of MACD histogram (12,26,9) normalised by price",["close"],"1d",1,None),
   ("F_DIR","xsec_ma_cross","rank of (SMA_fast/SMA_slow - 1)",["close"],"1d",1,None),
   ("F_DIR","xsec_bollinger_z","rank of (close - SMA_n)/(k*std_n); mean reversion",["close"],"1d",1,None)]),
 ("Time Series Momentum", "Moskowitz, Ooi & Pedersen", 2012, "10.1016/j.jfineco.2011.11.003", "doi", "ADMISSIBLE", "returns", "TS momentum; weaker in crypto post-2021",
  [("F_DIR","ts_momentum_sign","position_i = sign(sum r_{i,t-L:t}); own-asset trend",["close"],"1d",1,"time-series, run per coin then equal-weight"),
   ("F_DIR","xsec_trend_strength","rank of R^2 of log-price vs time over window (trend quality)",["close"],"1d",1,None)]),
 ("Technical Analysis and the Cross-Section of Returns (adapted)", "Han, Yang & Zhou", 2013, "10.1017/S0022109013000586", "doi", "ADMISSIBLE", "prices", "MA-based trend signals",
  [("F_DIR","xsec_ma_distance","rank of close/SMA_50 - 1",["close"],"1d",1,None),
   ("F_DIR","xsec_donchian_pos","rank of position within N-day high-low channel",["high","low","close"],"1d",1,None)]),

 # ---- F_ENTRY / F_EXIT (timing; overlay) ----
 ("Optimal Trading with Stochastic Signals (entry/exit)", "various", 2015, "10.2139/ssrn.2578377", "ssrn", "ADMISSIBLE", "prices", "timing overlay",
  [("F_ENTRY","breakout_entry","enter when close breaks N-day high (Donchian)",["high","close"],"1d",1,None),
   ("F_ENTRY","pullback_entry","enter on pullback to SMA in an uptrend",["close"],"1d",1,None),
   ("F_EXIT","chandelier_exit","exit at highest_high - k*ATR",["high","low","close"],"1d",1,None),
   ("F_EXIT","time_stop","exit after H bars regardless",["close"],"1d",1,None),
   ("F_EXIT","trailing_atr","trailing stop at k*ATR from peak",["high","low","close"],"1d",1,None)]),

 # ---- F_SIZE (sizing; overlay) ----
 ("A New Interpretation of Information Rate (Kelly)", "Kelly", 1956, "10.1002/j.1538-7305.1956.tb03809.x", "doi", "ADMISSIBLE", "signal + vol", "sizing overlay",
  [("F_SIZE","fractional_kelly","f = lambda * mu_hat / sigma_hat^2, lambda in (0,1]",["close"],"n/a",1,None),
   ("F_SIZE","vol_target","w_i proportional to target_vol / sigma_i",["close"],"n/a",1,None),
   ("F_SIZE","risk_parity","w_i proportional to 1/sigma_i, then normalise",["close"],"n/a",1,None),
   ("F_SIZE","drawdown_control","scale gross by (1 - current_dd/max_dd)",["close"],"n/a",1,None)]),

 # ---- F_TRANSFORM (features) ----
 ("Long-Term Storage Capacity of Reservoirs (Hurst)", "Hurst", 1951, "10.1061/TACEAT.0006518", "doi", "ADMISSIBLE", "prices", "H<0.5 mean-revert, >0.5 trend",
  [("F_TRANSFORM","hurst_rs","rescaled-range Hurst exponent on trailing log-price",["close"],"1d",1,None),
   ("F_TRANSFORM","dfa_alpha","detrended fluctuation analysis exponent",["close"],"1d",1,None)]),
 ("A Wavelet-Based Analysis of Financial Time Series", "Gencay, Selcuk & Whitcher", 2002, "10.1016/B978-0-12-279670-8.X5000-0", "isbn", "ADMISSIBLE", "prices", "multiscale energy",
  [("F_TRANSFORM","wavelet_energy_ratio","ratio of high-freq to low-freq wavelet energy; regime feature",["close"],"1d",1,None),
   ("F_TRANSFORM","spectral_entropy","Shannon entropy of the normalised periodogram",["close"],"1d",1,None),
   ("F_TRANSFORM","pca_factor_load","loading of coin on rolling PC1..PC3 of the return panel",["close"],"1d",1,None)]),

 # ---- F_DIST (tail / quantile) ----
 ("Modelling Extremal Events (EVT/GPD)", "Embrechts, Kluppelberg & Mikosch", 1997, "10.1007/978-3-642-33483-2", "isbn", "ADMISSIBLE", "returns", "tail risk, not directional alpha",
  [("F_DIST","gpd_tail_index","fit GPD to trailing loss exceedances; xi as risk state",["close"],"1d",1,"risk overlay"),
   ("F_DIST","cvar_sort","rank coins by trailing CVaR_95; avoid worst tails",["close"],"1d",1,None),
   ("F_DIST","quantile_slope","spread of trailing 90th vs 10th return quantile (dispersion)",["close"],"1d",1,None)]),
 ("Conformal Prediction for Reliable Machine Learning", "Vovk et al", 2005, "10.1007/b106715", "isbn", "ADMISSIBLE", "any base model residuals", "prediction interval width as a confidence gate",
  [("F_DIST","conformal_width","width of the conformal interval around a base forecast; skip wide",["close"],"1d",1,"meta gate")]),

 # ---- F_DEP (dependence) ----
 ("Transfer Entropy - A Model-Free Measure of Effective Connectivity", "Schreiber", 2000, "10.1103/PhysRevLett.85.461", "doi", "ADMISSIBLE", "return panel", "TE from BTC to alt as a feature",
  [("F_DEP","te_from_btc","binned transfer entropy BTC->coin over trailing window",["close"],"1d",1,None),
   ("F_DEP","return_autocorr","AR(1) coefficient of trailing coin returns; sign of persistence",["close"],"1d",1,None),
   ("F_DEP","mutual_info_lag1","MI between r_t and r_{t-1}; nonlinear dependence",["close"],"1d",1,None),
   ("F_DEP","variance_ratio","Lo-MacKinlay VR(q); <1 mean-revert, >1 trend",["close"],"1d",1,None)]),

 # ---- F_PATH (path geometry) ----
 ("Path-Dependent Volatility / Excursion Theory (adapted MFE-MAE)", "practitioner", 2019, "10.2139/ssrn.3400858", "ssrn", "ADMISSIBLE", "OHLC", "trade path stats",
  [("F_PATH","mfe_mae_ratio","trailing mean MFE / mean MAE over fixed-horizon windows",["high","low","close"],"1d",1,None),
   ("F_PATH","run_length","current up/down run length in daily closes; extremes revert",["close"],"1d",1,None),
   ("F_PATH","drawup_speed","recent max drawup / days-to-achieve; momentum quality",["close"],"1d",1,None)]),
 ("Optimal Stopping and Free-Boundary Problems", "Peskir & Shiryaev", 2006, "10.1007/978-3-7643-7390-0", "isbn", "ADMISSIBLE", "prices", "exit timing",
  [("F_PATH","optimal_stop_ou","OU optimal exit boundary for a mean-reverting spread",["close"],"days",1,None)]),

 # ---- F_MULTI (multi-timeframe) ----
 ("HAR / Multi-Horizon Momentum Interaction", "practitioner", 2020, "10.2139/ssrn.3608379", "ssrn", "ADMISSIBLE", "prices", "TF agreement",
  [("F_MULTI","tf_agreement","sign agreement of momentum at 1d/3d/7d/21d; count",["close"],"1d",1,None),
   ("F_MULTI","fast_slow_divergence","fast-TF momentum minus slow-TF momentum",["close"],"1d",1,None)]),

 # ---- F_FORMULA (symbolic / GP) ----
 ("A Field Guide to Genetic Programming", "Poli, Langdon & McPhee", 2008, "isbn:9781409200734", "isbn", "ADMISSIBLE", "OHLCV features", "high overfitting risk -> strict OOS + surrogate",
  [("F_FORMULA","gp_alpha_search","genetic programming over {ret, vol, range, volume} operators; XS rank",["open","high","low","close","volume"],"1d",1,"grid = seeds; each candidate re-tested via full ladder")]),

 # ---- F_LIQ (liquidity / impact) ----
 ("Illiquidity and Stock Returns (Amihud) + Kyle lambda", "Amihud / Kyle", 2002, "10.1016/S1386-4181(01)00024-6", "doi", "ADMISSIBLE", "OHLCV + bookDepth", "impact state",
  [("F_LIQ","amihud_illiq_v2","mean(|r|/dollar_vol) trailing; XS sort long illiquid",["close","quote_volume"],"1d",1,"already in F_XSEC; keep here as impact state"),
   ("F_LIQ","kyle_lambda_bar","slope dP ~ signed sqrt(volume); trailing",["close","volume","taker_buy_base"],"1d",1,None),
   ("F_LIQ","depth_to_move_1pct","from bookDepth: notional needed to move price 1%; impact proxy",["bookDepth"],"1d",1,"bookDepth 2023-01..2024-05 only"),
   ("F_LIQ","roll_impact","Roll effective spread from trade-price autocovariance",["close"],"1d",1,None)]),

 # ---- F_BOOK (order book; bookDepth 2023-2024) ----
 ("The Price Impact of Order Book Events / Microprice", "Cont-Kukanov-Stoikov / Stoikov", 2019, "10.2139/ssrn.2970694", "ssrn", "PARKED", "bookDepth only 2023-01..2024-05; no full-window book", "microprice is a strong short-horizon predictor",
  [("F_BOOK","depth_imbalance","(bid_notional_1pct - ask_notional_1pct)/(sum); from bookDepth",["bookDepth"],"minutes-1h",1,"2023-2024 window only"),
   ("F_BOOK","book_slope","near-band / far-band cumulative depth ratio",["bookDepth"],"1h",1,None),
   ("F_BOOK","microprice_tilt","(bid*ask_qty + ask*bid_qty)/(bid_qty+ask_qty) vs mid; requires bookTicker",["bookTicker"],"seconds",0,"PARKED: bookTicker only to 2024-04"),
   ("F_BOOK","book_resiliency","recovery speed of depth after a shock",["bookDepth"],"minutes",1,None)]),

 # ---- F_OI / F_LIQD => REJECT_DATA ----
 ("Open Interest and Cryptocurrency Returns", "various", 2022, "10.2139/ssrn.4062981", "ssrn", "REJECT_DATA", "Binance publishes only 30 days of OI history in bulk / REST; a 2y backtest is impossible", "OI-price divergence claimed predictive",
  []),
 ("Liquidation Cascades in Crypto Perps", "various", 2023, "10.2139/ssrn.4531323", "ssrn", "REJECT_DATA", "no bulk liquidation stream history; forceOrder is real-time only", "cascades amplify moves",
  []),
]


def run():
    np = nf = 0
    for (title, auth, yr, ident, kind, adm, reason, fals, fmls) in P:
        pid = C.add_paper(title=title, authors=auth, year=yr, identifier=ident, id_kind=kind,
                          resolved=True, retrieved=True, admissibility=adm,
                          admissibility_reason=reason, falsification=fals, scan_depth="method")
        np += 1
        for (div, name, eq, inp, oh, ok, notes) in fmls:
            C.add_formula(pid, div, name, eq, inp, original_horizon=oh, data_ok=ok, notes=notes)
            nf += 1
    print(f"batch3: +{np} papers +{nf} formulas")
    import json
    print(json.dumps(C.counts(), indent=1))


if __name__ == "__main__":
    run()
