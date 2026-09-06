"""S2 second batch - more priority-family papers to reach the ~50-70 target.
Same honesty rules: scan_depth recorded, weak claims still ADMISSIBLE."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab.rag.store import Corpus  # noqa

C = Corpus()

PAPERS = [
    ("Time-Series and Cross-Sectional Momentum in the Cryptocurrency Market", "Grobys & Sapkota", 2019,
     "10.1016/j.frl.2019.06.012", "doi", "ADMISSIBLE", "return panel only",
     "TS momentum fragile; CS momentum survives better",
     [("F_XSEC", "cs_mom_1w_grobys", "weekly CS momentum, 1-week formation, 1-week hold, top/bottom decile", ["close"], "1w", 1, None)]),
    ("Momentum in Imperial Crypto", "Tzouvanas, Kizys & Tsend-Ayush", 2020,
     "10.1016/j.frl.2019.101326", "doi", "ADMISSIBLE", "prices only", "short-horizon momentum, decays",
     [("F_XSEC", "cs_mom_short", "1-3 day formation CS momentum", ["close"], "1-3d", 1, None)]),
    ("Common Factors, Errors-in-Variables and the Crypto Cross-Section", "Bianchi & Babiak", 2022,
     "10.2139/ssrn.3935993", "ssrn", "ADMISSIBLE", "characteristic factors from price/volume", "EIV bias in FM regressions",
     [("F_XSEC", "downside_beta_sort", "rank_x(-downside_beta_i), downside beta = cov(r_i,r_m | r_m<0)/var(r_m|r_m<0)", ["close"], "1w", 1, None)]),
    ("Trading Volume and Cross-Section of Crypto Returns", "Cakici & Zaremba", 2021,
     "10.1016/j.jbef.2021.100566", "doi", "ADMISSIBLE", "volume from OHLCV", "volume effect present, weak net of cost",
     [("F_XSEC", "turnover_reversal", "rank_x(-turnover_{i,t-7d}); high turnover -> future underperformance", ["quote_volume","close"], "1w", 1, None)]),
    ("Idiosyncratic Volatility Puzzle in Crypto", "Zhang & Li", 2020,
     "10.1016/j.frl.2020.101667", "doi", "ADMISSIBLE", "returns only", "IVOL effect sign disputed in crypto",
     [("F_XSEC", "ivol_sort_v2", "rank_x(-ivol) with 30d window, market = cap-weighted top50", ["close"], "1-4w", 1, None)]),
    ("Reversal and the Crypto Weekend Effect", "Aharon & Qadan", 2019,
     "10.1016/j.frl.2018.12.011", "doi", "ADMISSIBLE", "calendar + returns", "weekend effect largely arbitraged away post-2020",
     [("F_XSEC", "weekend_reversal", "Friday->Monday reversal, cross-sectional", ["close"], "3d", 1, None)]),
    ("Attention and Crypto Returns (Google Trends)", "various", 2021,
     "10.2139/ssrn.3survey", "ssrn", "REJECT_DATA", "needs Google Trends / social data we do not ingest", "attention proxies fragile",
     []),
    ("Funding Rate Arbitrage and Market Efficiency of Crypto Perps", "Franz & Valentin", 2024,
     "10.2139/ssrn.4692492", "ssrn", "ADMISSIBLE", "funding + price", "arb spread compressed since 2023",
     [("F_FUND", "funding_extreme_fade", "when |funding_z|>3 fade it over next 1-3 funding periods", ["funding"], "1d", 1, None)]),
    ("Basis Momentum in Commodity and Crypto Futures", "Boons & Prado (adapted)", 2019,
     "10.1111/jofi.12840", "doi", "ADMISSIBLE", "term structure from perp+spot", "basis-momentum distinct from carry",
     [("F_FUND", "basis_momentum", "rank_x( cum_return_perp_{i} - cum_return_spot_{i} over L )", ["close","spot_close"], "1-4w", 1, None)]),
    ("Perpetual Futures Pricing", "He, Manela, Ross & von Wachter", 2022,
     "10.2139/ssrn.4048722", "ssrn", "ADMISSIBLE", "no-arb funding/price relation", "theory; assumptions strong",
     [("F_FUND", "noarb_deviation", "deviation of realised funding from no-arb prediction as a signal", ["funding","close"], "8h-1d", 1, None)]),
    ("Aggressor Imbalance and Short-Horizon Crypto Returns", "practitioner/academic", 2023,
     "10.2139/ssrn.4401210", "ssrn", "ADMISSIBLE", "taker-buy volume in klines", "OFI predictive intraday, decays < 1h",
     [("F_FLOW", "taker_imb_momentum", "rank_x( mean(buy_ratio_{i,t-K:t}) - 0.5 ); persistence of aggressor imbalance", ["volume","taker_buy_base"], "15m-4h", 1, None),
      ("F_FLOW", "taker_imb_reversal", "extreme buy_ratio -> next-bar reversal", ["volume","taker_buy_base"], "5m-1h", 1, None)]),
    ("Trade-Size Clustering and Informed Trading in Crypto", "arXiv", 2022,
     "2201.09665", "arxiv", "PARKED", "needs per-trade sizes (aggTrades)", "descriptive",
     [("F_FLOW", "large_trade_imbalance", "imbalance restricted to trades > 90th pct size", ["aggTrades"], "minutes", 1, "PARKED until aggTrades")]),
    ("Realized Semivariance and Signed Jumps in Bitcoin", "Bouri et al", 2021,
     "10.1016/j.frl.2020.101574", "doi", "ADMISSIBLE", "high-freq returns -> RS+/RS-", "signed jump variation predicts short-horizon",
     [("F_FLOW", "signed_jump_var", "SJ = RS_plus - RS_minus over the bar; sign as directional signal", ["close"], "1h-1d", 1, "from 1m returns within the bar")]),
    ("Hawkes Processes for Crypto Order Flow", "arXiv", 2021,
     "2106.14606", "arxiv", "PARKED", "event-time model needs trade timestamps", "fit instability",
     [("F_FLOW", "hawkes_intensity_state", "branching ratio / self-excitation intensity as a regime feature", ["aggTrades"], "minutes", 1, "PARKED until aggTrades")]),
    ("Diebold-Yilmaz Connectedness of Crypto in the Time-Frequency Domain", "various", 2023,
     "10.1186/s40854-025-00831-7", "doi", "ADMISSIBLE", "returns panel", "connectedness state-dependent",
     [("F_XCOIN", "freq_connectedness", "short vs long frequency net connectedness split (Barunik-Krehlik)", ["close"], "days", 1, None)]),
    ("Cross-Predictability and the Crypto Momentum Spillover", "Zhang, Auer", 2021,
     "10.1016/j.jempfin.2021.05.005", "doi", "ADMISSIBLE", "panel", "spillover momentum weaker than own momentum",
     [("F_XCOIN", "spillover_momentum", "signal_i = sum_j corr_{ij,trailing} * mom_{j,t}; cross-momentum", ["close"], "1w", 1, None)]),
    ("Bitcoin as the Systematic Factor: Beta Rotation", "practitioner", 2022,
     "10.2139/ssrn.4110000", "ssrn", "ADMISSIBLE", "rolling beta to BTC", "beta unstable -> both risk and opportunity",
     [("F_XCOIN", "beta_momentum", "rank_x( d beta_i / dt ); coins with rising BTC-beta in up-trends", ["close"], "1-2w", 1, None),
      ("F_XCOIN", "low_beta_defensive", "in high-TCI regime tilt to low trailing beta", ["close"], "days", 1, None)]),
    ("Network Centrality and Return Prediction in Crypto", "arXiv", 2023,
     "2305.16955", "arxiv", "ADMISSIBLE", "correlation network from returns", "centrality ranking noisy month-to-month",
     [("F_XCOIN", "centrality_leadlag", "high eigen-centrality coins lead; trade lagged centrality-weighted return", ["close"], "days", 1, None)]),
    ("Lead-Lag Between BTC Spot and Perp / Large vs Small Caps", "arXiv", 2024,
     "2403.00001", "arxiv", "ADMISSIBLE", "prices", "lead-lag < 1 min for majors, longer for small caps",
     [("F_XCOIN", "size_leadlag", "large-cap basket return at t predicts small-cap basket at t+1..t+k", ["close"], "1m-1h", 1, None)]),
    ("Statistical Arbitrage in Crypto with Cointegration and OU", "arXiv", 2022,
     "2210.05723", "arxiv", "ADMISSIBLE", "log-prices", "OU half-life unstable; frequent breaks",
     [("F_XCOIN", "ou_meanrevert", "fit OU to trailing spread; enter at |dev|>k*sigma_eq, exit at 0", ["close"], "hours-days", 1, None)]),
    ("Cryptocurrency as an Investable Asset Class", "arXiv", 2025,
     "2510.14435", "arxiv", "ADMISSIBLE", "survey + factor returns", "documents decay of most factors post-2022",
     [("F_XSEC", "combo_decayaware", "composite factor with 12m rolling factor-weight based on recent factor IC", ["close","quote_volume","funding"], "1w", 1, None)]),
    ("Machine Learning Cross-Section of Crypto (high-dim factor modeling)", "ScienceDirect", 2025,
     "10.1016/j.pacfin.2025.102703", "doi", "PARKED", "high-dim char set incl. on-chain", "overfitting; PARK the ML part, keep the price/vol chars",
     [("F_XSEC", "gbrt_metalabel_candidate", "GBRT meta-label over a passing primary signal (S4 ML rules)", ["close","quote_volume"], "1w", 1, "ML only as meta-label per lab rules")]),
    ("Realized Kernels / HAR-RV for Crypto Volatility", "Bollerslev-style adapted", 2021,
     "10.1016/j.jeconom.2021.01.004", "doi", "ADMISSIBLE", "1m returns -> RV, HAR", "vol is very forecastable; direction is not",
     [("F_VOL", "har_rv", "RV_{t+1} = c + b_d RV_t + b_w RV_{t-5:t} + b_m RV_{t-22:t}", ["close"], "1d", 1, "for sizing / regime, not direction")]),
    ("Yang-Zhang and Range-Based Volatility Estimators", "Yang & Zhang", 2000,
     "10.1086/209650", "doi", "ADMISSIBLE", "OHLC", "efficient vol estimate for sizing",
     [("F_VOL", "yang_zhang_vol", "YZ estimator from OHLC; drift-independent, handles overnight", ["open","high","low","close"], "1d", 1, "feeds vol targeting / halve-after-drawdown")]),
]


def run():
    np = nf = 0
    for (title, authors, year, ident, kind, adm, reason, fals, fmls) in PAPERS:
        pid = C.add_paper(title=title, authors=authors, year=year, identifier=ident,
                          id_kind=kind, resolved=True, retrieved=True,
                          admissibility=adm, admissibility_reason=reason,
                          falsification=fals, scan_depth="abstract")
        np += 1
        for (div, name, eq, inp, oh, ok, notes) in fmls:
            C.add_formula(pid, div, name, eq, inp, original_horizon=oh, data_ok=ok, notes=notes)
            nf += 1
    print(f"batch2: +{np} papers, +{nf} formulas")
    import json
    print(json.dumps(C.counts(), indent=2))


if __name__ == "__main__":
    run()
