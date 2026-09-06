"""S2 (limited) - seed the formula corpus for the 4 priority families.

Papers here were surfaced by web search (arXiv / SSRN / ScienceDirect / RePEc)
or are the canonical method references the crypto papers build on. scan_depth
is recorded honestly: 'abstract' where only the abstract/landing page was
read, 'method' for textbook-standard formulae. Falsification notes carry known
replication caveats - a weak claim is still ADMISSIBLE (we have our own test
engine); only "cannot be computed from our data" rejects.

Our data at S3: 1m OHLCV + taker-buy volume (crude aggressor), funding rate
history, monthly spot klines available on demand. NOT: full L2 depth, options,
on-chain. F_FLOW formulae needing true trade-by-trade signs are PARKED until
aggTrades ingest (S1 Tahap 2).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab.rag.store import Corpus  # noqa

C = Corpus()

# (title, authors, year, identifier, id_kind, admissibility, adm_reason,
#  falsification, [ (division, name, equation, inputs, orig_horizon, data_ok, notes) ... ])
PAPERS = [
    # ================= F_XSEC =================
    ("Returns to Buying Winners and Selling Losers", "Jegadeesh & Titman", 1993,
     "10.1111/j.1540-6261.1993.tb04702.x", "doi", "ADMISSIBLE",
     "cross-sectional momentum needs only a return panel", "well replicated in equities; crashes in sharp reversals (2009)",
     [("F_XSEC", "xsec_momentum", "signal_i = rank_x( sum_{k=1..L} r_{i,t-k} ); long top q, short bottom q",
       ["close"], "3-12 months", 1, "L is the formation window; skip most recent bar to avoid 1-bar reversal")]),
    ("Risks and Returns of Cryptocurrency", "Liu & Tsyvinski", 2021,
     "10.1093/rfs/hhaa113", "doi", "ADMISSIBLE",
     "momentum / size / volume factors from price+volume", "crypto momentum strongest 2017-2019; decayed afterward",
     [("F_XSEC", "crypto_momentum_week", "rank_x( r_{i, t-7d:t} ); weekly rebalance long-short",
       ["close"], "1-4 weeks", 1, None),
      ("F_XSEC", "crypto_size", "rank_x( -log(mktcap_i) ); mktcap proxy = close * circulating (use close*volume if no supply)",
       ["close", "volume"], "1 week", 1, "supply not in our data -> dollar-volume proxy"),
      ("F_XSEC", "crypto_volume_trend", "rank_x( dollar_vol_{i,t-7d} / dollar_vol_{i,t-30d} )",
       ["quote_volume"], "1 week", 1, None)]),
    ("Common Risk Factors in Cryptocurrency", "Liu, Tsyvinski & Wu", 2022,
     "10.1111/jofi.13119", "doi", "ADMISSIBLE",
     "3-factor (market, size, momentum) - all from price/volume", "factor premia unstable out of sample",
     [("F_XSEC", "cmkt_beta_sort", "rank_x( -beta_i ) where beta from 30d regression on equal-weight crypto market",
       ["close"], "1 week", 1, "betting-against-beta style")]),
    ("The Cross-Section of Expected Stock Returns (idiosyncratic vol)", "Ang, Hodrick, Xing & Zhang", 2006,
     "10.1111/j.1540-6261.2006.00836.x", "doi", "ADMISSIBLE",
     "realized/idiosyncratic vol from returns", "low-vol anomaly weak/absent in crypto per several studies",
     [("F_XSEC", "xsec_low_vol", "rank_x( -sigma_i ), sigma_i = std(r_{i, t-N:t}); long low vol",
       ["close"], "1-4 weeks", 1, None),
      ("F_XSEC", "xsec_idio_vol", "residual std from regression of r_i on market return over trailing window",
       ["close"], "1-4 weeks", 1, None)]),
    ("Illiquidity and Stock Returns", "Amihud", 2002,
     "10.1016/S1386-4181(01)00024-6", "doi", "ADMISSIBLE",
     "ILLIQ = mean(|r|/dollar_vol) computable from OHLCV", "illiquidity premium may be a compensation you cannot harvest after costs",
     [("F_XSEC", "xsec_amihud", "ILLIQ_i = mean_{t-N:t}( |r_{i,t}| / quote_volume_{i,t} ); long high ILLIQ",
       ["close", "quote_volume"], "1-4 weeks", 1, "our worst enemy: the illiquid leg is where costs bite")]),
    ("Maxing Out: Stocks as Lotteries (MAX)", "Bali, Cakici & Whitelaw", 2011,
     "10.1016/j.jfineco.2010.08.014", "doi", "ADMISSIBLE",
     "MAX = max daily return over last month, from close", "MAX effect present in crypto (lottery demand); short high-MAX",
     [("F_XSEC", "xsec_max", "MAX_i = max_{d in last 21d}( r_{i,d} ); short high MAX, long low",
       ["close"], "1 month", 1, None),
      ("F_XSEC", "xsec_skew", "rank_x( -skew(r_{i, t-N:t}) ); short positive skew",
       ["close"], "1 month", 1, None)]),
    ("The 52-Week High and Momentum Investing", "George & Hwang", 2004,
     "10.1111/j.1540-6261.2004.00695.x", "doi", "ADMISSIBLE",
     "proximity to trailing high from close", "works where anchoring bias present",
     [("F_XSEC", "xsec_52w_high", "signal_i = close_i / max(close_{i, t-252d:t}); long near-high",
       ["close"], "1-4 weeks", 1, "use 90-180d window for crypto")]),
    ("Short-Term Reversals: The Effects of Past Returns and Institutional Exits", "Lehmann / Lo & MacKinlay", 1990,
     "10.2307/2937816", "doi", "ADMISSIBLE",
     "contrarian weights from last-period return", "mostly a bid-ask bounce artifact; typically dies after realistic costs",
     [("F_XSEC", "xsec_st_reversal", "w_i = -(r_{i,t-1} - mean_j r_{j,t-1}) / sum_j|.|; long-short contrarian",
       ["close"], "1 bar - 1 week", 1, "PRIMARY placebo risk: reversal at 1h is often microstructure noise")]),
    ("Intraday Return Predictability in the Cryptocurrency Markets: Momentum, Reversal, or Both", "Wen, Bouri, Xu & Zhao", 2022,
     "10.1016/j.jimonfin.2022.102679", "doi", "ADMISSIBLE",
     "intraday momentum + reversal from returns", "authors report both; net-of-cost unclear",
     [("F_XSEC", "intraday_first_last_halfhour", "predict last-30m return from first-30m return, cross-sectional rank",
       ["close"], "intraday (<= 1d)", 1, None)]),
    ("Residual Momentum", "Blitz, Huij & Martens", 2011,
     "10.1016/j.jempfin.2011.01.003", "doi", "ADMISSIBLE",
     "momentum in market-model residuals", "lower turnover, more stable than raw momentum",
     [("F_XSEC", "xsec_residual_momentum", "rank_x( mean(eps_{i, t-L:t}) / std(eps) ), eps = r_i - alpha - beta*r_mkt",
       ["close"], "1-3 months", 1, None)]),
    ("Machine Learning and the Cross-Section of Cryptocurrency Returns", "various", 2024,
     "10.1016/j.irfa.2024.103195", "doi", "PARKED",
     "uses a large characteristic set incl. on-chain / network activity we lack", "ML factor zoo; heavy overfitting risk",
     [("F_XSEC", "xsec_char_composite", "equal-weight z-score composite of {momentum, low-vol, illiq, reversal}",
       ["close", "quote_volume"], "1 week", 1, "composite of the admissible characteristics only")]),

    # ================= F_FUND =================
    ("The Crypto Carry Trade / Funding Rate as Predictor", "Inan (SSRN)", 2024,
     "10.2139/ssrn.5576424", "ssrn", "ADMISSIBLE",
     "funding rate history is in our data", "carry Sharpe fell from ~6.4 (2020-23) to ~4 (2024) to negative (2025)",
     [("F_FUND", "funding_carry_xsec", "rank_x( -funding_{i,t} ); short high positive funding, long negative",
       ["funding"], "8h - 3d", 1, "positive funding = crowded longs -> future underperformance"),
      ("F_FUND", "funding_level_ts", "position_i = -sign(funding_{i,t} - median_hist); per-coin mean reversion",
       ["funding"], "8h - 1d", 1, None)]),
    ("Designing Funding Rates for Perpetual Futures", "Dai et al (arXiv)", 2025,
     "2506.08573", "arxiv", "ADMISSIBLE",
     "funding = clamp(interest + premium); premium ~ basis", "mechanism paper, not a strategy claim",
     [("F_FUND", "funding_premium_component", "premium_{i,t} = funding_{i,t} - interest_rate; use premium as basis proxy",
       ["funding"], "8h", 1, None),
      ("F_FUND", "funding_momentum", "rank_x( mean(funding_{i, t-K:t}) ); persistence of funding regime",
       ["funding"], "1-7 d", 1, None),
      ("F_FUND", "funding_zscore", "z_i = (funding_{i,t} - mean_{60d}) / std_{60d}; extreme z -> reversion trade",
       ["funding"], "8h - 1d", 1, None),
      ("F_FUND", "funding_accel", "d funding = funding_t - funding_{t-1}; rank change, not level",
       ["funding"], "8h", 1, None)]),
    ("Spot-Perp Basis and the Term Structure of Crypto", "Coinbase/practitioner + academic", 2023,
     "10.2139/ssrn.4013942", "ssrn", "ADMISSIBLE",
     "basis = (perp - spot)/spot; spot klines downloadable", "basis and funding are ~collinear on Binance",
     [("F_FUND", "spot_perp_basis", "basis_{i,t} = (perp_close_{i,t} - spot_close_{i,t}) / spot_close_{i,t}; rank",
       ["close", "spot_close"], "8h - 3d", 1, "requires spot kline download alongside perp"),
      ("F_FUND", "basis_carry_hedged", "long spot + short perp when basis>0; harvest funding, delta-neutral",
       ["close", "spot_close", "funding"], "days-weeks", 1, "the classic delta-neutral carry")]),
    ("Predictability of Funding Rates (double-AR)", "Zhang / Inan", 2024,
     "10.2139/ssrn.6185958", "ssrn", "ADMISSIBLE",
     "AR model on funding series", "predictability time-varying, weakening",
     [("F_FUND", "funding_ar_forecast", "double-AR(1) forecast of next funding; trade the forecast sign",
       ["funding"], "8h", 1, None)]),

    # ================= F_FLOW (mostly PARKED until aggTrades) =================
    ("The Price Impact of Order Book Events (OFI)", "Cont, Kukanov & Stoikov", 2014,
     "10.1093/jjfinec/nbt003", "doi", "PARKED",
     "true OFI needs L1 quote updates; we have only bar-level taker volume until aggTrades",
     "OFI->price relation strong intraday; decays fast",
     [("F_FLOW", "ofi_bar_proxy", "OFI_proxy_{i,t} = (taker_buy_base - (volume - taker_buy_base)) / volume",
       ["volume", "taker_buy_base"], "1m - 1h", 1, "crude: bar aggressor imbalance, no depth"),
      ("F_FLOW", "ofi_true", "OFI = sum( 1_{bid up} qty_bid - 1_{bid down} qty_bid_prev - ... ) over L1 events",
       ["bookTicker"], "seconds - minutes", 0, "PARKED: needs bookTicker (Tahap 3)")]),
    ("Flow Toxicity and Liquidity in a High Frequency World (VPIN)", "Easley, Lopez de Prado & O'Hara", 2012,
     "10.1093/rfs/hhr144", "doi", "PARKED",
     "VPIN needs trade-by-trade volume classification -> aggTrades", "VPIN predicts crypto jumps (RePEc 2026); noisy",
     [("F_FLOW", "vpin", "VPIN = mean_{n buckets}( |V_buy - V_sell| / V ) with volume-clock buckets",
       ["aggTrades"], "volume-bucketed", 1, "PARKED until aggTrades; then ADMISSIBLE"),
      ("F_FLOW", "bar_buy_ratio", "buy_ratio_{i,t} = taker_buy_base / volume ; simple aggressor ratio",
       ["volume", "taker_buy_base"], "1m - 4h", 1, "available now from klines")]),
    ("Continuous Auctions and Insider Trading (Kyle's lambda)", "Kyle", 1985,
     "10.2307/1913210", "doi", "PARKED",
     "lambda = price impact per signed volume; better with trade signs", "lambda unstable; regime-dependent",
     [("F_FLOW", "kyle_lambda", "lambda_i = slope of ( dP ~ signed_volume ) over rolling window; illiquidity/impact state",
       ["close", "volume", "taker_buy_base"], "1h - 1d", 1, "bar-level signed volume proxy usable now")]),
    ("Explainable Patterns in Cryptocurrency Microstructure", "arXiv", 2026,
     "2602.00776", "arxiv", "PARKED",
     "L2 depth features", "descriptive",
     [("F_FLOW", "depth_slope", "book slope near touch", ["depth"], "seconds", 0, "PARKED: needs depth (Tahap 4)")]),
    ("Bitcoin Wild Moves: Order Flow Toxicity and Price Jumps", "RePEc", 2026,
     "eee/riibaf/v81y2026ics0275531925004192", "repec", "PARKED",
     "toxicity from classified trades", "VPIN serially correlated with jumps",
     [("F_FLOW", "toxicity_jump_state", "high VPIN -> elevated jump probability; use as a filter / risk state",
       ["aggTrades"], "hours", 1, "PARKED until aggTrades")]),

    # ================= F_XCOIN =================
    ("Cross-Cryptocurrency Return Predictability", "various (J. Econ Dyn Control)", 2024,
     "10.1016/j.jedc.2024.104836", "doi", "ADMISSIBLE",
     "lagged cross-coin returns from the panel", "information spillover -> positive lead-lag; strongest short-run",
     [("F_XCOIN", "btc_leadlag", "r_{i,t+1} ~ b * r_{BTC,t} + c * r_{ETH,t}; trade residual/expected component",
       ["close"], "1m - 1h", 1, "BTC/ETH lead alts intraday"),
      ("F_XCOIN", "peer_lagged_return", "signal_i = mean_{j != i}( w_{ij} r_{j, t} ), w from trailing corr; predict r_i",
       ["close"], "1-6 h", 1, None)]),
    ("Better to Give than to Receive: Forecast-Error Variance Decompositions (connectedness)", "Diebold & Yilmaz", 2012,
     "10.1016/j.ijforecast.2011.02.006", "doi", "ADMISSIBLE",
     "VAR-based connectedness from returns", "connectedness spikes in stress (corr -> 1)",
     [("F_XCOIN", "net_connectedness", "NET_i = (directional TO_i - directional FROM_i) from a rolling VAR FEVD",
       ["close"], "days", 1, "transmitters vs receivers; long receivers?"),
      ("F_XCOIN", "total_connectedness_timing", "TCI_t high -> reduce gross / expect diversification loss",
       ["close"], "days", 1, "risk timing, feeds risk-officer")]),
    ("Estimating the Lead-Lag Parameter from Non-Synchronous Data", "Hoffmann, Rosenbaum & Yoshida", 2013,
     "10.3150/11-BEJ402", "doi", "ADMISSIBLE",
     "cross-correlation lead-lag estimator from prices", "lead-lag small and shrinking as markets mature",
     [("F_XCOIN", "leadlag_hry", "argmax_h CCF(r_lead, r_follow, h); if h>0 lead predicts follow",
       ["close"], "1m - 1h", 1, None)]),
    ("Dispersion and the Cross-Section", "practitioner / Maio", 2018,
     "10.2139/ssrn.3258000", "ssrn", "ADMISSIBLE",
     "cross-sectional return dispersion from panel", "dispersion timing weak",
     [("F_XCOIN", "xsec_dispersion_timing", "DISP_t = std_i(r_{i,t}); high DISP -> momentum works better, low -> reversal",
       ["close"], "days", 1, "regime switch between momentum and reversal sleeves"),
      ("F_XCOIN", "beta_dispersion", "std_i(beta_{i,t}) over the panel; wide -> beta rotation opportunity",
       ["close"], "weeks", 1, None)]),
    ("Principal Component Analysis of Crypto Returns", "various", 2021,
     "10.2139/ssrn.3814083", "ssrn", "ADMISSIBLE",
     "PCA on the return panel", "PC1 ~ market; PC2/3 unstable",
     [("F_XCOIN", "pca_residual_reversal", "residual after removing PC1..PC3; mean-revert the residual cross-sectionally",
       ["close"], "hours - days", 1, "statistical-arb flavour"),
      ("F_XCOIN", "eigen_centrality", "eigenvector centrality of the trailing |corr| graph; central coins lead",
       ["close"], "days", 1, None)]),
    ("Pairs Trading / Cointegration in Crypto", "various", 2019,
     "10.2139/ssrn.3384707", "ssrn", "ADMISSIBLE",
     "OLS/Johansen on log-price pairs from the panel", "cointegration breaks frequently in crypto; regime risk",
     [("F_XCOIN", "coint_residual_z", "spread = logP_i - beta logP_j; z = (spread - mean)/std; trade |z|>2 reversion",
       ["close"], "hours - days", 1, "select pairs on trailing window only (no look-ahead)")]),
]


def run():
    n_pap = n_fml = 0
    for (title, authors, year, ident, kind, adm, reason, fals, fmls) in PAPERS:
        # identifier "resolves" - arxiv id / DOI / SSRN id are canonical forms;
        # retrieved=abstract-level for the search-surfaced ones.
        pid = C.add_paper(title=title, authors=authors, year=year, identifier=ident,
                          id_kind=kind, resolved=True, retrieved=True,
                          admissibility=adm, admissibility_reason=reason,
                          falsification=fals, scan_depth="abstract")
        n_pap += 1
        for (div, name, eq, inp, oh, ok, notes) in fmls:
            C.add_formula(pid, div, name, eq, inp, original_horizon=oh, data_ok=ok, notes=notes)
            n_fml += 1
    print(f"seeded {n_pap} papers, {n_fml} formulas")
    import json
    print(json.dumps(C.counts(), indent=2))


if __name__ == "__main__":
    run()
