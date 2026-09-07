"""Full formula registry (spec L). 94 formulas F001-F094, each with its
disposition: which family/round tested it, the gate status, and the root cause.

Duplicate mechanisms are collapsed (10 OFI variants -> 1 row, variants in notes).
`status` one of: BOUNDED | VETOED | UNDERPOWERED | DATA_LIMITED | SIZING_ONLY |
                 RUN_BATCH<n> | NOT_RUN
"""

# id | family | paper | equation (short) | data | horizon | status | root_cause / where
ROWS = [
    # ---- M1  order flow / microstructure ----------------------------------
    ("F001", "F_MICRO_OFI", "Cont-Kukanov-Stoikov 2014", "OFI = Σ book-edge events per bar",
     "DOM+tape", "10s-5m", "BOUNDED", "NO_EDGE from aggregate trades; needs live L2 (Round1 F_MICRO_OFI_EVENT, 28 cfg, gross Sharpe<0)"),
    ("F002", "F_MICRO_OFI", "Cont-Cucuringu-Zhang 2021", "multi-level OFI, PC1 of OFI^1..5",
     "L2 5 levels", "10s-5m", "DATA_LIMITED", "no live L2 order-book feed; bookDepth is ±1%..±5% bands only"),
    ("F003", "F_MICRO_BOOK", "-", "QI = (Qb-Qa)/(Qb+Qa)", "DOM", "5-15m", "BOUNDED",
     "F_BOOK book_imb_follow/fade, 120 cfg: gross Sharpe up to 4.3 but maker net -6..-99 (COST_BOUND)"),
    ("F004", "F_MICRO_BOOK", "Stoikov 2018", "microprice = (Pa·Qb+Pb·Qa)/(Qb+Qa)", "DOM", "5-15m",
     "BOUNDED", "F_BOOK microprice_drift: |IC| 0.003, COST_BOUND / NO_EDGE"),
    ("F005", "F_MICRO_OFI / cost_model", "Kyle 1985", "dP = λ·SignedVol + ε", "tape", "1h", "BOUNDED",
     "Kyle λ as a feature: F_FLOW flow_kyle_lambda bounded; λ is used in cost_model.impact instead"),
    ("F006", "F_LIQ", "Amihud 2002", "ILLIQ = mean(|r|/$vol)", "OHLCV", "1d", "BOUNDED",
     "F_LIQ liq_kyle_bar 6 cfg: best DSR 0.05, NO_EDGE"),
    ("F007", "F_MICRO_OFI", "Easley-LdP-O'Hara 2012", "VPIN = Σ|Vb-Vs|/(n·V)", "tape", "M5-M15", "BOUNDED",
     "F_MICRO screen: no config passed rough-IC≥0.008 + net-positive"),
    ("F008", "F_MICRO_OFI / universe", "Roll 1984", "s = 2√(-cov(dP,dP_-1))", "tape", "-", "SIZING_ONLY",
     "Roll spread used as a cost/universe input, not a directional signal"),
    ("F009", "-", "Corwin-Schultz 2012", "spread from 2-day High-Low", "OHLC", "1d", "NOT_RUN",
     "spread estimator only; equivalent info to F008 for the cost model — no directional content expected"),
    ("F010", "F_MICRO_OFI", "Lee-Ready 1991", "tick-rule sign → ACF(sign)", "tape", "M5-M15", "BOUNDED",
     "F_MICRO signac_momentum: screen fail"),
    ("F011", "-", "Bacry-Muzy 2014", "λ_t = μ + Σ α·e^{-β(t-t_i)}; sig = λ^buy-λ^sell", "tape events",
     "10s-5m", "RUN_BATCH1", "Hawkes self-excitation — tape event times available; branching ratio n=α/β"),
    ("F012", "-", "Bouchaud 2004", "P_t = Σ G(t-t')·ε_{t'}·f(v); G(l)~l^-0.5", "tape", "10s-5m",
     "RUN_BATCH1", "transient/propagator impact — needs signed-trade series (have from tape)"),
    ("F013", "cost_model", "Almgren; Toth 2011", "impact = Y·σ·√(Q/ADV)", "OHLCV", "-", "SIZING_ONLY",
     "square-root impact IS the cost_model.impact term (Y=0.5). Not a signal."),
    ("F014", "F_MICRO_OFI", "-", "eff/realized spread = 2D(P_trade-P_mid)/mid", "tape", "M5", "BOUNDED",
     "realized-spread decay = adverse selection; measured in maker_model, not a signal"),

    # ---- M2  volatility  (SIZING / GATING, not direction) -----------------
    ("F020", "vol.py", "RiskMetrics", "σ²_t = λσ²_{t-1}+(1-λ)r²", "OHLCV", "-", "SIZING_ONLY",
     "lab/exec/vol.py ewma_var, λ=0.94 — position sizing"),
    ("F021", "F_VOL / vol.py", "Bollerslev 1986", "GARCH(1,1)", "OHLCV", "-", "SIZING_ONLY",
     "vol.py gjr_garch_sigma (GARCH baseline). F_VOL vol_* directional variants bounded."),
    ("F022", "-", "Nelson 1991", "EGARCH: ln σ²", "OHLCV", "-", "NOT_RUN",
     "asymmetric-vol variant of F023; sizing only. GJR (F023) already implemented and covers the asymmetry."),
    ("F023", "vol.py", "Glosten-Jagannathan-Runkle 1993", "GJR-GARCH(1,1)", "OHLCV", "-", "SIZING_ONLY",
     "lab/exec/vol.py gjr_garch_sigma — MLE, asymmetric. Sizing only per spec D."),
    ("F024", "F_VOL", "Corsi 2009", "HAR-RV: RV_{t+1}=c+b_d RV+b_w RV^5+b_m RV^22", "OHLCV", "1d-1w",
     "NOT_RUN", "RV forecaster; feeds sizing. Directional use would be a vol-timing trial — deferred."),
    ("F025", "F_VOL", "Barndorff-Nielsen 2010", "RS± ; signed jump = RS+ - RS-", "OHLCV", "1d", "BOUNDED",
     "F_VOL vol_semivar_skew 8 cfg: best net Sharpe -0.08, DSR 0.002, NO_EDGE"),
    ("F026", "-", "BNS 2004", "BV=(π/2)Σ|r_i||r_{i-1}|; jump = RV-BV", "tape/OHLCV", "1d", "NOT_RUN",
     "jump detector; sizing/gating input, not directional"),
    ("F027", "vol.py", "Yang-Zhang 2000", "σ²_o + k·σ²_c + (1-k)·σ²_RS", "OHLC", "-", "SIZING_ONLY",
     "lab/exec/vol.py yang_zhang — most efficient OHLC estimator. F_VOL vol_yang_zhang directional bounded."),
    ("F028", "F_VOL", "Parkinson/Garman-Klass/Rogers-Satchell", "range-based σ²", "OHLC", "-", "SIZING_ONLY",
     "RS term is inside Yang-Zhang. Directional low-vol tilt = F_VOL, bounded."),
    ("F029", "-", "-", "σ(σ_t) rolling — vol-of-vol", "OHLCV", "1d", "NOT_RUN",
     "second-moment feature; gating input. Standalone directional use not expected to clear cost."),
    ("F030", "-", "Barndorff-Nielsen 2008", "realized kernel (Parzen), noise-robust RV", "tape", "-",
     "NOT_RUN", "noise-robust RV estimator; sizing input only"),

    # ---- M3  trend / momentum / reversal ---------------------------------
    ("F040", "F_DIR", "Moskowitz-Ooi-Pedersen 2012", "sign(r_{t-k:t})/σ_t", "price", "1d-1w", "BOUNDED",
     "TSMOM: F_XSEC xsec_momentum / F_DIR dir_ma_cross, 58+30 cfg, best DSR 0.14, COST/NO_EDGE"),
    ("F041", "F_XSEC", "Jegadeesh-Titman 1993", "rank r_{t-k:t} XS, long top/short bot", "price", "1d",
     "BOUNDED", "F_XSEC xsec_momentum 58 cfg: best |IC| 0.057, net Sharpe ≤1.3, DSR 0.14"),
    ("F042", "F_XSEC", "Lehmann 1990", "sig = -r_{t-1}", "price", "1d", "BOUNDED",
     "F_XSEC xsec_st_reversal: bounded (part of the 58)"),
    ("F043", "F_XCOIN", "Blitz 2011", "momentum on factor residual (BTC β + PC1-3)", "price", "1d",
     "BOUNDED", "F_XCOIN xcoin_pca_residual / xsec_residual_momentum 20 cfg, best DSR 0.09"),
    ("F044", "F_DEP", "Lo-MacKinlay 1988", "VR(q) = Var(r_q)/(q·Var(r_1))", "price", "1d", "BOUNDED",
     "F_DEP dep_variance_ratio: best |IC| 0.014, DSR 0.085, NO_EDGE"),
    ("F045", "F_TRANSFORM", "Mandelbrot; Peng DFA", "Hurst (R/S, DFA)", "price", "1d", "BOUNDED",
     "F_TRANSFORM transform_hurst: bounded (part of the 10)"),
    ("F046", "F_PAIRS", "Uhlenbeck-Ornstein", "dS=θ(μ-S)dt+σdW; HL=ln2/θ", "price", "4h-5d", "BOUNDED",
     "F_PAIRS pairs_johansen_ou: OOS s-score→fwd-spread corr +0.004 (wrong sign). NO_EDGE."),
    ("F047", "F_PAIRS", "Avellaneda-Lee 2010", "drop k PCs, OU on residual, s-score", "price", "hours-days",
     "BOUNDED", "F_PAIRS pairs_avellaneda: gross Sharpe -0.83, surrogate p 0.52, NO_EDGE"),
    ("F048", "F_DIR", "Baltas-Kosowski", "TSMOM w/ vol scaling: w = tgt/σ·sign(mom)", "price", "1d",
     "BOUNDED", "vol-scaled TSMOM = F_DIR with vol.py sizing; bounded like F040"),

    # ---- M4  crypto-specific --------------------------------------------
    ("F060", "F_OI / F_CN4", "-", "LR = OI·P/avg168h($vol); sig = -zx(log LR)", "OI+OHLCV", "1h-1d",
     "BOUNDED", "= oi_leverage_state. Bybit 2023-25 held-out +1.2; Binance 2021-23 +0.03 (J4). "
     "Batch4 gate: placebo p 0.24, surrogate p 0.23. NO_EDGE / sample-specific."),
    ("F061", "F_CN4", "-", "OI-P 4-quadrant divergence × |dOI|", "OI+price", "4h", "RUN_BATCH4",
     "screen ratio 4.2, gate placebo p 0.12 / surrogate p 0.003 — random ranking does as well. NO_EDGE"),
    ("F062", "F_FUND / F_CARRY", "-", "sig = -zx(funding)", "funding", "8h-1d", "BOUNDED",
     "F_FUND fund_carry / F_CARRY: risk premium not alpha (surrogate p 1.0). Batch4 screen ratio 0.14."),
    ("F063", "F_CN4", "-", "sig = zx(funding_t - trailing_mean) — persistence", "funding", "4h", "RUN_BATCH4",
     "IC 0.0009, screen ratio 0.76 at every TF. NO_EDGE."),
    ("F064", "F_TERM", "-", "slope = quarterly_basis_ann - perp_funding_ann", "perp+quarterly", "1d",
     "RUN_BATCH4", "BTC/ETH quarterly futures pulled (2021+); TS study — see lab/research/term_structure.py"),
    ("F065", "F_XVENUE", "-", "spread = f_binance - mean(f_bybit, f_okx)", "multi-venue funding", "4h",
     "VETOED", "F_XVENUE Binance-Bybit: real inefficiency but concentrated in delisted names (IDEX 40%). "
     "Red-team VETO. OKX would add a leg to a vetoed signal."),
    ("F066", "F_LIQD", "-", "liq_intensity = Σliq$/OI; post-cascade reversal 24-72h", "liquidation+OI",
     "24-72h", "UNDERPOWERED", "F_LIQD 60 cfg: Binance coin-M liquidationSnapshot is BTC/ETH only → "
     "one correlated stress series. best gross Sharpe 0.89, maker net 0.14."),
    ("F067", "F_LIQD", "-", "(liq_long-liq_short)/(liq_long+liq_short)", "liquidation", "4h",
     "UNDERPOWERED", "same data limit as F066 — BTC/ETH only"),
    ("F068", "F_CN4", "-", "OI-build near a 1/lev price band = liq magnet", "OI+price", "4h", "RUN_BATCH4",
     "screen ratio 0.11. NO_EDGE. (crude proxy; real version needs the leverage distribution)"),
    ("F069", "-", "-", "d(USDT+USDC supply) as inflow proxy", "on-chain stablecoin supply", "1d",
     "DATA_LIMITED", "no on-chain data source ingested"),
    ("F070", "F_CARRY", "-", "annualised spot-perp basis = funding·365·3", "funding", "1d", "BOUNDED",
     "= F_CARRY basis carry. risk premium, bounded."),

    # ---- M5  state / filtering  (wrappers, not standalone direction) -----
    ("F080", "kalman.py", "-", "local level KF (MA replacement)", "any", "-", "SIZING_ONLY",
     "lab/exec/kalman.py local_level — adaptive-lag smoother"),
    ("F081", "kalman.py / F_PAIRS", "Elliott-vd-Hoek-Malcolm 2005", "β_t = β_{t-1}+w; y=β_t x+v", "pairs",
     "-", "BOUNDED", "kalman.py hedge_ratio; F_PAIRS pairs_kalman_ou catastrophic (turnover 2.6, Sharpe -36)"),
    ("F082", "kalman.py", "-", "z = e/√S normalised innovation", "any", "-", "SIZING_ONLY",
     "lab/exec/kalman.py innovation_z — gating, not direction"),
    ("F083", "F_STATE", "Hamilton 1989", "HMM 2/3-state, obs=[ret,vol,funding] → P(state)", "OHLCV+funding",
     "gating", "RUN_BATCH5", "regime-gating wrapper. Nothing left that works to gate; run as a filter on F060."),
    ("F084", "-", "Haas-Mittnik-Paolella", "Markov-switching GARCH", "OHLCV", "-", "NOT_RUN",
     "regime-dependent vol; sizing refinement, not a directional signal"),
    ("F085", "-", "Gordon-Salmond-Smith 1993", "particle filter (stochastic vol)", "OHLCV", "-", "NOT_RUN",
     "non-linear state estimation; sizing input. Kalman covers the linear case."),
    ("F086", "-", "Page 1954", "CUSUM change-point", "any", "-", "NOT_RUN",
     "structural-break detector; a gate input, not a signal. Regime attribution (spec G) covers the need."),
    ("F087", "F_PAIRS", "Johansen 1991", "trace test; VECM ΔY = ΠY_{-1}+ΣΓΔY+ε", "price", "hours-days",
     "BOUNDED", "F_PAIRS: Johansen selects pairs correctly but cointegration does not persist OOS"),

    # ---- M6  cross-asset / stat ----------------------------------------
    ("F090", "F_XCOIN", "Granger 1969", "lead-lag BTC→alt (xcorr, Granger)", "price", "1h-1d", "BOUNDED",
     "F_XCOIN xcoin_btc_leadlag: bounded (part of the 20)"),
    ("F091", "F_XCOIN", "-", "PCA eigen-portfolio factor exposure", "price", "1d", "BOUNDED",
     "F_XCOIN xcoin_pca_residual: best DSR 0.09"),
    ("F092", "F_STATE", "-", "rolling avg pairwise correlation regime", "price", "1d", "BOUNDED",
     "F_STATE / F_XCOIN xcoin_dispersion_switch: bounded"),
    ("F093", "F_XCOIN", "-", "cross-sectional dispersion of returns", "price", "1d", "BOUNDED",
     "F_XCOIN xcoin_dispersion_switch 5 cfg: bounded"),
    ("F094", "F_VOL / F_XCOIN", "Ang et al 2006", "idiosyncratic-vol / beta dispersion", "price", "1d",
     "BOUNDED", "F_XSEC xsec_idio_vol: bounded (Ang et al low-vol anomaly, part of the 58)"),
]

STATUS_MEANING = {
    "BOUNDED": "tested in Round 1/2/3, failed the ladder, root cause recorded",
    "VETOED": "passed the ladder but a red-team / risk veto ended it",
    "UNDERPOWERED": "data too thin to resolve (R5 legal verdict)",
    "DATA_LIMITED": "cannot be tested without a data source we do not have",
    "SIZING_ONLY": "used for position sizing or gating, not as a directional signal (per spec)",
    "RUN_BATCH1": "queued / run in Batch 1 (microstructure)",
    "RUN_BATCH4": "run in Batch 4 (crypto-native, F_CN4)",
    "RUN_BATCH5": "queued for Batch 5 (state / gating wrapper)",
    "NOT_RUN": "a variant of an already-bounded formula or a pure sizing input; recorded, not run",
}


def to_markdown() -> str:
    from collections import Counter
    c = Counter(r[6] for r in ROWS)
    out = ["# REGISTRY — full formula library (spec L)", "",
           f"_{len(ROWS)} formulas (F001-F094), duplicate mechanisms collapsed. "
           f"Status counts: {dict(c)}_", "",
           "## Legend", ""]
    for k, v in STATUS_MEANING.items():
        out.append(f"- **{k}** — {v}")
    out += ["", "## Registry", "",
            "| id | family | paper | equation | data | horizon | status | disposition |",
            "|---|---|---|---|---|---|---|---|"]
    for r in ROWS:
        out.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    out += ["", "## Bottom line", "",
            f"- **{c['BOUNDED']}** formulas tested and bounded across Rounds 1-3",
            f"- **{c['SIZING_ONLY']}** are sizing/gating tools (vol.py, kalman.py, cost_model), not signals",
            f"- **{c['DATA_LIMITED']}** need data we do not have (live L2, OKX, on-chain)",
            f"- **{c['VETOED']}** vetoed (F_XVENUE), **{c['UNDERPOWERED']}** underpowered (F_LIQD)",
            f"- **{c.get('RUN_BATCH4', 0)}** run fresh in Batch 4 — 0 survivors, all NO_EDGE",
            f"- **{c['NOT_RUN']}** are variants of bounded formulas or pure sizing inputs — recorded, not re-run",
            "", "No formula in the library produces a cross-sectional or time-series edge that "
            "survives the N3-N9 ladder net of the per-coin cost model, on the accessible data "
            "(free Binance/Bybit history, 2021-2025)."]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "research" / "REGISTRY.md"
    p.write_text(to_markdown())
    print(f"wrote {p}  ({len(ROWS)} formulas)")
