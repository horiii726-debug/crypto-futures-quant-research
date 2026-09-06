"""Panel container + causal cross-sectional feature library.

A Panel wraps aligned (T x N) frames: close, quote_volume, taker_buy_base,
volume, high, low, and (optional) funding on the bar grid. Every feature
returns a (T x N) array of per-asset scores computed using information up to
and including bar t only. Cross-sectional standardisation at each t is
contemporaneous (across assets) and therefore causal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Panel:
    close: pd.DataFrame
    quote_volume: pd.DataFrame | None = None
    taker_buy_base: pd.DataFrame | None = None
    volume: pd.DataFrame | None = None
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    funding: pd.DataFrame | None = None      # aligned to bar grid, ffilled within interval
    bar_hours: float = 1.0

    @property
    def index(self):
        return self.close.index

    @property
    def symbols(self):
        return list(self.close.columns)

    def logret(self) -> pd.DataFrame:
        return np.log(self.close).diff()

    def fwd_ret(self, k: int = 1) -> pd.DataFrame:
        """Simple return realised over the k bars AFTER the current close -
        i.e. what a position opened now and held k bars earns. Used as the
        label; never as a feature."""
        return self.close.shift(-k) / self.close - 1.0

    def mask_valid(self, min_names: int = 5) -> pd.Series:
        return self.close.notna().sum(axis=1) >= min_names

    def sub(self, idx) -> "Panel":
        f = lambda d: (d.loc[idx] if d is not None else None)  # noqa
        return Panel(self.close.loc[idx], f(self.quote_volume), f(self.taker_buy_base),
                     f(self.volume), f(self.high), f(self.low), f(self.funding),
                     self.bar_hours)


# ---- helpers --------------------------------------------------------------

def zscore_x(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score at each timestamp (row-wise)."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def rank_x(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank in (0,1], row-wise, NaNs preserved."""
    return df.rank(axis=1, pct=True)


def _roll_sum(df, w):
    return df.rolling(w, min_periods=max(2, w // 2)).sum()


def _roll_std(df, w):
    return df.rolling(w, min_periods=max(3, w // 2)).std()


# ---- feature library: name -> callable(Panel, **params) -> (T x N) score ---

def f_xsec_momentum(p: Panel, lookback: int = 168, skip: int = 1) -> pd.DataFrame:
    lp = np.log(p.close)
    mom = lp.shift(skip) - lp.shift(skip + lookback)
    return zscore_x(mom)


def f_xsec_st_reversal(p: Panel, lookback: int = 1) -> pd.DataFrame:
    r = p.close.pct_change(lookback, fill_method=None)   # no pad -> delisted stays NaN
    return -zscore_x(r)


def f_xsec_low_vol(p: Panel, lookback: int = 168) -> pd.DataFrame:
    vol = _roll_std(p.logret(), lookback)
    return -zscore_x(vol)


def f_xsec_idio_vol(p: Panel, lookback: int = 336) -> pd.DataFrame:
    r = p.logret()
    mkt = r.mean(axis=1)
    # rolling beta then residual std, per asset (vectorised over assets)
    cov = r.mul(mkt, axis=0).rolling(lookback, min_periods=lookback // 2).mean() \
        .sub(r.rolling(lookback, min_periods=lookback // 2).mean().mul(
            mkt.rolling(lookback, min_periods=lookback // 2).mean(), axis=0))
    var_m = mkt.rolling(lookback, min_periods=lookback // 2).var()
    beta = cov.div(var_m, axis=0)
    resid = r.sub(beta.mul(mkt, axis=0))
    return -zscore_x(_roll_std(resid, lookback))


def f_xsec_amihud(p: Panel, lookback: int = 168) -> pd.DataFrame:
    illiq = (p.logret().abs() / p.quote_volume.replace(0, np.nan)) \
        .rolling(lookback, min_periods=lookback // 2).mean()
    return zscore_x(np.log(illiq))


def f_xsec_max(p: Panel, lookback: int = 504) -> pd.DataFrame:
    mx = p.logret().rolling(lookback, min_periods=lookback // 2).max()
    return -zscore_x(mx)


def f_xsec_skew(p: Panel, lookback: int = 504) -> pd.DataFrame:
    sk = p.logret().rolling(lookback, min_periods=lookback // 2).skew()
    return -zscore_x(sk)


def f_xsec_52w_high(p: Panel, lookback: int = 2160) -> pd.DataFrame:
    hi = p.close.rolling(lookback, min_periods=lookback // 3).max()
    return zscore_x(p.close / hi)


def f_xsec_volume_trend(p: Panel, fast: int = 168, slow: int = 720) -> pd.DataFrame:
    qv = p.quote_volume
    ratio = _roll_sum(qv, fast) / _roll_sum(qv, slow).replace(0, np.nan) * (slow / fast)
    return zscore_x(np.log(ratio))


def f_xsec_turnover_reversal(p: Panel, lookback: int = 168) -> pd.DataFrame:
    return -zscore_x(_roll_sum(p.quote_volume, lookback))


def f_xsec_residual_momentum(p: Panel, lookback: int = 336, skip: int = 1) -> pd.DataFrame:
    r = p.logret()
    mkt = r.mean(axis=1)
    var_m = mkt.rolling(lookback, min_periods=lookback // 2).var()
    cov = r.mul(mkt, axis=0).rolling(lookback, min_periods=lookback // 2).mean() \
        .sub(r.rolling(lookback, min_periods=lookback // 2).mean().mul(
            mkt.rolling(lookback, min_periods=lookback // 2).mean(), axis=0))
    beta = cov.div(var_m, axis=0)
    resid = r.sub(beta.mul(mkt, axis=0))
    rm = resid.shift(skip).rolling(lookback, min_periods=lookback // 2).mean() \
        / _roll_std(resid.shift(skip), lookback)
    return zscore_x(rm)


# ---- F_FUND -------------------------------------------------------------

def f_fund_carry(p: Panel, **_) -> pd.DataFrame:
    return -zscore_x(p.funding)


def f_fund_momentum(p: Panel, lookback: int = 21) -> pd.DataFrame:
    return -zscore_x(p.funding.rolling(lookback, min_periods=lookback // 2).mean())


def f_fund_zscore(p: Panel, lookback: int = 720) -> pd.DataFrame:
    mu = p.funding.rolling(lookback, min_periods=lookback // 3).mean()
    sd = p.funding.rolling(lookback, min_periods=lookback // 3).std().replace(0, np.nan)
    z = (p.funding - mu) / sd
    return -z  # extreme positive funding z -> short (fade)


def f_fund_accel(p: Panel, lookback: int = 3) -> pd.DataFrame:
    return -zscore_x(p.funding.diff(lookback))


# ---- F_FLOW (kline aggressor proxy) -----------------------------------------

def _buy_ratio(p: Panel) -> pd.DataFrame:
    return (p.taker_buy_base / p.volume.replace(0, np.nan)).clip(0, 1)


def f_flow_imb_momentum(p: Panel, lookback: int = 24) -> pd.DataFrame:
    imb = (_buy_ratio(p) - 0.5).rolling(lookback, min_periods=lookback // 2).mean()
    return zscore_x(imb)


def f_flow_imb_reversal(p: Panel, lookback: int = 4) -> pd.DataFrame:
    imb = (_buy_ratio(p) - 0.5).rolling(lookback, min_periods=1).mean()
    return -zscore_x(imb)


def f_flow_signed_vol(p: Panel, lookback: int = 24) -> pd.DataFrame:
    signed = (2 * _buy_ratio(p) - 1) * p.quote_volume
    return zscore_x(_roll_sum(signed, lookback))


def f_flow_kyle_lambda(p: Panel, lookback: int = 168) -> pd.DataFrame:
    dp = p.logret()
    sv = (2 * _buy_ratio(p) - 1) * np.sqrt(p.quote_volume.clip(lower=0))
    cov = (dp * sv).rolling(lookback, min_periods=lookback // 2).mean()
    var = (sv * sv).rolling(lookback, min_periods=lookback // 2).mean().replace(0, np.nan)
    lam = cov / var
    return -zscore_x(lam)   # high impact/illiquidity -> avoid (or fade)


# ---- F_XCOIN ----------------------------------------------------------------

def f_xcoin_btc_leadlag(p: Panel, lead: int = 1, ref: str = "BTCUSDT") -> pd.DataFrame:
    if ref not in p.close.columns:
        ref = p.close.columns[int(np.argmax(p.quote_volume.sum().values))]
    r = p.logret()
    beta_ref = r.rolling(336, min_periods=100).cov(r[ref]).div(
        r[ref].rolling(336, min_periods=100).var(), axis=0)
    pred = beta_ref.mul(r[ref].shift(lead - 1), axis=0)
    return zscore_x(pred)


def f_xcoin_peer_return(p: Panel, lookback: int = 6) -> pd.DataFrame:
    r = p.logret()
    peer = r.rolling(lookback, min_periods=1).mean()
    peer_mean = peer.mean(axis=1)
    return zscore_x(peer.sub(peer_mean, axis=0))  # relative peer strength


def f_xcoin_spillover_momentum(p: Panel, lookback: int = 168, corr_win: int = 336) -> pd.DataFrame:
    r = p.logret()
    mom = (np.log(p.close).shift(1) - np.log(p.close).shift(1 + lookback))
    # weight each coin's momentum by trailing correlation to the others (avg)
    avg_corr = r.rolling(corr_win, min_periods=corr_win // 2).corr(r.mean(axis=1))
    spill = (mom * avg_corr).mean(axis=1)
    return zscore_x(mom.sub(spill, axis=0))


def f_xcoin_pca_residual(p: Panel, lookback: int = 336, n_pc: int = 3) -> pd.DataFrame:
    lr = p.logret()
    r = lr.fillna(0.0)
    alive = lr.notna()                       # coin tradable at bar t
    out = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
    step = max(1, lookback // 8)
    for i in range(lookback, len(r), step):
        w = r.iloc[i - lookback:i]
        X = w.values - w.values.mean(0)
        try:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            recon = U[:, :n_pc] @ np.diag(S[:n_pc]) @ Vt[:n_pc]
            resid_last = (X[-1] - recon[-1])
        except np.linalg.LinAlgError:
            resid_last = np.zeros(X.shape[1])
        out.iloc[i] = -resid_last
    out = out.ffill(limit=step)
    return zscore_x(out.where(alive))        # drop delisted coins (R10 safe)


def f_xcoin_dispersion_switch(p: Panel, lookback: int = 168, mom_lb: int = 168) -> pd.DataFrame:
    r = p.logret()
    disp = r.std(axis=1).rolling(lookback, min_periods=lookback // 2).mean()
    disp_hi = disp > disp.rolling(720, min_periods=200).median()
    mom = zscore_x(np.log(p.close).shift(1) - np.log(p.close).shift(1 + mom_lb))
    rev = -mom
    return mom.where(disp_hi, rev)  # momentum in high dispersion, reversal in low


# =========================================================================
# P2 expansion: additional divisions (cross-sectional, causal, from OHLCV+fund)
# =========================================================================

def _wilder_rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def f_dir_rsi(p: Panel, n: int = 14) -> pd.DataFrame:
    return -zscore_x(_wilder_rsi(p.close, n) - 50)          # fade RSI extremes


def f_dir_macd(p: Panel, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.DataFrame:
    lp = np.log(p.close)
    macd = lp.ewm(span=fast, adjust=False).mean() - lp.ewm(span=slow, adjust=False).mean()
    hist = macd - macd.ewm(span=sig, adjust=False).mean()
    return zscore_x(hist)


def f_dir_ma_cross(p: Panel, fast: int = 10, slow: int = 50) -> pd.DataFrame:
    return zscore_x(p.close.rolling(fast).mean() / p.close.rolling(slow).mean() - 1.0)


def f_dir_bollinger_z(p: Panel, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    m = p.close.rolling(n).mean()
    s = p.close.rolling(n).std().replace(0, np.nan)
    return -zscore_x((p.close - m) / (k * s))               # mean reversion


def f_dir_trend_r2(p: Panel, n: int = 30) -> pd.DataFrame:
    lp = np.log(p.close)
    x = np.arange(n)
    xd = x - x.mean()
    def r2(col):
        return col.rolling(n).apply(
            lambda y: (np.dot(xd, y - y.mean()) ** 2) /
                      ((xd @ xd) * np.sum((y - y.mean()) ** 2) + 1e-18) *
                      np.sign(np.dot(xd, y - y.mean())), raw=True)
    return zscore_x(lp.apply(r2))


def f_dir_donchian_pos(p: Panel, n: int = 20) -> pd.DataFrame:
    hi = p.high.rolling(n).max()
    lo = p.low.rolling(n).min()
    return zscore_x((p.close - lo) / (hi - lo).replace(0, np.nan))


def f_transform_hurst(p: Panel, n: int = 60) -> pd.DataFrame:
    lr = p.logret()
    def hurst(y):
        y = y[np.isfinite(y)]
        if len(y) < 20:
            return np.nan
        z = np.cumsum(y - y.mean())
        R = z.max() - z.min()
        S = y.std()
        return np.log(R / S + 1e-12) / np.log(len(y)) if S > 0 else np.nan
    h = lr.rolling(n).apply(hurst, raw=True)
    return zscore_x(h - 0.5)          # >0 trending, <0 mean-reverting


def f_transform_spectral_entropy(p: Panel, n: int = 64) -> pd.DataFrame:
    lr = p.logret().fillna(0.0)
    def se(y):
        f = np.abs(np.fft.rfft(y - y.mean())) ** 2
        s = f.sum()
        if s <= 0:
            return np.nan
        pr = f / s
        return -np.sum(pr * np.log(pr + 1e-12))
    return -zscore_x(lr.rolling(n).apply(se, raw=True))     # low entropy = structure


def f_dep_variance_ratio(p: Panel, q: int = 5, n: int = 60) -> pd.DataFrame:
    lr = p.logret()
    var1 = lr.rolling(n).var()
    varq = lr.rolling(n).sum().rolling(q).apply(lambda x: np.var(x), raw=True) if False else \
        (lr.rolling(q).sum()).rolling(n).var() / q
    vr = varq / var1.replace(0, np.nan)
    return zscore_x(vr - 1.0)


def f_dep_autocorr1(p: Panel, n: int = 45) -> pd.DataFrame:
    lr = p.logret()
    ac = lr.rolling(n).apply(lambda y: np.corrcoef(y[:-1], y[1:])[0, 1]
                             if len(y) > 5 and np.std(y) > 0 else np.nan, raw=True)
    return zscore_x(ac)


def f_path_run_length(p: Panel, cap: int = 10) -> pd.DataFrame:
    s = np.sign(p.close.diff())
    def runlen(col):
        out = np.zeros(len(col))
        c = 0
        for i in range(1, len(col)):
            if col[i] == col[i - 1] and col[i] != 0:
                c += 1
            else:
                c = 0
            out[i] = c * (col[i] if col[i] != 0 else 0)
        return np.clip(out, -cap, cap)
    rl = s.apply(lambda c: pd.Series(runlen(c.values), index=c.index))
    return -zscore_x(rl)             # long runs revert


def f_path_mfe_mae(p: Panel, n: int = 20) -> pd.DataFrame:
    hi = p.high.rolling(n).max()
    lo = p.low.rolling(n).min()
    mfe = hi / p.close.shift(n) - 1.0
    mae = 1.0 - lo / p.close.shift(n)
    return zscore_x(mfe / (mae + 1e-9))


def f_multi_tf_agreement(p: Panel, tfs=(3, 7, 21, 45)) -> pd.DataFrame:
    lp = np.log(p.close)
    agree = sum(np.sign(lp - lp.shift(t)) for t in tfs)
    return zscore_x(agree)


def f_multi_fast_slow(p: Panel, fast: int = 7, slow: int = 45) -> pd.DataFrame:
    lp = np.log(p.close)
    return zscore_x((lp - lp.shift(fast)) - (lp - lp.shift(slow)))


def f_vol_yang_zhang(p: Panel, n: int = 20) -> pd.DataFrame:
    """No `open` panel available -> Parkinson high-low estimator blended with
    close-to-close (a Garman-Klass variant without the open term)."""
    h, l, c = np.log(p.high), np.log(p.low), np.log(p.close)
    park = (1.0 / (4.0 * np.log(2.0))) * (h - l) ** 2
    cc = (c - c.shift(1)) ** 2
    gk = 0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * cc
    est = (0.5 * park + 0.5 * gk).clip(lower=0).rolling(n, min_periods=n // 2).mean()
    return -zscore_x(np.sqrt(est))                   # low-vol tilt


def f_vol_semivar_skew(p: Panel, n: int = 20) -> pd.DataFrame:
    r = p.logret()
    rsp = (r.clip(lower=0) ** 2).rolling(n).sum()
    rsm = (r.clip(upper=0) ** 2).rolling(n).sum()
    return -zscore_x((rsp - rsm) / (rsp + rsm + 1e-12))   # signed-jump tilt


def f_liq_kyle_bar(p: Panel, n: int = 30) -> pd.DataFrame:
    return f_flow_kyle_lambda(p, n)                  # alias, kept in F_LIQ namespace


def f_state_vol_regime_mom(p: Panel, vol_n: int = 20, mom_n: int = 20) -> pd.DataFrame:
    """regime overlay expressed as a tradeable XS signal: in low-vol regime use
    momentum, in high-vol regime use short reversal."""
    r = p.logret()
    rv = r.rolling(vol_n).std()
    hi = rv > rv.rolling(120, min_periods=40).median()
    lp = np.log(p.close)
    mom = zscore_x(lp.shift(1) - lp.shift(1 + mom_n))
    return (-mom).where(hi, mom)


REGISTRY_P2 = {
    "dir_rsi": (f_dir_rsi, "F_DIR"), "dir_macd": (f_dir_macd, "F_DIR"),
    "dir_ma_cross": (f_dir_ma_cross, "F_DIR"), "dir_bollinger_z": (f_dir_bollinger_z, "F_DIR"),
    "dir_donchian_pos": (f_dir_donchian_pos, "F_DIR"),
    "transform_hurst": (f_transform_hurst, "F_TRANSFORM"),
    "transform_spectral_entropy": (f_transform_spectral_entropy, "F_TRANSFORM"),
    "dep_variance_ratio": (f_dep_variance_ratio, "F_DEP"),
    "dep_autocorr1": (f_dep_autocorr1, "F_DEP"),
    "path_run_length": (f_path_run_length, "F_PATH"),
    "path_mfe_mae": (f_path_mfe_mae, "F_PATH"),
    "multi_tf_agreement": (f_multi_tf_agreement, "F_MULTI"),
    "multi_fast_slow": (f_multi_fast_slow, "F_MULTI"),
    "vol_yang_zhang": (f_vol_yang_zhang, "F_VOL"),
    "vol_semivar_skew": (f_vol_semivar_skew, "F_VOL"),
    "liq_kyle_bar": (f_liq_kyle_bar, "F_LIQ"),
    "state_vol_regime_mom": (f_state_vol_regime_mom, "F_STATE"),
}


REGISTRY = {
    # F_XSEC
    "xsec_momentum": (f_xsec_momentum, "F_XSEC"),
    "xsec_st_reversal": (f_xsec_st_reversal, "F_XSEC"),
    "xsec_low_vol": (f_xsec_low_vol, "F_XSEC"),
    "xsec_idio_vol": (f_xsec_idio_vol, "F_XSEC"),
    "xsec_amihud": (f_xsec_amihud, "F_XSEC"),
    "xsec_max": (f_xsec_max, "F_XSEC"),
    "xsec_skew": (f_xsec_skew, "F_XSEC"),
    "xsec_52w_high": (f_xsec_52w_high, "F_XSEC"),
    "xsec_volume_trend": (f_xsec_volume_trend, "F_XSEC"),
    "xsec_turnover_reversal": (f_xsec_turnover_reversal, "F_XSEC"),
    "xsec_residual_momentum": (f_xsec_residual_momentum, "F_XSEC"),
    # F_FUND
    "fund_carry": (f_fund_carry, "F_FUND"),
    "fund_momentum": (f_fund_momentum, "F_FUND"),
    "fund_zscore": (f_fund_zscore, "F_FUND"),
    "fund_accel": (f_fund_accel, "F_FUND"),
    # F_FLOW
    "flow_imb_momentum": (f_flow_imb_momentum, "F_FLOW"),
    "flow_imb_reversal": (f_flow_imb_reversal, "F_FLOW"),
    "flow_signed_vol": (f_flow_signed_vol, "F_FLOW"),
    "flow_kyle_lambda": (f_flow_kyle_lambda, "F_FLOW"),
    # F_XCOIN
    "xcoin_btc_leadlag": (f_xcoin_btc_leadlag, "F_XCOIN"),
    "xcoin_peer_return": (f_xcoin_peer_return, "F_XCOIN"),
    "xcoin_spillover_momentum": (f_xcoin_spillover_momentum, "F_XCOIN"),
    "xcoin_pca_residual": (f_xcoin_pca_residual, "F_XCOIN"),
    "xcoin_dispersion_switch": (f_xcoin_dispersion_switch, "F_XCOIN"),
}

REGISTRY.update(REGISTRY_P2)
