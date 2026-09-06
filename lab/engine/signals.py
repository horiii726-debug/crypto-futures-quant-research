"""Spec -> feature resolver with a module whitelist.

A hypothesis spec references features by name only. This module maps names to
callables drawn from a WHITELIST of vetted feature modules. research-coder may
add modules to the whitelist; it may never inline a parameter chosen from
results (R4 / research-coder charter).

Every feature callable has signature:  f(ctx: FeatureContext) -> np.ndarray
and must be causal (leak_canary.assert_causal is run in CI).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lab.engine import kernels

WHITELIST = {
    "lab.features.price",     # returns, momentum, vol
    "lab.features.xsec",      # cross-sectional rank / dispersion
    "lab.features.flow",      # order-flow imbalance (needs aggTrades)
    "lab.features.funding",   # funding premium / basis
}


@dataclass
class FeatureContext:
    close: np.ndarray            # (T,) or (T, N)
    high: np.ndarray = None
    low: np.ndarray = None
    volume: np.ndarray = None
    funding: np.ndarray = None
    meta: dict = None


# ---- built-in causal primitives (used by the calibration pipeline) --------

def feat_momentum(ctx: FeatureContext, lookback: int = 20) -> np.ndarray:
    c = np.asarray(ctx.close, float)
    logp = np.log(np.maximum(c, 1e-12))
    if logp.ndim == 1:
        out = np.full_like(logp, np.nan)
        out[lookback:] = logp[lookback:] - logp[:-lookback]
        return out
    out = np.full_like(logp, np.nan)
    out[lookback:] = logp[lookback:] - logp[:-lookback]
    return out


def feat_xsec_rank(ctx: FeatureContext, lookback: int = 20) -> np.ndarray:
    """Cross-sectional rank of trailing momentum. (T, N) -> (T, N) in (0,1]."""
    mom = feat_momentum(ctx, lookback)
    return kernels.rank_lastaxis(mom)


def feat_vol(ctx: FeatureContext, span: int = 50) -> np.ndarray:
    c = np.asarray(ctx.close, float)
    r = np.zeros_like(c)
    r[1:] = c[1:] / c[:-1] - 1.0
    return np.sqrt(np.maximum(kernels.ewma(r ** 2, 2.0 / (span + 1.0)), 0.0))


_BUILTIN = {
    "momentum": feat_momentum,
    "xsec_rank": feat_xsec_rank,
    "vol": feat_vol,
}


def resolve(name: str):
    if name in _BUILTIN:
        return _BUILTIN[name]
    if "." in name:
        mod, attr = name.rsplit(".", 1)
        if mod not in WHITELIST:
            raise PermissionError(f"module {mod} not on the feature whitelist")
        import importlib
        return getattr(importlib.import_module(mod), attr)
    raise KeyError(f"unknown feature '{name}' (not builtin, not dotted-path)")


def build_signal(spec: dict, ctx: FeatureContext) -> np.ndarray:
    """spec = {"feature": "xsec_rank", "params": {...}, "map": "long_top_short_bottom"}"""
    fn = resolve(spec["feature"])
    raw = fn(ctx, **spec.get("params", {}))
    mapping = spec.get("map", "zscore")
    return _to_position(raw, mapping, spec.get("map_params", {}))


def _to_position(raw: np.ndarray, mapping: str, p: dict) -> np.ndarray:
    raw = np.asarray(raw, float)
    if mapping == "sign":
        return np.nan_to_num(np.sign(raw))
    if mapping == "zscore":
        mu = np.nanmean(raw, axis=0, keepdims=True)
        sd = np.nanstd(raw, axis=0, keepdims=True)
        z = (raw - mu) / np.where(sd > 0, sd, np.nan)
        return np.clip(np.nan_to_num(z), -1, 1)
    if mapping == "long_top_short_bottom":
        q = p.get("q", 0.3)
        pos = np.zeros_like(raw)
        pos[raw >= (1 - q)] = 1.0
        pos[raw <= q] = -1.0
        pos[np.isnan(raw)] = 0.0
        return pos
    raise KeyError(mapping)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.01, (500, 6)), axis=0)
    ctx = FeatureContext(close=px)
    sig = build_signal({"feature": "xsec_rank", "params": {"lookback": 20},
                        "map": "long_top_short_bottom", "map_params": {"q": 0.33}}, ctx)
    print("signal shape", sig.shape, "gross", np.abs(sig).sum(axis=1)[:5])
