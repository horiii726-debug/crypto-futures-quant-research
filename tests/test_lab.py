#!/usr/bin/env python3
"""S0 mandatory test suite. All 10 must pass.

Run: python3 tests/test_lab.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# isolated ledger - the S0 suite must not touch the production research ledger (R3)
os.environ["CRYPTO_LAB_LEDGER_SQLITE"] = str(ROOT / "ledger" / "_selftest.sqlite")

from lab.engine import kernels                              # noqa: E402
from lab.engine.backtest import (forward_returns, run_backtest,  # noqa: E402
                                 simple_returns, cost_sweep)
from lab.engine.labels import (average_uniqueness, triple_barrier,  # noqa: E402
                               barrier_payoff_distribution)
from lab.engine.signals import FeatureContext, feat_momentum, feat_vol  # noqa: E402
from lab.guards.leak_canary import assert_causal, future_perturbation  # noqa: E402
from lab.ledger import Ledger                               # noqa: E402
from lab.make_synth import generate                         # noqa: E402
from lab.stats.mcs import model_confidence_set              # noqa: E402
from lab.stats.surrogate import surrogate_threshold         # noqa: E402

CALIB_COSTS = {"taker_fee": 0.0004, "maker_fee": 0.0002, "slippage_bps": 1.0}
TOL = 1e-10

_results = []


def _run(name, fn):
    try:
        detail = fn()
        _results.append((name, True, detail or ""))
        print(f"  PASS  {name}   {detail or ''}")
    except Exception as e:
        _results.append((name, False, repr(e)))
        print(f"  FAIL  {name}   {e}")
        traceback.print_exc()


# --------------------------------------------------------------------------
# 1. causality of each feature (future perturbation, dev < 1e-10)
# --------------------------------------------------------------------------

def test_1_feature_causality():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 0.01, 4000)) + 5.0

    def f_mom(a):
        return feat_momentum(FeatureContext(close=a), lookback=24)

    def f_vol(a):
        return feat_vol(FeatureContext(close=a), span=50)

    devs = {}
    for nm, f in (("momentum", f_mom), ("vol", f_vol)):
        ok, info = assert_causal(f, x, tol=TOL)
        devs[nm] = max(info["future_perturbation"]["max_dev"],
                       info["prefix_stability"]["max_dev"])
        assert ok, f"{nm} not causal: {info}"

    # xsec_rank on a (T,N) panel: perturb future rows, check current rank stable
    panel = np.cumsum(rng.normal(0, 0.01, (3000, 8)), axis=0) + 5.0
    from lab.engine.signals import feat_xsec_rank
    base = feat_xsec_rank(FeatureContext(close=panel), lookback=20)
    pert = panel.copy()
    pert[1500:] += rng.normal(0, 0.5, pert[1500:].shape)
    p2 = feat_xsec_rank(FeatureContext(close=pert), lookback=20)
    d = np.nanmax(np.abs(np.nan_to_num(base[:1500]) - np.nan_to_num(p2[:1500])))
    devs["xsec_rank"] = float(d)
    assert d <= TOL, f"xsec_rank leaked future info: dev={d}"
    return f"max dev {max(devs.values()):.1e} ({devs})"


# --------------------------------------------------------------------------
# 2. off-by-one, two-sided: oracle t+1 NOT profit, oracle t+2 profit
# --------------------------------------------------------------------------

def test_2_off_by_one():
    rng = np.random.default_rng(1)
    close = (100 * np.cumprod(1 + rng.normal(0, 0.01, (6000, 1)))).reshape(-1, 1)
    ret = simple_returns(close)[:, 0]           # ret[k] = P[k]/P[k-1]-1

    def oracle(h):
        s = np.zeros((len(close), 1))
        s[: len(close) - h, 0] = np.sign(ret[h:])   # s[t] knows sign(ret[t+h])
        return s

    r1 = run_backtest(close, oracle(1), CALIB_COSTS, execution_lag=1)
    r2 = run_backtest(close, oracle(2), CALIB_COSTS, execution_lag=1)
    # engine: pnl[k] = signal[k-2]*ret[k].
    #   oracle(1): signal[k-2]=sign(ret[k-1]) -> misaligned -> ~0 gross
    #   oracle(2): signal[k-2]=sign(ret[k])   -> aligned    -> sum|ret| >> 0
    assert r1["total_gross"] < 0.05 * r2["total_gross"], \
        f"oracle t+1 should NOT profit: gross1={r1['total_gross']:.3f} gross2={r2['total_gross']:.3f}"
    assert r1["total_net"] <= 0.0 + 1e-9, f"oracle t+1 net positive: {r1['total_net']:.3f}"
    assert r2["total_net"] > 0.5, f"oracle t+2 should profit strongly: {r2['total_net']:.3f}"
    return f"gross t+1={r1['total_gross']:.3f}  gross t+2={r2['total_gross']:.1f}  net t+2={r2['total_net']:.1f}"


# --------------------------------------------------------------------------
# 3. cost directional: cost up -> net down, gross unchanged
# --------------------------------------------------------------------------

def test_3_cost_directional():
    rng = np.random.default_rng(2)
    T, N = 3000, 6
    px = 100 * np.cumprod(1 + rng.normal(0, 0.01, (T, N)), axis=0)
    sig = np.sign(rng.normal(size=(T, N)))
    sweep = cost_sweep(px, sig, CALIB_COSTS, multipliers=(0.5, 1, 2, 4, 8))
    g0 = sweep[0]["gross"]
    assert all(abs(s["gross"] - g0) < 1e-9 for s in sweep), f"gross moved: {sweep}"
    nets = [s["net"] for s in sweep]
    assert all(nets[i] > nets[i + 1] for i in range(len(nets) - 1)), \
        f"net not monotonically decreasing in cost: {nets}"
    return f"gross const={g0:.4f}  net {nets[0]:.3f} -> {nets[-1]:.3f}"


# --------------------------------------------------------------------------
# 4. barrier geometry changes the payoff distribution
# --------------------------------------------------------------------------

def test_4_barrier_geometry():
    rng = np.random.default_rng(3)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 5000)))
    ev = np.arange(120, 4800, 4)
    tight = barrier_payoff_distribution(
        triple_barrier(px, ev, pt_mult=0.5, sl_mult=0.5, max_hold=30))
    wide = barrier_payoff_distribution(
        triple_barrier(px, ev, pt_mult=5.0, sl_mult=5.0, max_hold=30))
    # tight barriers -> few verticals, small |payoff|; wide -> many verticals
    assert tight["p_vertical"] < wide["p_vertical"] - 0.1, \
        f"vertical share not changed: tight={tight['p_vertical']:.2f} wide={wide['p_vertical']:.2f}"
    assert wide["std"] > tight["std"] * 1.3, \
        f"payoff dispersion not changed: tight std={tight['std']:.4f} wide std={wide['std']:.4f}"
    return (f"p_vertical {tight['p_vertical']:.2f}->{wide['p_vertical']:.2f}  "
            f"std {tight['std']:.4f}->{wide['std']:.4f}")


# --------------------------------------------------------------------------
# 5. uniqueness detects overlap
# --------------------------------------------------------------------------

def test_5_uniqueness_overlap():
    n_bars = 6000
    hold = 50
    # heavily overlapping: an event every bar, each held `hold` bars
    t0_o = np.arange(0, n_bars - hold, 1)
    t1_o = t0_o + hold
    au_o = average_uniqueness(t0_o, t1_o, n_bars)
    # non-overlapping: events spaced wider than the holding window
    t0_n = np.arange(0, n_bars - hold, hold + 5)
    t1_n = t0_n + hold
    au_n = average_uniqueness(t0_n, t1_n, n_bars)
    assert au_o["avg_uniqueness"] < 0.10, f"overlap not detected: {au_o}"
    assert au_n["avg_uniqueness"] > 0.90, f"non-overlap misread: {au_n}"
    assert au_o["overlap_factor"] > 5 * au_n["overlap_factor"]
    return (f"overlap u={au_o['avg_uniqueness']:.3f} (factor {au_o['overlap_factor']:.1f})  "
            f"disjoint u={au_n['avg_uniqueness']:.3f}")


# --------------------------------------------------------------------------
# 6. ledger rejects UPDATE and DELETE
# --------------------------------------------------------------------------

def test_6_ledger_append_only():
    L = Ledger()
    sid = L.add_source("arxiv", "2401.00001", resolved=True, retrieved=True,
                       title="probe row")
    blocked = {"update": False, "delete": False}
    try:
        L.conn.execute("UPDATE sources SET title='hacked' WHERE id=?", (sid,))
        L.conn.commit()
    except sqlite3.Error as e:
        blocked["update"] = "append-only" in str(e).lower() or "rejected" in str(e).lower()
    try:
        L.conn.execute("DELETE FROM sources WHERE id=?", (sid,))
        L.conn.commit()
    except sqlite3.Error as e:
        blocked["delete"] = "append-only" in str(e).lower() or "rejected" in str(e).lower()
    L.conn.rollback()
    # confirm the row is untouched
    cur = L.conn.execute("SELECT title FROM sources WHERE id=?", (sid,))
    title = cur.fetchone()[0]
    assert blocked["update"], "UPDATE was not blocked by a trigger"
    assert blocked["delete"], "DELETE was not blocked by a trigger"
    assert title == "probe row", f"row mutated despite triggers: {title!r}"
    # every result table must carry both triggers
    cur = L.conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
    trg = {r[0] for r in cur.fetchall()}
    from lab.ledger import RESULT_TABLES
    for t in RESULT_TABLES:
        assert f"no_update_{t}" in trg and f"no_delete_{t}" in trg, f"missing triggers on {t}"
    return f"UPDATE & DELETE blocked on all {len(RESULT_TABLES)} result tables"


# --------------------------------------------------------------------------
# 7 + 8. calibration: zero-edge control loses ~cost; planted edge is detected
# --------------------------------------------------------------------------

_calib_cache = {}


def _calib(edge, seed, family):
    key = (edge, seed, family)
    if key not in _calib_cache:
        from tests.calibration import classify, evaluate_dataset
        import yaml
        gates = yaml.safe_load((ROOT / "config" / "gates.yaml").read_text())
        L = Ledger()
        res = evaluate_dataset(edge, seed, family, L, n_bars=9000, n_coins=10,
                               n_surr=120)
        _calib_cache[key] = classify(res, gates)
    return _calib_cache[key]


def test_7_zero_edge_control_loses_cost():
    r = _calib(0.0, 313, "F_TEST_CTRL")
    assert not r["is_candidate"], f"zero-edge control passed as candidate: {r}"
    assert r["sharpe_net"] < 0, f"control net sharpe not negative: {r['sharpe_net']}"
    # net loss should be on the order of the transaction cost, not a blow-up
    assert -4.0 * r["total_cost"] < r["total_net"] < 0, \
        f"control loss not ~cost: net={r['total_net']:.3f} cost={r['total_cost']:.3f}"
    assert not r["clears_noise"], "control cleared the surrogate noise ceiling"
    return f"net={r['total_net']:.3f}  cost={r['total_cost']:.3f}  dsr={r['dsr']:.2f} -> {r['verdict']}"


def test_8_planted_edge_detected():
    r = _calib(0.35, 414, "F_TEST_EDGE")
    assert r["is_candidate"], f"planted IC=0.35 edge NOT detected: {r}"
    assert r["clears_noise"] and r["dsr"] >= r["dsr_min"] and r["sharpe_net"] > 0
    return f"net_sharpe={r['sharpe_net']:.2f}  dsr={r['dsr']:.2f}  surr {r['surrogate_obs']:.3f}>{r['surrogate_q99']:.3f} -> {r['verdict']}"


# --------------------------------------------------------------------------
# 9. MCS selftest does not crown a single winner on random data
# --------------------------------------------------------------------------

def test_9_mcs_no_single_winner():
    rng = np.random.default_rng(42)
    T, K = 500, 200
    losses = rng.normal(0.0, 1.0, (T, K)) ** 2
    res = model_confidence_set(losses, alpha=0.10, n_boot=300, seed=1)
    assert not res["singleton"], "MCS collapsed to one winner on exchangeable noise"
    assert res["mcs_size"] >= K // 4, f"MCS kept too few: {res['mcs_size']}/{K}"
    return f"kept {res['mcs_size']}/{K} models (singleton={res['singleton']})"


# --------------------------------------------------------------------------
# 10. surrogate produces a sensible noise threshold
# --------------------------------------------------------------------------

def test_10_surrogate_threshold():
    rng = np.random.default_rng(0)
    n = 4000
    f = rng.normal(size=n)

    def ic(feat, lab):
        m = np.isfinite(feat) & np.isfinite(lab)
        if m.sum() < 5 or feat[m].std() == 0 or lab[m].std() == 0:
            return 0.0
        return abs(float(np.corrcoef(feat[m], lab[m])[0, 1]))

    noise = surrogate_threshold(ic, f, rng.normal(size=n), n_surr=250, block=24, seed=1)
    edge = surrogate_threshold(ic, f, 0.3 * f + rng.normal(size=n), n_surr=250,
                               block=24, seed=2)
    assert 0.0 < noise["q99"] < 0.15, f"noise ceiling implausible: {noise['q99']}"
    assert not noise["clears_noise"], "pure noise cleared its own ceiling"
    assert edge["clears_noise"] and edge["p_value"] < 0.05, f"real edge missed: {edge}"
    return (f"noise q99={noise['q99']:.3f} (obs {noise['observed']:.3f}), "
            f"edge obs={edge['observed']:.3f} p={edge['p_value']:.3f}")


TESTS = [
    ("1  feature causality (dev<1e-10)", test_1_feature_causality),
    ("2  off-by-one two-sided", test_2_off_by_one),
    ("3  cost directional", test_3_cost_directional),
    ("4  barrier geometry -> payoff", test_4_barrier_geometry),
    ("5  uniqueness detects overlap", test_5_uniqueness_overlap),
    ("6  ledger rejects UPDATE/DELETE", test_6_ledger_append_only),
    ("7  zero-edge control loses ~cost", test_7_zero_edge_control_loses_cost),
    ("8  planted edge detected", test_8_planted_edge_detected),
    ("9  MCS no single winner on noise", test_9_mcs_no_single_winner),
    ("10 surrogate noise threshold", test_10_surrogate_threshold),
]


def main() -> int:
    print(f"kernel backend: {kernels.backend()}  numba={kernels._HAVE_NUMBA}")
    ks = kernels.verify_kernels(verbose=False)
    print(f"verify_kernels: {ks['verdict']}")
    if not ks["all_identical"]:
        print("ABORT: kernels not bit-identical")
        return 1
    print("running 10 tests...\n")
    for name, fn in TESTS:
        _run(name, fn)
    n_pass = sum(1 for _, ok, _ in _results if ok)
    print(f"\n{'=' * 60}\n{n_pass}/{len(TESTS)} passed")
    if n_pass != len(TESTS):
        for nm, ok, d in _results:
            if not ok:
                print(f"  FAILED: {nm}  -> {d}")
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
