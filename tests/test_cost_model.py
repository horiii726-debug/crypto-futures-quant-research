#!/usr/bin/env python3
"""P0.4 — mandatory unit tests for lab/exec/cost_model.py.

If the BTC reconciliation misses by >15%, STOP: there is a bug, do not re-score.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CRYPTO_LAB_LEDGER_SQLITE", str(ROOT / "ledger" / "_selftest.sqlite"))

from lab.exec import cost_model as cm  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def main() -> int:
    print("=== cost_model unit tests (P0.4) ===")

    # 1. liquidity ordering: cost(BTC) < cost(HOOK) < cost(DAR)
    b = cm.roundtrip_bps("BTCUSDT", 20_000, "hybrid")
    h = cm.roundtrip_bps("HOOKUSDT", 20_000, "hybrid")
    d = cm.roundtrip_bps("DARUSDT", 20_000, "hybrid")
    check("ordering  BTC < HOOK < DAR", b < h < d, f"{b:.2f} < {h:.2f} < {d:.2f}")

    # 2. cost monotone increasing in notional
    sizes = [1_000, 10_000, 100_000, 1_000_000]
    cs = [cm.cost_bps("SOLUSDT", "buy", n, None, "taker", leg="entry")["total_bps"] for n in sizes]
    check("monotone in notional (SOL, taker)", all(x <= y + 1e-9 for x, y in zip(cs, cs[1:])),
          " -> ".join(f"{x:.2f}" for x in cs))

    # 3. p_fill falls when volatility rises
    lo = cm.p_fill("BTCUSDT", leg="entry", notional=20_000, sigma_1h_ratio=1.0)
    hi = cm.p_fill("BTCUSDT", leg="entry", notional=20_000, sigma_1h_ratio=3.0)
    check("p_fill drops when vol rises", hi < lo, f"{hi:.3f} < {lo:.3f}")

    # 4. BTC hybrid roundtrip reconciles to exec_calibration (5.662 bps) within 15%
    rt25 = cm.roundtrip_bps("BTCUSDT", 25_000, "hybrid")
    target = 5.662335661165034
    err = abs(rt25 - target) / target
    check("BTC reconciliation within 15%", err <= 0.15,
          f"{rt25:.3f} vs {target:.3f}  (err {err:.1%})  band [4.81, 6.51]")

    # 5. impact -> 0 as Q -> 0
    i0 = cm.impact_bps("DARUSDT", 0.0)
    i_small = cm.impact_bps("DARUSDT", 1.0)
    check("impact -> 0 as Q -> 0", i0 == 0.0 and i_small < 0.1, f"i(0)={i0}, i(1)={i_small:.4f}")

    # 6. taker never cheaper than fee; hybrid never above taker for a liquid coin
    tk = cm.cost_bps("ETHUSDT", "buy", 20_000, None, "taker", leg="entry")["total_bps"]
    hy = cm.cost_bps("ETHUSDT", "buy", 20_000, None, "hybrid", leg="entry")["total_bps"]
    check("hybrid <= taker (ETH)", hy <= tk + 1e-9, f"hybrid {hy:.2f} <= taker {tk:.2f}")
    check("taker >= maker fee", tk >= cm.FEE_TAKER_BPS - 1e-9, f"{tk:.2f} >= {cm.FEE_TAKER_BPS}")

    # 7. every prio50 coin returns a finite cost (fallbacks work)
    import pandas as pd
    coins = pd.read_parquet(ROOT / "data" / "train" / "prio50_2y" / "1h" / "close.parquet").columns.tolist()
    s = cm.cost_oneway_bps_by_coin(coins, 20_000, "hybrid")
    check("all prio50 coins priced, finite, positive",
          bool(s.notna().all() and (s > 0).all()), f"{len(s)} coins, range [{s.min():.2f}, {s.max():.2f}]")

    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
