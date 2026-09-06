"""Calibrate the hybrid execution model per liquidity tier on the real
aggTrades tape, using real best-bid-qty from bookTicker as the queue size."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab.exec.maker_model import calibrate_symbol  # noqa

PROC = ROOT / "data" / "processed"

# real best-bid-qty notional from bookTicker (2023-11..2024-03), $; queue_ahead
# = queue_fraction * this (you join mid-queue).
BIDQTY = {"BTCUSDT": 112236, "SOLUSDT": 1830, "DARUSDT": 1248, "WLDUSDT": 229, "HOOKUSDT": 102}
QUEUE_FRACTION = 0.7

# real half-spread (bps) from bookTicker where available, else aggTrades
HALF_SPREAD = {"BTCUSDT": 0.012, "SOLUSDT": 0.051, "DARUSDT": 3.6, "WLDUSDT": 0.40, "HOOKUSDT": 0.97}

# aggTrades days available for these (cached)
DAYS = ["2023-10-18", "2024-02-21", "2024-06-19", "2024-10-16", "2025-01-15",
        "2025-04-16", "2025-07-16"]


def main():
    out = []
    for sym, bq in BIDQTY.items():
        r = calibrate_symbol(sym, DAYS, queue_ahead_notional=bq * QUEUE_FRACTION,
                             maker_fee=0.0002, taker_fee=0.0005,
                             half_spread_bps=HALF_SPREAD[sym],
                             entry_wait_sec=45.0, exit_wait_sec=10.0,
                             hold_lookahead_sec=120.0, seed=3)
        rt = r.get("roundtrip", {})
        print(f"{sym:9} p_fill entry={r['entry_sim'].get('p_fill'):.2f} exit={r['exit_sim'].get('p_fill'):.2f} "
              f"| adv_sel entry={r['entry_sim'].get('adverse_selection_bps'):.2f}bps "
              f"| hybrid_rt={rt.get('hybrid_roundtrip_bps'):.2f}bps "
              f"taker_rt={rt.get('taker_only_roundtrip_bps'):.2f}bps "
              f"saving={rt.get('saving_pct', 0)*100:.0f}%")
        out.append({"symbol": sym, **r})
    (PROC / "exec_calibration.json").write_text(json.dumps(out, indent=1, default=float))

    # universe hybrid round-trip: liquidity-weighted-ish (equal weight across
    # tiers is conservative - the illiquid legs bind a cross-sec book).
    rts = [o["roundtrip"]["hybrid_roundtrip_bps"] for o in out
           if o.get("roundtrip", {}).get("status") == "ok"]
    takers = [o["roundtrip"]["taker_only_roundtrip_bps"] for o in out
              if o.get("roundtrip", {}).get("status") == "ok"]
    summary = {
        "hybrid_roundtrip_bps_median": float(np.median(rts)),
        "hybrid_roundtrip_bps_mean": float(np.mean(rts)),
        "taker_roundtrip_bps_median": float(np.median(takers)),
        "by_symbol": {o["symbol"]: o["roundtrip"].get("hybrid_roundtrip_bps")
                      for o in out if o.get("roundtrip", {}).get("status") == "ok"},
    }
    (PROC / "exec_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("\n", json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
