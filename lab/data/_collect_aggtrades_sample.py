"""Sampled aggTrades download for spread/slippage measurement only.
~20 days per symbol spread across the study window - NOT the full tape."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lab.data.binance_vision as bv  # noqa
from lab.data.spread import realised_spread_aggtrades  # noqa

SPEC = ROOT / "data" / "raw" / "_spec"
OUT = ROOT / "data" / "processed" / "spread_measured.json"

# 2 sample days per month, 2023-10 .. 2025-07 (avoid partial edge months)
rng = np.random.default_rng(7)
DAYS = []
for m in pd.period_range("2023-10", "2025-07", freq="M"):
    for d in sorted(rng.choice(range(3, 27), size=2, replace=False)):
        DAYS.append(f"{m}-{d:02d}")


def one(sym):
    try:
        return realised_spread_aggtrades(sym, DAYS)
    except Exception as e:  # noqa
        return {"symbol": sym, "status": f"err:{e}"[:80]}


def main():
    pool = json.load(open(SPEC / "universe_pool.json"))
    pool = [s for s in pool if s != "SLERFUSDT"][:28]
    print(f"aggTrades spread sample: {len(pool)} symbols x {len(DAYS)} days")
    t0 = time.time()
    res = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(one, s): s for s in pool}
        for fu in as_completed(futs):
            r = fu.result()
            res.append(r)
            print(f"  {r.get('symbol'):15} {r.get('status'):10} "
                  f"half_spread_bps={r.get('half_spread_bps')} ({time.time()-t0:.0f}s)", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    ok = [r for r in res if r.get("status") == "ok" and r.get("half_spread_bps") == r.get("half_spread_bps")]
    if ok:
        hs = sorted(r["half_spread_bps"] for r in ok)
        print(f"\nhalf-spread bps: median={np.median(hs):.2f} "
              f"p25={np.percentile(hs,25):.2f} p75={np.percentile(hs,75):.2f} "
              f"min={hs[0]:.2f} max={hs[-1]:.2f}")
    print(f"DONE {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
