"""premiumIndexKlines (perp premium over index) + fundingRate, full history,
for the 55-symbol universe. Cheap. -> basis / delta-neutral carry study."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lab.data.binance_vision as bv  # noqa
from lab.data.ingest import month_range  # noqa

OUT = ROOT / "data" / "processed" / "basis"
OUT.mkdir(parents=True, exist_ok=True)
SYMS = json.load(open(ROOT / "data/raw/_spec/universe_pool.json"))
SYMS = [s for s in SYMS if s != "SLERFUSDT"]
MONTHS = month_range("2023-09", "2025-08")


def one(sym):
    f = OUT / f"{sym}.parquet"
    if f.exists():
        return sym, "cached", len(pd.read_parquet(f))
    try:
        pi = bv.index_klines(sym, MONTHS, kind="premiumIndexKlines", interval="1h")
        fr = bv.funding(sym, MONTHS)
    except Exception as e:  # noqa
        return sym, f"err:{e}"[:60], 0
    if pi.empty:
        return sym, "no_pi", 0
    pi = pi.set_index("dt")[["close"]].rename(columns={"close": "premium"})
    if len(fr):
        fpi = fr.set_index("dt")["fundingRate"].reindex(pi.index, method="ffill")
        pi["funding"] = fpi
    pi.to_parquet(f)
    return sym, "ok", len(pi)


def main():
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(one, SYMS))
    ok = sum(1 for _, s, _ in res if s in ("ok", "cached"))
    print(f"basis: {ok}/{len(SYMS)} ok in {time.time()-t0:.0f}s")
    for s, st, n in res:
        if st not in ("ok", "cached"):
            print(f"  {s} {st}")


if __name__ == "__main__":
    main()
