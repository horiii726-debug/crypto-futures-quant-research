"""Coin-M (inverse) liquidation snapshots for BTCUSD_PERP + ETHUSD_PERP,
full window (2023-06 .. 2024-09). Every forced-liquidation order. Small.
-> F_LIQD market-wide liquidation-pressure timing signal."""
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab.data.binance_vision import fetch_file, NotFound
SYMS=["BTCUSD_PERP","ETHUSD_PERP"]
DAYS=[d.strftime("%Y-%m-%d") for d in pd.date_range("2023-06-25","2024-09-20")]
def one(sym,d):
    key=f"data/futures/cm/daily/liquidationSnapshot/{sym}/{sym}-liquidationSnapshot-{d}.zip"
    if (ROOT/"data"/"raw"/key).exists(): return "cached"
    try: fetch_file(key,verify=False); return "ok"
    except NotFound: return "404"
    except Exception: return "err"
def main():
    tasks=[(s,d) for s in SYMS for d in DAYS]
    t0=time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        r=list(ex.map(lambda a:one(*a),tasks))
    from collections import Counter
    print(Counter(r), f"{time.time()-t0:.0f}s")
if __name__=="__main__": main()
