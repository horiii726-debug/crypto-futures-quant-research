"""Full daily bookDepth for ~12 liquid coins over its ENTIRE available window
(2023-01 .. 2024-05). 30-second snapshots, +-1%..+-5% bands. ~0.5 MB/day.
-> proper intraday F_BOOK study."""
import sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab.data.binance_vision import fetch_file, NotFound
SYMS = ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","BNBUSDT","LINKUSDT",
        "AVAXUSDT","ADAUSDT","LTCUSDT","BCHUSDT","INJUSDT"]
DAYS = [d.strftime("%Y-%m-%d") for d in pd.date_range("2023-09-01","2024-05-17")]

def one(sym, d):
    key = f"data/futures/um/daily/bookDepth/{sym}/{sym}-bookDepth-{d}.zip"
    p = ROOT/"data"/"raw"/key
    if p.exists(): return "cached", p.stat().st_size
    try:
        b = fetch_file(key, verify=False); return "ok", len(b)
    except NotFound: return "404", 0
    except Exception as e: return f"err", 0

def main():
    tasks=[(s,d) for s in SYMS for d in DAYS]
    print(f"bookDepth v2: {len(SYMS)} syms x {len(DAYS)} days = {len(tasks)}")
    t0=time.time(); done=by=0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(one,s,d):(s,d) for s,d in tasks}
        for f in as_completed(futs):
            st,n=f.result(); done+=1; by+=n
            if done%1000==0: print(f"  [{done}/{len(tasks)}] {by/1e9:.1f}GB {time.time()-t0:.0f}s",flush=True)
    print(f"DONE {by/1e9:.2f}GB {time.time()-t0:.0f}s")

if __name__=="__main__": main()
