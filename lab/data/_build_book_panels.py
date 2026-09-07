import sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from lab.data.bookdepth import bar_features, PROC
SYMS=["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","BNBUSDT","LINKUSDT","AVAXUSDT","ADAUSDT","LTCUSDT","BCHUSDT","INJUSDT"]
MONTHS=[str(p) for p in pd.period_range("2023-09","2024-05",freq="M")]
BARS=[5,15]
CH=PROC/"_ch"
def task(s,m,b):
    o=CH/f"{s}_{m}_{b}m.parquet"
    if o.exists(): return s,m,b,"cached"
    per=pd.Period(m,"M")
    days=[d.strftime("%Y-%m-%d") for d in pd.date_range(per.start_time,min(per.end_time,pd.Timestamp("2024-05-17")),freq="D")]
    try: df=bar_features(s,days,b)
    except Exception as e: return s,m,b,f"err:{e}"[:50]
    if len(df): o.parent.mkdir(parents=True,exist_ok=True); df.to_parquet(o)
    return s,m,b,"ok" if len(df) else "empty"
def main():
    CH.mkdir(parents=True,exist_ok=True)
    tasks=[(s,m,b) for s in SYMS for m in MONTHS for b in BARS]
    print(f"{len(tasks)} chunks"); t0=time.time(); done=0
    with ProcessPoolExecutor(max_workers=6) as ex:
        for f in as_completed({ex.submit(task,*t):t for t in tasks}):
            s,m,b,st=f.result(); done+=1
            if st!="cached": print(f"  [{done}/{len(tasks)}] {s} {m} {b}m {st} ({time.time()-t0:.0f}s)",flush=True)
    for s in SYMS:
        for b in BARS:
            parts=sorted(CH.glob(f"{s}_*_{b}m.parquet"))
            if not parts: continue
            df=pd.concat([pd.read_parquet(p) for p in parts]).sort_index()
            df=df[~df.index.duplicated(keep="last")]
            df.to_parquet(PROC/f"{s}_{b}m.parquet")
            print(f"  panel {s} {b}m {df.shape} {df.index.min().date()}..{df.index.max().date()}")
    print(f"DONE {time.time()-t0:.0f}s")
if __name__=="__main__": main()
