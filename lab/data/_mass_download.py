"""Download -> Rust-process -> delete-raw pipeline. Keeps disk bounded while
building the full tape + book bar panels."""
import shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from lab.data.binance_vision import fetch_file, NotFound
BIN = ROOT / "rustkernels" / "target" / "release"
RAW = ROOT / "data" / "raw" / "data" / "futures" / "um" / "daily"
TAPE_OUT = ROOT / "data" / "processed" / "tape"
BOOK_OUT = ROOT / "data" / "processed" / "bookdepth_bars"

TAPE_SYMS = ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","BNBUSDT","LINKUSDT","AVAXUSDT",
             "ADAUSDT","LTCUSDT","ZECUSDT","ARBUSDT","UNIUSDT","NEARUSDT","SUIUSDT","AAVEUSDT",
             "INJUSDT","1000PEPEUSDT","WLDUSDT","TAOUSDT"]
TAPE_MONTHS = [str(p) for p in pd.period_range("2023-09","2025-08",freq="M")]
BOOK_SYMS = TAPE_SYMS[:16]
BOOK_MONTHS = [str(p) for p in pd.period_range("2023-01","2024-05",freq="M")]


def _dl(kind, sym, day):
    key = f"data/futures/um/daily/{kind}/{sym}/{sym}-{kind}-{day}.zip"
    try:
        fetch_file(key, verify=False); return 1
    except (NotFound, Exception):
        return 0


def _rust(kind, sym, bar, days, out_dir):
    exe = BIN / ("tape" if kind == "aggTrades" else "book")
    zdir = RAW / kind / sym
    r = subprocess.run([str(exe), sym, str(bar), str(zdir), *days], capture_output=True, text=True, timeout=3600)
    if r.returncode != 0 or "\n" not in r.stdout:
        return None
    import io
    df = pd.read_csv(io.StringIO(r.stdout))
    df["dt"] = pd.to_datetime(df["t0"], unit="ms", utc=True)
    df = df.set_index("dt").drop(columns=["t0"])
    if kind == "aggTrades":
        df["cum_delta"] = df["ofi_usd"].cumsum()
        rr = df["close"] / df["open"]
        df["ret"] = np.log(rr.where(rr > 0)); df["intensity"] = df["n_trades"] / (bar * 60.0)
        rr2 = df["vwap"] / df["vwap"].shift(1); df["twap_ret"] = np.log(rr2.where(rr2 > 0))
    else:
        df["depth_withdraw"] = df["total_depth"].pct_change()
    return df


def process_symbol(kind, sym, months, bars, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / f"{sym}_{bars[-1]}m.parquet"
    if done_marker.exists():
        return sym, "cached"
    per = [pd.Period(m, "M") for m in months]
    days = []
    for p in per:
        end = min(p.end_time, pd.Timestamp("2024-05-17") if kind == "bookDepth" else p.end_time)
        days += [d.strftime("%Y-%m-%d") for d in pd.date_range(p.start_time, end, freq="D")]
    # download all days (threaded)
    t = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        got = sum(ex.map(lambda d: _dl(kind, sym, d), days))
    # rust process each bar size
    panels = {}
    for b in bars:
        df = _rust(kind, sym, b, days, out_dir)
        if df is not None and len(df):
            df.to_parquet(out_dir / f"{sym}_{b}m.parquet")
            panels[b] = df.shape
    # delete raw for this symbol
    zdir = RAW / kind / sym
    if zdir.exists():
        shutil.rmtree(zdir)
    return sym, f"got={got} panels={panels} {time.time()-t:.0f}s"


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("tape", "all"):
        print(f"=== TAPE: {len(TAPE_SYMS)} syms x 24mo, bars 5/15 ===", flush=True)
        for s in TAPE_SYMS:
            print(" ", *process_symbol("aggTrades", s, TAPE_MONTHS, [5, 15], TAPE_OUT), flush=True)
    if which in ("book", "all"):
        print(f"=== BOOK: {len(BOOK_SYMS)} syms x 17mo, bars 5/15 ===", flush=True)
        for s in BOOK_SYMS:
            print(" ", *process_symbol("bookDepth", s, BOOK_MONTHS, [5, 15], BOOK_OUT), flush=True)
    print("MASS DOWNLOAD DONE", flush=True)


if __name__ == "__main__":
    main()
