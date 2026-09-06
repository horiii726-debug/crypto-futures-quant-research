"""data.binance.vision client - free bulk historical data, no API key.

Covers: symbol discovery (incl. delisted, for survivorship), monthly/daily
file download with SHA256 verification, and light zip->DataFrame parsing.
"""
from __future__ import annotations

import hashlib
import io
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://data.binance.vision"
LIST_HOST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw"

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
AGGTRADE_COLS = ["agg_id", "price", "qty", "first_id", "last_id", "ts", "is_buyer_maker"]
BOOKTICKER_COLS = ["update_id", "best_bid", "best_bid_qty", "best_ask", "best_ask_qty",
                   "transaction_time", "event_time"]


class NotFound(RuntimeError):
    pass


def _get(url: str, timeout: int = 120, retries: int = 4) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "crypto-lab/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):     # file genuinely absent - do NOT retry
                raise NotFound(f"{e.code} {url}")
            last = e
            time.sleep(1.0 * (i + 1))
        except Exception as e:  # noqa
            last = e
            time.sleep(1.0 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


# ---- symbol discovery (survivorship) ----------------------------------------

def list_symbols(market: str = "futures/um", kind: str = "klines",
                 freq: str = "daily") -> list[str]:
    """Every symbol that ever had data of this kind - includes delisted."""
    prefix = f"data/{market}/{freq}/{kind}/"
    out, marker = [], ""
    while True:
        url = f"{LIST_HOST}?delimiter=/&prefix={prefix}"
        if marker:
            url += f"&marker={urllib.parse.quote(marker)}"
        xml = _get(url, timeout=60).decode()
        got = re.findall(rf"<Prefix>{re.escape(prefix)}([^/]+)/</Prefix>", xml)
        out.extend(got)
        m = re.search(r"<NextMarker>([^<]+)</NextMarker>", xml)
        trunc = re.search(r"<IsTruncated>(\w+)</IsTruncated>", xml)
        if trunc and trunc.group(1) == "true" and m:
            marker = m.group(1)
        else:
            break
    return sorted(set(out))


def list_files(symbol: str, market: str = "futures/um", kind: str = "klines",
               freq: str = "monthly", interval: str | None = "1m") -> list[str]:
    seg = f"{kind}/{symbol}" + (f"/{interval}" if interval else "")
    prefix = f"data/{market}/{freq}/{seg}/"
    out, marker = [], ""
    while True:
        url = f"{LIST_HOST}?prefix={prefix}"
        if marker:
            url += f"&marker={urllib.parse.quote(marker)}"
        xml = _get(url, timeout=60).decode()
        keys = re.findall(r"<Key>([^<]+\.zip)</Key>", xml)
        out.extend(keys)
        m = re.search(r"<NextMarker>([^<]+)</NextMarker>", xml)
        trunc = re.search(r"<IsTruncated>(\w+)</IsTruncated>", xml)
        if trunc and trunc.group(1) == "true":
            marker = m.group(1) if m else (keys[-1] if keys else "")
            if not marker:
                break
        else:
            break
    return sorted(out)


# ---- download + verify ------------------------------------------------------

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_file(key: str, verify: bool = True) -> bytes:
    """key like data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"""
    dst = CACHE / key
    if dst.exists():
        return dst.read_bytes()
    blob = _get(f"{BASE}/{key}")
    if verify:
        try:
            chk = _get(f"{BASE}/{key}.CHECKSUM").decode().split()[0]
            if _sha256(blob) != chk:
                raise RuntimeError(f"checksum mismatch for {key}")
        except RuntimeError:
            raise
        except Exception:
            pass  # checksum file sometimes absent for very old data
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(blob)
    return blob


def _read_zip_csv(blob: bytes, cols: list[str]) -> pd.DataFrame:
    """Read the single CSV in the zip and assign `cols` BY POSITION.

    Binance bulk files are inconsistent: some carry a header row (with column
    names that vary across kinds/years), some do not. We detect a header by
    checking whether the first field of row 1 is numeric, skip it if present,
    and always name columns positionally so downstream code sees one schema.
    """
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    first_field = raw.split(b"\n", 1)[0].split(b",")[0].strip().decode(errors="ignore")
    has_header = not re.fullmatch(r"-?\d+(\.\d+)?([eE]-?\d+)?", first_field or "")
    df = pd.read_csv(io.BytesIO(raw), header=None, skiprows=1 if has_header else 0,
                     names=cols)
    return df


def klines(symbol: str, months: list[str], interval: str = "1m",
           freq: str = "monthly") -> pd.DataFrame:
    frames = []
    for mo in months:
        key = f"data/futures/um/{freq}/klines/{symbol}/{interval}/{symbol}-{interval}-{mo}.zip"
        try:
            blob = fetch_file(key)
        except RuntimeError as e:
            if "404" in str(e) or "failed" in str(e):
                continue
            raise
        df = _read_zip_csv(blob, KLINE_COLS)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=KLINE_COLS)
    df = pd.concat(frames, ignore_index=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    # normalise microsecond timestamps (Binance switched in 2025) to ms
    if (df["open_time"] > 2_000_000_000_000_000).any():
        df.loc[df["open_time"] > 2_000_000_000_000_000, "open_time"] //= 1000
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def agg_trades(symbol: str, days: list[str]) -> pd.DataFrame:
    frames = []
    for d in days:
        key = f"data/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{d}.zip"
        try:
            blob = fetch_file(key)
        except RuntimeError:
            continue
        df = _read_zip_csv(blob, AGGTRADE_COLS)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=AGGTRADE_COLS)
    df = pd.concat(frames, ignore_index=True)
    for c in ["price", "qty"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("int64")
    if (df["ts"] > 2_000_000_000_000_000).any():
        df.loc[df["ts"] > 2_000_000_000_000_000, "ts"] //= 1000
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"])
    return df.sort_values("ts").reset_index(drop=True)


FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]


def funding(symbol: str, months: list[str]) -> pd.DataFrame:
    """Funding-rate history from data.binance.vision bulk files (no REST, no
    rate-limit ban risk)."""
    frames = []
    for mo in months:
        key = f"data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{mo}.zip"
        try:
            blob = fetch_file(key, verify=False)
        except RuntimeError:
            continue
        frames.append(_read_zip_csv(blob, FUNDING_COLS))
    if not frames:
        return pd.DataFrame(columns=FUNDING_COLS + ["dt", "fundingRate"])
    df = pd.concat(frames, ignore_index=True)
    df["calc_time"] = pd.to_numeric(df["calc_time"], errors="coerce").astype("int64")
    if (df["calc_time"] > 2_000_000_000_000_000).any():
        df.loc[df["calc_time"] > 2_000_000_000_000_000, "calc_time"] //= 1000
    df["fundingRate"] = pd.to_numeric(df["last_funding_rate"], errors="coerce")
    df["dt"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
    return df.drop_duplicates("calc_time").sort_values("calc_time").reset_index(drop=True)


if __name__ == "__main__":
    syms = list_symbols()
    print(f"{len(syms)} symbols ever listed (futures/um klines)")
    print("sample:", syms[:5], "...", syms[-5:])
