"""CoinAPI.io client. FREE TIER = ~100 credits/day, each OHLCV call returns up
to 100 bars = 1 credit. This is for SPOT-CHECKS and venues without a free bulk
archive (Coinbase, Kraken, OKX, Deribit) - NOT bulk history. For Binance/Bybit
use data.binance.vision / public.bybit.com (free, unlimited, no key)."""
from __future__ import annotations
import json, os, time, urllib.parse, urllib.request
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://rest.coinapi.io/v1"


def _key() -> str:
    k = os.environ.get("COINAPI_KEY", "")
    if not k:
        p = ROOT / "config" / "secrets.yaml"
        if p.exists():
            k = ((yaml.safe_load(p.read_text()) or {}).get("coinapi", {}) or {}).get("api_key", "")
    return k


def _get(path, params=None):
    k = _key()
    if not k:
        raise RuntimeError("COINAPI_KEY not set (config/secrets.yaml -> coinapi.api_key)")
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"X-CoinAPI-Key": k, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=25) as r:
        used = r.headers.get("X-RateLimit-Used"); rem = r.headers.get("X-RateLimit-Remaining")
        data = json.loads(r.read())
    return data, {"used": used, "remaining": rem}


def quota():
    _, h = _get("/exchanges", {"filter_exchange_id": "BINANCE"})
    return h


def ohlcv(symbol_id: str, period="1HRS", limit=100, start=None):
    """symbol_id e.g. 'BINANCEFTS_PERP_BTC_USDT', 'OKEX_PERP_BTC_USDT',
    'DERIBIT_PERP_BTC_USD'. See /symbols for the list."""
    p = {"period_id": period, "limit": min(limit, 100)}
    if start:
        p["time_start"] = start
    return _get(f"/ohlcv/{symbol_id}/history", p)


def list_perp_symbols(exchange_id: str):
    d, h = _get("/symbols", {"filter_symbol_id": f"{exchange_id}_PERP"})
    return [s["symbol_id"] for s in d], h


if __name__ == "__main__":
    try:
        print("quota:", quota())
    except Exception as e:  # noqa
        print("coinapi not configured or failed:", e)
