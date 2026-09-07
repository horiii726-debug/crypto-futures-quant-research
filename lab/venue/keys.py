"""Secret loading + authenticated venue clients.

Keys come from (in priority order):
  1. environment variables  BINANCE_READ_KEY / BINANCE_READ_SECRET / BYBIT_* ...
  2. config/secrets.yaml     (gitignored)
Never logged, never returned in reports.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
_SECRETS_PATH = ROOT / "config" / "secrets.yaml"


def _load_file() -> dict:
    if _SECRETS_PATH.exists():
        return yaml.safe_load(_SECRETS_PATH.read_text()) or {}
    return {}


def get_key(venue: str, kind: str = "read") -> tuple[str, str]:
    """venue in {'binance','bybit'}, kind in {'read','testnet','trade'}."""
    ev = f"{venue.upper()}_{kind.upper()}"
    k = os.environ.get(f"{ev}_KEY", "")
    s = os.environ.get(f"{ev}_SECRET", "")
    if not (k and s):
        d = _load_file().get(venue, {}).get(kind, {}) or {}
        k, s = d.get("api_key", ""), d.get("api_secret", "")
    return k, s


def have(venue: str, kind: str = "read") -> bool:
    k, s = get_key(venue, kind)
    return bool(k and s)


def status() -> dict:
    out = {}
    for v in ("binance", "bybit"):
        for kind in ("read", "testnet", "trade"):
            out[f"{v}.{kind}"] = "SET" if have(v, kind) else "missing"
    out["secrets_file"] = "present" if _SECRETS_PATH.exists() else "absent (create config/secrets.yaml)"
    return out


# ---- minimal signed request helpers (no external SDK) --------------------

def _http(url, headers=None, data=None, method="GET"):
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"User-Agent": "crypto-lab/1.0", **(headers or {})})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def binance_signed(path: str, params: dict | None = None, *, kind="read",
                   base="https://fapi.binance.com", method="GET"):
    k, s = get_key("binance", kind)
    if not (k and s):
        raise RuntimeError(f"binance {kind} key not configured (config/secrets.yaml)")
    if kind == "testnet":
        base = "https://testnet.binancefuture.com"
    p = dict(params or {})
    p["timestamp"] = int(time.time() * 1000)
    p["recvWindow"] = 5000
    qs = urllib.parse.urlencode(p)
    sig = hmac.new(s.encode(), qs.encode(), hashlib.sha256).hexdigest()
    url = f"{base}{path}?{qs}&signature={sig}"
    return _http(url, headers={"X-MBX-APIKEY": k}, method=method)


def bybit_signed(path: str, params: dict | None = None, *, kind="read",
                 base="https://api.bybit.com", method="GET"):
    k, s = get_key("bybit", kind)
    if not (k and s):
        raise RuntimeError(f"bybit {kind} key not configured (config/secrets.yaml)")
    if kind == "testnet":
        base = "https://api-testnet.bybit.com"
    ts = str(int(time.time() * 1000))
    recv = "5000"
    p = dict(params or {})
    qs = urllib.parse.urlencode(sorted(p.items()))
    payload = ts + k + recv + qs
    sig = hmac.new(s.encode(), payload.encode(), hashlib.sha256).hexdigest()
    url = f"{base}{path}?{qs}"
    hdr = {"X-BAPI-API-KEY": k, "X-BAPI-TIMESTAMP": ts, "X-BAPI-RECV-WINDOW": recv,
           "X-BAPI-SIGN": sig}
    return _http(url, headers=hdr, method=method)


if __name__ == "__main__":
    import json as _j
    print(_j.dumps(status(), indent=2))
    if have("binance", "read"):
        try:
            r = binance_signed("/fapi/v2/account")
            print("binance account: assets=", len(r.get("assets", [])),
                  "canTrade=", r.get("canTrade"))
        except Exception as e:  # noqa
            print("binance check failed:", e)
    if have("bybit", "read"):
        try:
            r = bybit_signed("/v5/account/wallet-balance", {"accountType": "UNIFIED"})
            print("bybit wallet:", r.get("retMsg"))
        except Exception as e:  # noqa
            print("bybit check failed:", e)
