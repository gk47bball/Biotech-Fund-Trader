"""CUSIP -> US ticker mapping via OpenFIGI, with on-disk JSON cache.

OpenFIGI is free and CUSIP-aware. Without an API key the rate limit is ~25
requests/min; with a key it's higher. We batch up to 100 CUSIPs per call.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from .config import CUSIP_CACHE, APIConfig

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"

_US_EXCHANGES = {
    "US",  # composite
    "UN", "UQ", "UR", "UA", "UB", "UF", "UI", "UM", "UV", "UW", "UX", "UD", "UP",
}


def _load_cache() -> dict[str, str | None]:
    if CUSIP_CACHE.exists():
        return json.loads(CUSIP_CACHE.read_text())
    return {}


def _save_cache(cache: dict[str, str | None]) -> None:
    CUSIP_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def map_cusips(cusips: list[str], api: APIConfig) -> dict[str, str | None]:
    """Return {cusip: ticker_or_None}. Misses cached as None to avoid retry storms."""
    cache = _load_cache()
    todo = sorted({c for c in cusips if c and c not in cache})
    if not todo:
        return {c: cache.get(c) for c in cusips}

    headers = {"Content-Type": "application/json"}
    if api.openfigi_api_key:
        headers["X-OPENFIGI-APIKEY"] = api.openfigi_api_key

    batch_size = 100
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        body = [{"idType": "ID_CUSIP", "idValue": c} for c in batch]
        for attempt in range(5):
            r = requests.post(OPENFIGI_URL, headers=headers, json=body, timeout=30)
            if r.status_code == 429:
                time.sleep(6 * (attempt + 1))
                continue
            if r.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        else:
            raise RuntimeError("OpenFIGI rate-limit exceeded; supply OPENFIGI_API_KEY")

        for cusip, result in zip(batch, r.json()):
            cache[cusip] = _pick_ticker(result)
        _save_cache(cache)
        # Polite pacing for the unauthenticated rate limit.
        time.sleep(0.3 if api.openfigi_api_key else 2.5)

    return {c: cache.get(c) for c in cusips}


def _pick_ticker(result: dict) -> str | None:
    data = result.get("data") or []
    # Prefer common-stock equity on a US exchange.
    for entry in data:
        if (entry.get("securityType2") or "").lower() == "common stock" \
                and entry.get("exchCode") in _US_EXCHANGES:
            return entry.get("ticker")
    for entry in data:
        if entry.get("marketSector") == "Equity" and entry.get("exchCode") in _US_EXCHANGES:
            return entry.get("ticker")
    for entry in data:
        if entry.get("ticker"):
            return entry["ticker"]
    return None
