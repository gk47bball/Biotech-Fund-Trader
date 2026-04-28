"""EODHD price client. Daily OHLCV per ticker, cached as parquet on disk."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from .config import PRICES_CACHE, APIConfig

EODHD_BASE = "https://eodhd.com/api"


class PriceClient:
    """Fetches and caches daily adjusted prices for US tickers."""

    def __init__(self, api: APIConfig):
        if not api.eodhd_api_key:
            raise RuntimeError("EODHD_API_KEY not set")
        self.api_key = api.eodhd_api_key
        self._mem: dict[str, pd.DataFrame] = {}

    def _path(self, ticker: str) -> Path:
        safe = ticker.replace("/", "_").replace(".", "_")
        return PRICES_CACHE / f"{safe}.parquet"

    def get_eod(self, ticker: str) -> pd.DataFrame:
        """Return DataFrame indexed by date with columns: open, high, low, close, adjusted_close, volume.

        Returns an empty DataFrame on lookup failure (e.g., delisted/unknown).
        """
        if ticker in self._mem:
            return self._mem[ticker]

        path = self._path(ticker)
        if path.exists():
            df = pd.read_parquet(path)
            self._mem[ticker] = df
            return df

        url = f"{EODHD_BASE}/eod/{ticker}.US"
        params = {"api_token": self.api_key, "fmt": "json", "period": "d"}
        for attempt in range(5):
            try:
                r = requests.get(url, params=params, timeout=30)
                if r.status_code == 404:
                    df = pd.DataFrame()
                    self._mem[ticker] = df
                    return df
                if r.status_code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                r.raise_for_status()
                rows = r.json()
                break
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
        else:
            df = pd.DataFrame()
            self._mem[ticker] = df
            return df

        if not rows:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            keep = ["open", "high", "low", "close", "adjusted_close", "volume"]
            df = df[[c for c in keep if c in df.columns]]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not df.empty:
            df.to_parquet(path)
        self._mem[ticker] = df
        return df

    def price_on_or_before(self, ticker: str, date: str | pd.Timestamp, field: str = "adjusted_close") -> float | None:
        df = self.get_eod(ticker)
        if df.empty:
            return None
        ts = pd.Timestamp(date)
        sub = df.loc[df.index <= ts]
        if sub.empty:
            return None
        return float(sub.iloc[-1][field])

    def price_on_or_after(self, ticker: str, date: str | pd.Timestamp, field: str = "adjusted_close") -> tuple[pd.Timestamp, float] | None:
        df = self.get_eod(ticker)
        if df.empty:
            return None
        ts = pd.Timestamp(date)
        sub = df.loc[df.index >= ts]
        if sub.empty:
            return None
        row = sub.iloc[0]
        return sub.index[0], float(row[field])
