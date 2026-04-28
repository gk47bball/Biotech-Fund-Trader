"""
EODHD Historical Data API client.

Endpoint used
-------------
GET https://eodhd.com/api/eod/{TICKER}.US
    ?api_token=<token>
    &from=YYYY-MM-DD
    &to=YYYY-MM-DD
    &fmt=json
    &period=d

Response shape (list of dicts)
-------------------------------
[
  {"date":"2019-01-02","open":196.10,"high":200.03,
   "low":195.36,"close":199.55,"adjusted_close":196.30,"volume":2147483},
  ...
]

Usage
-----
    client = EODHDClient(api_token="xxxx")
    prices, volumes = client.fetch_universe(
        tickers=["AMGN","GILD"],
        start="2019-01-01",
        end="2024-12-31",
    )
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://eodhd.com/api/eod"
_CACHE_DIR = Path(".backtest_cache")
_EXCHANGE = "US"
_REQUEST_DELAY = 0.25   # seconds between API calls (free tier ~20 req/s)
_TIMEOUT = 30           # HTTP timeout in seconds


class EODHDClient:
    """Thin wrapper around the EODHD EOD historical-data endpoint."""

    def __init__(self, api_token: str, cache: bool = True):
        self.api_token = api_token
        self.cache = cache
        if cache:
            _CACHE_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_universe(
        self,
        tickers: List[str],
        start: str,
        end: str,
        benchmark: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
        """
        Download adjusted-close prices and volumes for every ticker.

        Returns
        -------
        prices     : DataFrame[date × ticker]  (adjusted_close)
        volumes    : DataFrame[date × ticker]
        benchmark  : Series[date] or None
        """
        all_tickers = list(tickers)
        if benchmark and benchmark not in all_tickers:
            all_tickers = all_tickers + [benchmark]

        frames: Dict[str, pd.DataFrame] = {}
        for i, ticker in enumerate(all_tickers):
            try:
                df = self._fetch_ticker(ticker, start, end)
                if df is not None and not df.empty:
                    frames[ticker] = df
                else:
                    logger.warning("No data returned for %s", ticker)
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", ticker, exc)

            # Polite delay between requests (skip after last)
            if i < len(all_tickers) - 1:
                time.sleep(_REQUEST_DELAY)

        if not frames:
            raise RuntimeError("EODHD returned no data for any ticker.")

        close_dict = {t: df["adjusted_close"] for t, df in frames.items()}
        vol_dict   = {t: df["volume"]         for t, df in frames.items()}

        prices  = pd.DataFrame(close_dict).sort_index()
        volumes = pd.DataFrame(vol_dict).sort_index()

        prices  = prices.ffill().bfill()
        volumes = volumes.fillna(0.0)

        universe_cols = [t for t in tickers if t in prices.columns]
        missing = set(tickers) - set(universe_cols)
        if missing:
            logger.warning("Tickers unavailable: %s", sorted(missing))

        bench_series: Optional[pd.Series] = None
        if benchmark and benchmark in prices.columns:
            bench_series = prices[benchmark]

        coverage = prices[universe_cols].notna().mean()
        low = coverage[coverage < 0.80]
        if not low.empty:
            logger.warning("Low coverage (<80%%): %s", low.to_dict())

        logger.info(
            "EODHD: %d/%d tickers loaded | %d trading days (%s → %s)",
            len(universe_cols), len(tickers),
            len(prices),
            prices.index[0].date() if len(prices) else "?",
            prices.index[-1].date() if len(prices) else "?",
        )
        return prices[universe_cols], volumes[universe_cols], bench_series

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_ticker(
        self, ticker: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        cache_path = _CACHE_DIR / f"eodhd_{ticker}_{start}_{end}.pkl"

        if self.cache and cache_path.exists():
            logger.debug("Cache hit: %s", ticker)
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)

        url = f"{_BASE_URL}/{ticker}.{_EXCHANGE}"
        params = {
            "api_token": self.api_token,
            "from": start,
            "to": end,
            "fmt": "json",
            "period": "d",
        }

        logger.debug("Fetching %s from EODHD …", ticker)
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        if not data:
            return None

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # Keep only the columns we need; rename for consistency
        keep = ["open", "high", "low", "close", "adjusted_close", "volume"]
        df = df[[c for c in keep if c in df.columns]]
        df = df.apply(pd.to_numeric, errors="coerce")

        if self.cache:
            with open(cache_path, "wb") as fh:
                pickle.dump(df, fh)

        return df
