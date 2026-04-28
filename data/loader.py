"""
DataLoader — selects the right price-data backend.

Priority
--------
1. EODHD API (when api_token is provided and network is reachable)
2. Synthetic GBM data (offline / CI fallback)

The caller only ever sees the normalised (prices, volumes, benchmark) triple;
the backend is transparent.
"""

import logging
import os
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Read token from env-var so it is never hard-coded in source files
_ENV_TOKEN = os.getenv("EODHD_API_TOKEN", "")


class DataLoader:
    """
    Unified loader that wraps EODHD or synthetic backends.

    Parameters
    ----------
    api_token : EODHD API token.  If empty/None the synthetic backend is used.
    cache     : Persist per-ticker responses on disk (.backtest_cache/).
    force_synthetic : Always use synthetic data (useful for unit tests / CI).
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        cache: bool = True,
        force_synthetic: bool = False,
    ):
        # Resolve token: explicit arg > env-var
        self.api_token = api_token or _ENV_TOKEN or ""
        self.cache = cache
        self.force_synthetic = force_synthetic

    # ------------------------------------------------------------------
    def load(
        self,
        tickers: List[str],
        start: str,
        end: str,
        benchmark: str = "XBI",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
        """
        Returns
        -------
        prices     : DataFrame[date × ticker]  adjusted close
        volumes    : DataFrame[date × ticker]
        benchmark  : Series[date] or None
        """
        if self.force_synthetic or not self.api_token:
            return self._load_synthetic(tickers, start, end, benchmark)

        try:
            return self._load_eodhd(tickers, start, end, benchmark)
        except Exception as exc:
            logger.warning(
                "EODHD fetch failed (%s). Falling back to synthetic data.", exc
            )
            return self._load_synthetic(tickers, start, end, benchmark)

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _load_eodhd(
        self, tickers, start, end, benchmark
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
        from data.eodhd import EODHDClient

        client = EODHDClient(api_token=self.api_token, cache=self.cache)
        return client.fetch_universe(
            tickers=tickers, start=start, end=end, benchmark=benchmark
        )

    def _load_synthetic(
        self, tickers, start, end, benchmark
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
        from data.synthetic import generate

        logger.info("Using synthetic price data (no EODHD token / network unavailable).")
        return generate(
            tickers=tickers, start=start, end=end, benchmark=benchmark
        )
