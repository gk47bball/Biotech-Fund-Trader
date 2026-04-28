"""
Synthetic biotech price-series generator (offline / CI fallback).

Models each stock as correlated Geometric Brownian Motion with
biotech-realistic parameters:
  - Annual drift  :  5 – 25 %  (random per ticker)
  - Annual vol    : 35 – 90 %  (much higher than large-cap equities)
  - Market beta   : 0.6 – 1.4  vs a common sector factor
  - Rare jumps    : ±20 – 50 %  (FDA events, data read-outs)

The benchmark (XBI proxy) is a value-weighted average of the simulated
universe with added noise.
"""

import hashlib
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ticker_seed(ticker: str) -> int:
    """Deterministic seed from ticker string so results are reproducible."""
    return int(hashlib.md5(ticker.encode()).hexdigest(), 16) % (2**31)


def generate(
    tickers: List[str],
    start: str,
    end: str,
    benchmark: Optional[str] = None,
    base_price: float = 80.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    """
    Returns
    -------
    prices     : DataFrame[date × ticker]  (adjusted close)
    volumes    : DataFrame[date × ticker]
    bench      : Series[date] or None
    """
    trading_days = pd.bdate_range(start=start, end=end, freq="B")
    n = len(trading_days)
    dt = 1.0 / 252.0

    # ── Sector factor (common to all tickers) ──────────────────────────
    sector_rng = np.random.default_rng(42)
    sector_mu  = 0.08          # 8% sector drift
    sector_vol = 0.28          # 28% sector vol
    sector_z   = sector_rng.standard_normal(n)
    sector_ret = (sector_mu - 0.5 * sector_vol**2) * dt + sector_vol * np.sqrt(dt) * sector_z
    sector_idx = np.exp(np.cumsum(sector_ret))

    prices_dict:  dict = {}
    volumes_dict: dict = {}

    for ticker in tickers:
        rng = np.random.default_rng(_ticker_seed(ticker))

        # Per-ticker parameters
        mu       = rng.uniform(0.05, 0.25)       # annual drift
        sigma    = rng.uniform(0.35, 0.90)       # annual vol
        beta     = rng.uniform(0.60, 1.40)       # sector beta
        idio_vol = np.sqrt(max(sigma**2 - beta**2 * sector_vol**2, 0.05**2))
        s0       = base_price * rng.uniform(0.4, 3.0)

        # Jump process (FDA-style)
        jump_freq  = rng.uniform(0.5, 2.0)       # jumps per year
        jump_p_pos = rng.uniform(0.35, 0.55)     # P(positive jump)
        jump_size  = rng.uniform(0.20, 0.50)     # abs magnitude

        # Idiosyncratic returns
        idio_z = rng.standard_normal(n)
        idio_ret = (
            (mu - beta * sector_mu - 0.5 * idio_vol**2) * dt
            + idio_vol * np.sqrt(dt) * idio_z
        )

        # Combined return with sector factor
        combined_ret = beta * sector_ret + idio_ret

        # Inject jumps
        jump_times = rng.poisson(jump_freq * dt, n).astype(bool)
        for t in np.where(jump_times)[0]:
            sign  = 1 if rng.random() < jump_p_pos else -1
            mag   = rng.uniform(0.15, jump_size)
            combined_ret[t] += sign * mag

        log_price = np.log(s0) + np.cumsum(combined_ret)
        price_series = np.exp(log_price)

        # Volume: mean ~500k shares, correlated with abs return
        base_vol_shares = rng.uniform(200_000, 2_000_000)
        abs_ret = np.abs(combined_ret)
        vol_noise = rng.lognormal(0, 0.4, n)
        volume_series = base_vol_shares * (1.0 + 3.0 * abs_ret / abs_ret.std()) * vol_noise

        prices_dict[ticker]  = price_series
        volumes_dict[ticker] = volume_series

    prices_df  = pd.DataFrame(prices_dict,  index=trading_days)
    volumes_df = pd.DataFrame(volumes_dict, index=trading_days)

    # ── Benchmark: equal-weight basket + noise ─────────────────────────
    bench_series: Optional[pd.Series] = None
    if benchmark is not None:
        bench_rng = np.random.default_rng(999)
        bench_noise = bench_rng.standard_normal(n) * 0.005
        bench_raw = sector_idx * 100 * np.exp(np.cumsum(bench_noise))
        bench_series = pd.Series(bench_raw, index=trading_days, name=benchmark)

    logger.info(
        "Synthetic data: %d tickers | %d trading days (%s → %s)",
        len(tickers), n,
        trading_days[0].date(), trading_days[-1].date(),
    )
    return prices_df, volumes_df, bench_series
