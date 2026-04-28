"""
Cross-sectional signal generation for the biotech universe.

Two alpha sources are blended:
  1. 12-1 month price momentum (skip most-recent month to avoid reversal)
  2. Short-term mean reversion (z-score of 20-day returns, inverted)

Both signals are cross-sectionally ranked (0 → 1) before blending so that
the weights are comparable regardless of signal magnitude.
"""

import numpy as np
import pandas as pd


class SignalGenerator:
    def __init__(
        self,
        momentum_lookback: int = 252,
        momentum_skip: int = 21,
        reversion_lookback: int = 20,
        signal_blend: float = 0.70,
    ):
        self.momentum_lookback = momentum_lookback
        self.momentum_skip = momentum_skip
        self.reversion_lookback = reversion_lookback
        self.signal_blend = signal_blend

    # ------------------------------------------------------------------
    # Individual signals
    # ------------------------------------------------------------------

    def momentum_signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        """(t-lookback to t-skip) total return, cross-sectionally ranked."""
        ret = prices.shift(self.momentum_skip) / prices.shift(self.momentum_lookback) - 1
        return ret.rank(axis=1, pct=True)

    def mean_reversion_signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Negative z-score of rolling returns → buy recent underperformers.
        Cross-sectionally ranked so high rank = most oversold.
        """
        r = prices.pct_change()
        mu = r.rolling(self.reversion_lookback).mean()
        sigma = r.rolling(self.reversion_lookback).std().replace(0.0, np.nan)
        z = (r - mu) / sigma
        return (-z).rank(axis=1, pct=True)

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def volatility_filter(
        self, prices: pd.DataFrame, max_ann_vol: float = 1.50
    ) -> pd.DataFrame:
        """True where 63-day realised vol ≤ max_ann_vol (default 150% ann.)."""
        ann_vol = prices.pct_change().rolling(63).std() * np.sqrt(252)
        return ann_vol <= max_ann_vol

    def liquidity_filter(
        self, volumes: pd.DataFrame, min_20d_avg_shares: float = 50_000
    ) -> pd.DataFrame:
        """True where 20-day average share volume ≥ threshold."""
        return volumes.rolling(20).mean() >= min_20d_avg_shares

    # ------------------------------------------------------------------
    # Combined signal
    # ------------------------------------------------------------------

    def combined_signal(
        self, prices: pd.DataFrame, volumes: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Blended signal matrix (dates × tickers).
        NaN where a ticker fails either filter on that date.
        """
        mom = self.momentum_signal(prices)
        rev = self.mean_reversion_signal(prices)

        combined = self.signal_blend * mom + (1.0 - self.signal_blend) * rev

        vol_ok = self.volatility_filter(prices)
        liq_ok = self.liquidity_filter(volumes)

        # Mask filtered-out tickers with NaN so they are skipped in ranking
        return combined.where(vol_ok & liq_ok, other=np.nan)
