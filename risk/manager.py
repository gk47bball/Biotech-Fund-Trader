"""
Risk management: trailing stops, portfolio-level drawdown halt,
target-weight construction, and trade delta computation.
"""

import logging
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd

from portfolio.portfolio import Position

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(
        self,
        max_position_pct: float = 0.08,
        min_position_pct: float = 0.01,
        stop_loss_pct: float = 0.25,
        portfolio_stop_pct: float = 0.20,
        max_positions: int = 20,
    ):
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.portfolio_stop_pct = portfolio_stop_pct
        self.max_positions = max_positions

    # ------------------------------------------------------------------
    # Stop-loss checks
    # ------------------------------------------------------------------

    def check_stops(
        self,
        positions: Dict[str, Position],
        current_prices: Dict[str, float],
    ) -> Set[str]:
        """
        Update trailing stops and return the set of tickers that have
        breached their stop price.
        """
        stopped: Set[str] = set()
        for ticker, pos in positions.items():
            price = current_prices.get(ticker, np.nan)
            if np.isnan(price) or price <= 0:
                continue

            # Raise trailing stop if price has moved up
            new_stop = price * (1.0 - self.stop_loss_pct)
            if new_stop > pos.stop_price:
                pos.stop_price = new_stop

            if price < pos.stop_price:
                logger.info(
                    "Stop triggered: %s @ %.2f (stop=%.2f, entry=%.2f, loss=%.1f%%)",
                    ticker,
                    price,
                    pos.stop_price,
                    pos.entry_price,
                    (price / pos.entry_price - 1.0) * 100,
                )
                stopped.add(ticker)
        return stopped

    def is_portfolio_halted(self, current_nav: float, peak_nav: float) -> bool:
        """Return True if portfolio drawdown from peak exceeds the limit."""
        if peak_nav <= 0:
            return False
        return (current_nav / peak_nav - 1.0) <= -self.portfolio_stop_pct

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def target_weights(
        self,
        signals: pd.Series,
        top_n: int,
    ) -> pd.Series:
        """
        Convert a cross-sectional signal series into equal-weight target
        portfolio weights, capped at max_position_pct.

        Returns a Series indexed by ticker with weights summing to ≤ 1.
        """
        valid = signals.dropna()
        if valid.empty:
            return pd.Series(dtype=float)

        top = valid.nlargest(min(top_n, len(valid), self.max_positions))
        if top.empty:
            return pd.Series(dtype=float)

        n = len(top)
        w = min(1.0 / n, self.max_position_pct)
        weights = pd.Series(w, index=top.index)

        # Normalise if capping caused sum < 1 but also guard against > 1
        total = weights.sum()
        if total > 1.0:
            weights /= total

        # Drop names below minimum meaningful size
        weights = weights[weights >= self.min_position_pct]
        return weights

    # ------------------------------------------------------------------
    # Trade generation
    # ------------------------------------------------------------------

    def compute_trades(
        self,
        target_weights: pd.Series,
        positions: Dict[str, Position],
        current_prices: Dict[str, float],
        nav: float,
        stopped: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        """
        Return {ticker: share_delta} where positive = buy and negative = sell.
        Trades below 0.5 % of NAV are suppressed to avoid churning.
        """
        if stopped is None:
            stopped = set()

        trades: Dict[str, float] = {}
        min_trade_value = nav * 0.005   # 50 bps threshold

        # --- Exits: stopped or not in target ---
        for ticker, pos in positions.items():
            if ticker in stopped or ticker not in target_weights.index:
                trades[ticker] = -pos.shares   # full liquidation

        # --- Entries / adjustments ---
        for ticker, target_w in target_weights.items():
            price = current_prices.get(ticker, np.nan)
            if np.isnan(price) or price <= 0:
                continue

            target_shares = (target_w * nav) / price
            current_shares = positions[ticker].shares if ticker in positions else 0.0
            delta = target_shares - current_shares

            if abs(delta * price) >= min_trade_value:
                trades[ticker] = delta

        return trades
