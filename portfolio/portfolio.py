"""
Portfolio state: cash, open positions, NAV history, trade log.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Position:
    ticker: str
    shares: float
    entry_price: float          # average cost per share (incl. slippage)
    entry_date: pd.Timestamp
    cost_basis: float           # total cash paid (incl. commissions)
    stop_price: float           # trailing stop level – rises but never falls
    current_price: float = 0.0  # updated every bar


class Portfolio:
    """
    Tracks cash, open positions, and NAV on each trading day.

    All prices supplied to buy/sell are *market* prices (Open of next bar in
    live; Close of current bar in this daily backtest). Slippage and
    commission are applied internally.
    """

    def __init__(
        self,
        initial_capital: float,
        commission_pct: float,
        slippage_pct: float,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        self.positions: Dict[str, Position] = {}
        self.nav_history: List[dict] = []
        self.trade_log: List[dict] = []
        self.peak_nav: float = initial_capital

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nav(self) -> float:
        equity = sum(p.shares * p.current_price for p in self.positions.values())
        return self.cash + equity

    @property
    def equity(self) -> float:
        return sum(p.shares * p.current_price for p in self.positions.values())

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def _tc(self, notional: float) -> float:
        """One-way transaction cost (commission + slippage)."""
        return abs(notional) * (self.commission_pct + self.slippage_pct)

    def buy(
        self,
        ticker: str,
        price: float,
        shares: float,
        date: pd.Timestamp,
        stop_pct: float = 0.25,
    ) -> None:
        if shares <= 0 or price <= 0:
            return

        eff_price = price * (1.0 + self.slippage_pct)   # fill above mid
        notional = eff_price * shares
        tc = notional * self.commission_pct              # slippage already in price
        total_outflow = notional + tc

        # Scale down if insufficient cash
        if total_outflow > self.cash:
            affordable = self.cash / (eff_price * (1.0 + self.commission_pct))
            shares = max(affordable, 0.0)
            if shares < 1.0:
                return
            notional = eff_price * shares
            tc = notional * self.commission_pct
            total_outflow = notional + tc

        if ticker in self.positions:
            pos = self.positions[ticker]
            total_shares = pos.shares + shares
            avg_price = (pos.entry_price * pos.shares + eff_price * shares) / total_shares
            pos.shares = total_shares
            pos.entry_price = avg_price
            pos.cost_basis += total_outflow
            pos.current_price = price
        else:
            stop = eff_price * (1.0 - stop_pct)
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                entry_price=eff_price,
                entry_date=date,
                cost_basis=total_outflow,
                stop_price=stop,
                current_price=price,
            )

        self.cash -= total_outflow

        self.trade_log.append(
            dict(
                date=date,
                ticker=ticker,
                action="BUY",
                shares=shares,
                price=eff_price,
                notional=notional,
                tc=tc,
                pnl=np.nan,
            )
        )
        logger.debug("BUY  %s × %.0f @ %.2f  cash=%.0f", ticker, shares, eff_price, self.cash)

    def sell(
        self,
        ticker: str,
        price: float,
        date: pd.Timestamp,
        reason: str = "SIGNAL",
        shares: Optional[float] = None,
    ) -> None:
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]
        sell_shares = shares if shares is not None else pos.shares
        sell_shares = min(sell_shares, pos.shares)
        if sell_shares <= 0:
            return

        eff_price = price * (1.0 - self.slippage_pct)   # fill below mid
        notional = eff_price * sell_shares
        tc = notional * self.commission_pct
        net_proceeds = notional - tc
        pnl = net_proceeds - pos.cost_basis * (sell_shares / pos.shares)

        self.cash += net_proceeds

        self.trade_log.append(
            dict(
                date=date,
                ticker=ticker,
                action="SELL",
                shares=sell_shares,
                price=eff_price,
                notional=notional,
                tc=tc,
                pnl=pnl,
                reason=reason,
            )
        )
        logger.debug(
            "SELL %s × %.0f @ %.2f  pnl=%.0f  reason=%s",
            ticker, sell_shares, eff_price, pnl, reason,
        )

        if sell_shares >= pos.shares:
            del self.positions[ticker]
        else:
            pos.shares -= sell_shares
            pos.cost_basis *= (pos.shares / (pos.shares + sell_shares))

    # ------------------------------------------------------------------
    # Price updates and NAV recording
    # ------------------------------------------------------------------

    def update_prices(self, prices: Dict[str, float]) -> None:
        for ticker, pos in self.positions.items():
            p = prices.get(ticker, np.nan)
            if p and not np.isnan(p):
                pos.current_price = p

    def record_nav(self, date: pd.Timestamp, prices: Dict[str, float]) -> None:
        self.update_prices(prices)
        current_nav = self.nav
        self.peak_nav = max(self.peak_nav, current_nav)
        drawdown = (current_nav / self.peak_nav) - 1.0 if self.peak_nav > 0 else 0.0

        self.nav_history.append(
            dict(
                date=date,
                nav=current_nav,
                cash=self.cash,
                cash_pct=self.cash / current_nav if current_nav > 0 else 1.0,
                n_positions=len(self.positions),
                peak_nav=self.peak_nav,
                drawdown=drawdown,
            )
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_nav_series(self) -> pd.Series:
        if not self.nav_history:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.nav_history)
        return df.set_index("date")["nav"]

    def get_nav_df(self) -> pd.DataFrame:
        if not self.nav_history:
            return pd.DataFrame()
        df = pd.DataFrame(self.nav_history)
        return df.set_index("date")

    def get_trade_log(self) -> pd.DataFrame:
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)
