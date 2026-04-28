"""
Core event-driven backtest engine.

Daily loop
----------
1. Update position prices with today's close.
2. Check portfolio-level drawdown halt.
3. Check individual trailing stops → liquidate stopped positions.
4. On rebalance days: compute signals → target weights → execute trades.
5. Record NAV.

Re-entry after portfolio halt
------------------------------
Once the portfolio drawdown limit is hit, all positions are liquidated.
The engine waits until the benchmark has recovered 5% from its trough
before re-entering (max 63-day cool-down).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import BacktestConfig
from data.loader import DataLoader
from engine._reporter import BacktestResults
from metrics.performance import PerformanceMetrics
from portfolio.portfolio import Portfolio
from risk.manager import RiskManager
from strategy.signals import SignalGenerator

logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(
        self,
        config: BacktestConfig,
        eodhd_token: Optional[str] = None,
        force_synthetic: bool = False,
    ):
        self.config = config
        self._loader = DataLoader(
            api_token=eodhd_token,
            cache=True,
            force_synthetic=force_synthetic,
        )
        self._signals = SignalGenerator(
            momentum_lookback=config.momentum_lookback,
            momentum_skip=config.momentum_skip,
            reversion_lookback=config.reversion_lookback,
            signal_blend=config.signal_blend,
        )
        self._risk = RiskManager(
            max_position_pct=config.max_position_pct,
            min_position_pct=config.min_position_pct,
            stop_loss_pct=config.stop_loss_pct,
            portfolio_stop_pct=config.portfolio_stop_pct,
            max_positions=config.max_positions,
        )
        self._portfolio = Portfolio(
            initial_capital=config.initial_capital,
            commission_pct=config.commission_pct,
            slippage_pct=config.slippage_pct,
        )

    # ------------------------------------------------------------------
    # Rebalance calendar
    # ------------------------------------------------------------------

    def _is_rebalance_day(
        self, date: pd.Timestamp, prev: Optional[pd.Timestamp]
    ) -> bool:
        if prev is None:
            return True
        freq = self.config.rebalance_frequency
        if freq == "daily":
            return True
        if freq == "weekly":
            return date.isocalendar().week != prev.isocalendar().week
        if freq == "monthly":
            return date.month != prev.month
        return False

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> "BacktestResults":
        cfg = self.config
        _banner(cfg)

        # ── 1. Load data ─────────────────────────────────────────────────
        prices, volumes, benchmark = self._loader.load(
            tickers=cfg.universe,
            start=cfg.start_date,
            end=cfg.end_date,
            benchmark=cfg.benchmark,
        )

        # ── 2. Pre-compute signal matrix (vectorised) ────────────────────
        logger.info("Pre-computing signal matrix …")
        signal_matrix = self._signals.combined_signal(prices, volumes)

        # ── 3. Event loop ────────────────────────────────────────────────
        portfolio = self._portfolio
        risk = self._risk
        prev_date: Optional[pd.Timestamp] = None
        portfolio_halted = False
        halt_date: Optional[pd.Timestamp] = None
        bench_trough: float = np.inf
        rebalance_count = 0

        trading_days = prices.index
        logger.info("Running event loop: %d trading days …", len(trading_days))

        for date in trading_days:
            prices_today: dict = prices.loc[date].to_dict()

            # ── a) Price update ──────────────────────────────────────────
            portfolio.update_prices(prices_today)

            # ── b) Portfolio drawdown halt ───────────────────────────────
            if not portfolio_halted:
                if risk.is_portfolio_halted(portfolio.nav, portfolio.peak_nav):
                    logger.warning(
                        "%s: Portfolio drawdown limit hit (%.1f%%). Moving to cash.",
                        date.date(),
                        (portfolio.nav / portfolio.peak_nav - 1) * 100,
                    )
                    for ticker in list(portfolio.positions.keys()):
                        p = prices_today.get(ticker, np.nan)
                        if p and not np.isnan(p):
                            portfolio.sell(ticker, p, date, "PORTFOLIO_HALT")
                    portfolio_halted = True
                    halt_date = date
                    bench_trough = (
                        benchmark.loc[date] if benchmark is not None else np.inf
                    )

            # ── c) Re-entry after halt ───────────────────────────────────
            if portfolio_halted and halt_date is not None:
                days_in_cash = (date - halt_date).days
                if benchmark is not None and date in benchmark.index:
                    b_now = benchmark.loc[date]
                    bench_trough = min(bench_trough, b_now)
                    bench_recovery = b_now / bench_trough - 1.0 if bench_trough > 0 else 0.0
                else:
                    bench_recovery = 0.0

                if days_in_cash >= 63 or bench_recovery >= 0.05:
                    logger.info(
                        "%s: Re-entering market (days_cash=%d, bench_recovery=%.1f%%).",
                        date.date(), days_in_cash, bench_recovery * 100,
                    )
                    portfolio_halted = False
                    halt_date = None
                    bench_trough = np.inf

            # ── d) Individual stops ──────────────────────────────────────
            if not portfolio_halted and portfolio.positions:
                stopped = risk.check_stops(portfolio.positions, prices_today)
                for ticker in stopped:
                    p = prices_today.get(ticker, np.nan)
                    if p and not np.isnan(p):
                        portfolio.sell(ticker, p, date, "STOP_LOSS")

            # ── e) Rebalance ─────────────────────────────────────────────
            if not portfolio_halted and self._is_rebalance_day(date, prev_date):
                rebalance_count += 1
                if date in signal_matrix.index:
                    today_signals = signal_matrix.loc[date]
                    target_w = risk.target_weights(today_signals, top_n=cfg.top_n)

                    trades = risk.compute_trades(
                        target_weights=target_w,
                        positions=portfolio.positions,
                        current_prices=prices_today,
                        nav=portfolio.nav,
                    )

                    # Sells first (free up cash)
                    for ticker, delta in sorted(trades.items(), key=lambda kv: kv[1]):
                        if delta < 0:
                            p = prices_today.get(ticker, np.nan)
                            if p and not np.isnan(p):
                                portfolio.sell(ticker, p, date, "REBALANCE")

                    # Then buys
                    for ticker, delta in sorted(trades.items(), key=lambda kv: -kv[1]):
                        if delta > 0:
                            p = prices_today.get(ticker, np.nan)
                            if p and not np.isnan(p):
                                portfolio.buy(ticker, p, delta, date, cfg.stop_loss_pct)

            # ── f) Record NAV ────────────────────────────────────────────
            portfolio.record_nav(date, prices_today)
            prev_date = date

        logger.info("Event loop complete. Rebalances: %d", rebalance_count)

        # ── 4. Build results ─────────────────────────────────────────────
        nav_series = portfolio.get_nav_series()
        trade_log = portfolio.get_trade_log()

        metrics = PerformanceMetrics(
            nav=nav_series,
            benchmark=benchmark,
            initial_capital=cfg.initial_capital,
            risk_free_rate=cfg.risk_free_rate,
            trade_log=trade_log,
        ).compute()

        return BacktestResults(
            nav=nav_series,
            benchmark=benchmark,
            nav_df=portfolio.get_nav_df(),
            trade_log=trade_log,
            metrics=metrics,
            config=cfg,
        )


# ──────────────────────────────────────────────────────────────────────
def _banner(cfg: BacktestConfig) -> None:
    logger.info("=" * 62)
    logger.info("  BIOTECH FUND BACKTEST ENGINE")
    logger.info("=" * 62)
    logger.info("  Period    : %s → %s", cfg.start_date, cfg.end_date)
    logger.info("  Capital   : $%s", f"{cfg.initial_capital:,.0f}")
    logger.info("  Universe  : %d tickers", len(cfg.universe))
    logger.info("  Top-N     : %d", cfg.top_n)
    logger.info("  Rebalance : %s", cfg.rebalance_frequency)
    logger.info("  Benchmark : %s", cfg.benchmark)
    logger.info("=" * 62)
