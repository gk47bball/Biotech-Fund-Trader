"""
BacktestResults container and pretty-printer.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import BacktestConfig


class BacktestResults:
    def __init__(
        self,
        nav: pd.Series,
        benchmark: Optional[pd.Series],
        nav_df: pd.DataFrame,
        trade_log: pd.DataFrame,
        metrics: Dict,
        config: BacktestConfig,
    ):
        self.nav = nav
        self.benchmark = benchmark
        self.nav_df = nav_df
        self.trade_log = trade_log
        self.metrics = metrics
        self.config = config

    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        m = self.metrics
        cfg = self.config

        def _fmt(val, fmt):
            try:
                return format(val, fmt)
            except (TypeError, ValueError):
                return "N/A"

        print()
        print("=" * 62)
        print("  BIOTECH FUND BACKTEST RESULTS")
        print("=" * 62)
        if len(self.nav) >= 2:
            print(f"  Period      : {self.nav.index[0].date()} → {self.nav.index[-1].date()}")
        print(f"  Initial NAV : ${cfg.initial_capital:>14,.0f}")
        print(f"  Final NAV   : ${self.nav.iloc[-1]:>14,.0f}")
        print()
        print("  ── Returns ─────────────────────────────────────────")
        print(f"  Total Return     : {_fmt(m.get('total_return'), '.1%'):>10}")
        print(f"  CAGR             : {_fmt(m.get('cagr'), '.1%'):>10}")
        print(f"  Benchmark CAGR   : {_fmt(m.get('benchmark_cagr'), '.1%'):>10}  ({cfg.benchmark})")
        print(f"  Excess Return    : {_fmt(m.get('excess_return'), '.1%'):>10}")
        print()
        print("  ── Risk ────────────────────────────────────────────")
        print(f"  Ann. Volatility  : {_fmt(m.get('volatility'), '.1%'):>10}")
        print(f"  Max Drawdown     : {_fmt(m.get('max_drawdown'), '.1%'):>10}")
        print(f"  Max DD Duration  : {_fmt(m.get('max_dd_duration_days'), 'd'):>10} days")
        print(f"  Calmar Ratio     : {_fmt(m.get('calmar_ratio'), '.2f'):>10}")
        print()
        print("  ── Risk-Adjusted ───────────────────────────────────")
        print(f"  Sharpe Ratio     : {_fmt(m.get('sharpe_ratio'), '.2f'):>10}")
        print(f"  Sortino Ratio    : {_fmt(m.get('sortino_ratio'), '.2f'):>10}")
        print(f"  Alpha (CAPM)     : {_fmt(m.get('alpha'), '.1%'):>10}")
        print(f"  Beta             : {_fmt(m.get('beta'), '.2f'):>10}")
        print()
        print("  ── Monthly Returns ─────────────────────────────────")
        print(f"  Best Month       : {_fmt(m.get('best_month'), '.1%'):>10}")
        print(f"  Worst Month      : {_fmt(m.get('worst_month'), '.1%'):>10}")
        print(f"  Avg Month        : {_fmt(m.get('avg_month'), '.1%'):>10}")
        print()
        print("  ── Trades ──────────────────────────────────────────")
        print(f"  Total Trades     : {_fmt(m.get('total_trades'), 'd'):>10}")
        print(f"  Win Rate         : {_fmt(m.get('win_rate'), '.1%'):>10}")
        print(f"  Avg Win (% cap)  : {_fmt(m.get('avg_win'), '.2%'):>10}")
        print(f"  Avg Loss (% cap) : {_fmt(m.get('avg_loss'), '.2%'):>10}")
        print(f"  Profit Factor    : {_fmt(m.get('profit_factor'), '.2f'):>10}")
        print("=" * 62)

    def monthly_returns_table(self) -> pd.DataFrame:
        """Pivot table of monthly returns (year × month)."""
        monthly = self.nav.resample("ME").last().pct_change().dropna()
        monthly.index = pd.MultiIndex.from_arrays(
            [monthly.index.year, monthly.index.month],
            names=["Year", "Month"],
        )
        return monthly.unstack("Month").rename(
            columns={i: pd.Timestamp(2000, i, 1).strftime("%b") for i in range(1, 13)}
        )

    def top_holdings_snapshot(self, date: Optional[pd.Timestamp] = None) -> pd.Series:
        """Return position weights on the last (or specified) date."""
        if self.nav_df.empty:
            return pd.Series(dtype=float)
        return self.nav_df.get("n_positions", pd.Series(dtype=float))
