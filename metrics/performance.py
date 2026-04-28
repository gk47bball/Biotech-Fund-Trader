"""
Quantitative performance analytics for a completed backtest.

Metrics computed
----------------
Total return, CAGR, benchmark CAGR, excess return
Annualised volatility, max drawdown, Calmar ratio
Sharpe ratio (annualised), Sortino ratio
Alpha (CAPM), Beta
Trade-level: total trades, win rate, avg win, avg loss, profit factor
Monthly return distribution (best / worst / avg month)
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


class PerformanceMetrics:
    TRADING_DAYS = 252

    def __init__(
        self,
        nav: pd.Series,
        benchmark: Optional[pd.Series],
        initial_capital: float,
        risk_free_rate: float = 0.045,
        trade_log: Optional[pd.DataFrame] = None,
    ):
        self.nav = nav
        self.benchmark = benchmark
        self.initial_capital = initial_capital
        self.rfr = risk_free_rate
        self.trade_log = trade_log

    # ------------------------------------------------------------------
    def compute(self) -> Dict:
        nav = self.nav.copy()
        ret = nav.pct_change().dropna()

        if len(nav) < 2:
            return {}

        # ── Return metrics ───────────────────────────────────────────────
        total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
        n_years = len(nav) / self.TRADING_DAYS
        cagr = (1.0 + total_return) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

        # ── Risk metrics ─────────────────────────────────────────────────
        vol = ret.std() * np.sqrt(self.TRADING_DAYS)

        rolling_max = nav.cummax()
        dd_series = nav / rolling_max - 1.0
        max_dd = dd_series.min()

        calmar = abs(cagr / max_dd) if max_dd < 0 else np.nan

        # ── Sharpe ───────────────────────────────────────────────────────
        daily_rfr = self.rfr / self.TRADING_DAYS
        excess_daily = ret - daily_rfr
        sharpe = (
            excess_daily.mean() / ret.std() * np.sqrt(self.TRADING_DAYS)
            if ret.std() > 0 else np.nan
        )

        # ── Sortino ──────────────────────────────────────────────────────
        downside_ret = ret[ret < daily_rfr]
        downside_vol = downside_ret.std() * np.sqrt(self.TRADING_DAYS)
        sortino = (cagr - self.rfr) / downside_vol if downside_vol > 0 else np.nan

        # ── Benchmark / alpha / beta ─────────────────────────────────────
        bench_cagr = alpha = beta = excess_return = 0.0
        if self.benchmark is not None:
            b = self.benchmark.reindex(nav.index).ffill().bfill()
            b_ret = b.pct_change().dropna()
            common = ret.index.intersection(b_ret.index)
            if len(common) > 10:
                r_c = ret.reindex(common)
                b_c = b_ret.reindex(common)

                bench_total = b.iloc[-1] / b.iloc[0] - 1.0
                bench_cagr = (1.0 + bench_total) ** (1.0 / n_years) - 1.0

                cov_mat = np.cov(r_c.values, b_c.values)
                beta = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] > 0 else 0.0
                alpha = cagr - (self.rfr + beta * (bench_cagr - self.rfr))
                excess_return = cagr - bench_cagr

        # ── Monthly returns ──────────────────────────────────────────────
        monthly_nav = nav.resample("ME").last()
        monthly_ret = monthly_nav.pct_change().dropna()
        best_month = monthly_ret.max()
        worst_month = monthly_ret.min()
        avg_month = monthly_ret.mean()

        # ── Max drawdown duration ────────────────────────────────────────
        dd_dur = self._max_drawdown_duration(nav)

        # ── Trade-level stats ────────────────────────────────────────────
        trade_stats = self._trade_stats()

        return dict(
            total_return=total_return,
            cagr=cagr,
            benchmark_cagr=bench_cagr,
            excess_return=excess_return,
            volatility=vol,
            max_drawdown=max_dd,
            max_dd_duration_days=dd_dur,
            calmar_ratio=calmar,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            alpha=alpha,
            beta=beta,
            best_month=best_month,
            worst_month=worst_month,
            avg_month=avg_month,
            **trade_stats,
        )

    # ------------------------------------------------------------------
    def _trade_stats(self) -> Dict:
        defaults = dict(
            total_trades=0,
            win_rate=np.nan,
            avg_win=np.nan,
            avg_loss=np.nan,
            profit_factor=np.nan,
        )
        if self.trade_log is None or self.trade_log.empty:
            return defaults

        sells = self.trade_log[self.trade_log["action"] == "SELL"].copy()
        sells = sells.dropna(subset=["pnl"])
        if sells.empty:
            return defaults

        wins = sells[sells["pnl"] > 0]["pnl"]
        losses = sells[sells["pnl"] <= 0]["pnl"]

        total_win = wins.sum()
        total_loss = abs(losses.sum())

        return dict(
            total_trades=len(sells),
            win_rate=len(wins) / len(sells) if len(sells) > 0 else np.nan,
            avg_win=wins.mean() / self.initial_capital if not wins.empty else np.nan,
            avg_loss=losses.mean() / self.initial_capital if not losses.empty else np.nan,
            profit_factor=total_win / total_loss if total_loss > 0 else np.nan,
        )

    @staticmethod
    def _max_drawdown_duration(nav: pd.Series) -> int:
        """Return max number of calendar days spent below a previous peak."""
        peak_date = nav.index[0]
        peak_val = nav.iloc[0]
        max_dur = 0
        for date, val in nav.items():
            if val >= peak_val:
                peak_val = val
                peak_date = date
            else:
                dur = (date - peak_date).days
                max_dur = max(max_dur, dur)
        return max_dur
