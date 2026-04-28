"""Backtest metrics: total return, CAGR, vol, Sharpe, max drawdown, hit rate."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .backtest import BacktestResult


TRADING_DAYS = 252


def summarize(result: BacktestResult) -> dict:
    nav = result.nav_curve.dropna()
    if nav.empty or nav.iloc[0] <= 0:
        return {"error": "empty nav"}

    rets = nav.pct_change().dropna()
    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    days = (nav.index[-1] - nav.index[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1.0
    vol = rets.std() * math.sqrt(TRADING_DAYS) if len(rets) else 0.0
    sharpe = (rets.mean() * TRADING_DAYS) / (rets.std() * math.sqrt(TRADING_DAYS)) if rets.std() else 0.0
    max_dd = _max_drawdown(nav)

    closed = [t for t in result.trades if t.return_pct is not None]
    wins = [t for t in closed if (t.return_pct or 0) > 0]
    avg_ret = float(np.mean([t.return_pct for t in closed])) if closed else 0.0
    median_ret = float(np.median([t.return_pct for t in closed])) if closed else 0.0
    hold_days = float(np.mean([t.holding_days for t in closed if t.holding_days is not None])) if closed else 0.0

    out = {
        "starting_capital": float(nav.iloc[0]),
        "ending_capital":   float(nav.iloc[-1]),
        "total_return":     float(total_return),
        "cagr":             float(cagr),
        "annual_vol":       float(vol),
        "sharpe":           float(sharpe),
        "max_drawdown":     float(max_dd),
        "n_trades":         len(closed),
        "hit_rate":         float(len(wins) / len(closed)) if closed else 0.0,
        "avg_trade_return": avg_ret,
        "median_trade_return": median_ret,
        "avg_hold_days":    hold_days,
        "skipped_count":    len(result.skipped_entries),
    }

    if result.benchmark_curve is not None and not result.benchmark_curve.empty:
        bench = result.benchmark_curve.dropna()
        bench_return = bench.iloc[-1] / bench.iloc[0] - 1.0
        bench_cagr = (bench.iloc[-1] / bench.iloc[0]) ** (1 / years) - 1.0
        out["benchmark_total_return"] = float(bench_return)
        out["benchmark_cagr"] = float(bench_cagr)
        out["alpha_cagr"] = float(cagr - bench_cagr)
    return out


def _max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def per_fund_breakdown(result: BacktestResult) -> pd.DataFrame:
    rows = []
    for t in result.trades:
        if t.return_pct is None:
            continue
        rows.append({
            "fund": t.fund, "ticker": t.ticker, "issuer": t.issuer,
            "entry_date": t.entry_date, "exit_date": t.exit_date,
            "return": t.return_pct, "pnl": t.pnl, "hold_days": t.holding_days,
            "drift": t.drift, "exit_reason": t.exit_reason,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    grouped = df.groupby("fund").agg(
        n_trades=("return", "count"),
        avg_return=("return", "mean"),
        median_return=("return", "median"),
        hit_rate=("return", lambda s: float((s > 0).mean())),
        total_pnl=("pnl", "sum"),
        avg_hold_days=("hold_days", "mean"),
    ).round(4)
    return grouped.sort_values("total_pnl", ascending=False)
