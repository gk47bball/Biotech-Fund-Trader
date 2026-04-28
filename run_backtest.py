"""
Entry point for the Biotech Fund Backtest Engine.

Usage
-----
    # With EODHD API (token via env-var)
    export EODHD_API_TOKEN=6825543d2b2f23.48334508
    python run_backtest.py

    # With EODHD API (token via flag)
    python run_backtest.py --eodhd-token 6825543d2b2f23.48334508

    # Custom parameters
    python run_backtest.py --start 2020-01-01 --end 2023-12-31
    python run_backtest.py --capital 5000000 --rebalance weekly
    python run_backtest.py --top-n 10 --stop-loss 0.20

    # Monthly returns table
    python run_backtest.py --monthly-table
"""

import argparse
import logging
import os
import sys

from config import BacktestConfig
from engine.backtest import BacktestEngine

_DEFAULT_TOKEN = "6825543d2b2f23.48334508"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Biotech Fund Backtest Engine")
    p.add_argument("--start",        default="2019-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end",          default="2024-12-31", help="End date (YYYY-MM-DD)")
    p.add_argument("--capital",      type=float, default=10_000_000, help="Initial capital ($)")
    p.add_argument("--rebalance",    default="monthly", choices=["daily", "weekly", "monthly"])
    p.add_argument("--top-n",        type=int, default=15, help="Number of positions to hold")
    p.add_argument("--stop-loss",    type=float, default=0.25, help="Trailing stop fraction")
    p.add_argument("--blend",        type=float, default=0.70, help="Momentum signal weight (0–1)")
    p.add_argument("--benchmark",    default="XBI", help="Benchmark ticker")
    p.add_argument("--eodhd-token",  default=None, help="EODHD API token (overrides env-var)")
    p.add_argument("--no-cache",     action="store_true", help="Bypass disk cache")
    p.add_argument("--verbose",      action="store_true", help="DEBUG-level logging")
    p.add_argument("--monthly-table", action="store_true", help="Print monthly returns grid")
    return p.parse_args()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(stream=sys.stdout, level=level, format=fmt, datefmt="%H:%M:%S")
    for lib in ("urllib3", "requests", "peewee"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def main():
    args = _parse_args()
    _setup_logging(args.verbose)

    # Resolve EODHD token: CLI flag > env-var > built-in default
    token = (
        args.eodhd_token
        or os.getenv("EODHD_API_TOKEN", "")
        or _DEFAULT_TOKEN
    )

    cfg = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        rebalance_frequency=args.rebalance,
        top_n=args.top_n,
        stop_loss_pct=args.stop_loss,
        signal_blend=args.blend,
        benchmark=args.benchmark,
    )

    engine = BacktestEngine(cfg, eodhd_token=token)
    results = engine.run()

    results.print_summary()

    if args.monthly_table:
        import math
        print("\nMonthly Returns (%)")
        print("-" * 62)
        tbl = results.monthly_returns_table()
        fmt_tbl = tbl.map(lambda x: f"{x:.1%}" if isinstance(x, float) and not math.isnan(x) else "")
        print(fmt_tbl.to_string())

    return results


if __name__ == "__main__":
    main()
