"""CLI entrypoint: fetch filings, build signals, run backtest, save report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .config import APIConfig, FUNDS, OUTPUT_DIR, StrategyParams
from .cusip_map import map_cusips
from .metrics import per_fund_breakdown, summarize
from .prices import PriceClient
from .sec_edgar import fetch_all_funds, list_13f_filings
from .signals import all_signals


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="biotech_backtest", description="13F copy-trade backtest")
    sub = p.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", help="Run the full backtest")
    bt.add_argument("--start", default="2014-01-01")
    bt.add_argument("--end", default="2024-12-31")
    bt.add_argument("--drift-threshold", type=float, default=0.15)
    bt.add_argument("--add-threshold", type=float, default=0.25)
    bt.add_argument("--exit-trim-threshold", type=float, default=0.50)
    bt.add_argument("--position-size-pct", type=float, default=0.02)
    bt.add_argument("--starting-capital", type=float, default=1_000_000.0)
    bt.add_argument("--max-positions", type=int, default=0)
    bt.add_argument("--benchmark", default="SPY")
    bt.add_argument("--funds", nargs="+", default=None,
                    help="Optional subset of fund names from config.FUNDS")
    bt.add_argument("--output-prefix", default="run")

    sub.add_parser("verify-funds", help="Print configured fund CIKs and most-recent 13F dates")
    return p


def cmd_verify_funds(api: APIConfig) -> int:
    rows = []
    for name, cik in FUNDS.items():
        try:
            listings = list_13f_filings(cik, api)
            if listings:
                latest = max(listings, key=lambda f: f["filing_date"])
                rows.append((name, cik, latest["filing_date"], latest["report_date"], len(listings)))
            else:
                rows.append((name, cik, "-", "-", 0))
        except Exception as e:
            rows.append((name, cik, f"ERROR: {e}", "", 0))
    print(f"{'Fund':<28} {'CIK':<12} {'Last filed':<12} {'Last quarter':<12} {'#13Fs'}")
    for r in rows:
        print(f"{r[0]:<28} {r[1]:<12} {str(r[2]):<12} {str(r[3]):<12} {r[4]}")
    return 0


def cmd_backtest(args: argparse.Namespace, api: APIConfig) -> int:
    funds = FUNDS if not args.funds else {k: v for k, v in FUNDS.items() if k in set(args.funds)}
    if not funds:
        print("No funds matched.", file=sys.stderr)
        return 2

    params = StrategyParams(
        drift_threshold=args.drift_threshold,
        add_threshold=args.add_threshold,
        exit_trim_threshold=args.exit_trim_threshold,
        position_size_pct=args.position_size_pct,
        max_positions=args.max_positions,
        starting_capital=args.starting_capital,
        start_date=args.start, end_date=args.end,
        benchmark=args.benchmark,
    )

    print(f"[1/5] Fetching 13F filings for {len(funds)} funds...")
    filings_by_fund = fetch_all_funds(funds, params.start_date, params.end_date, api)
    total_filings = sum(len(v) for v in filings_by_fund.values())
    print(f"      {total_filings} filings parsed.")

    print("[2/5] Generating signals...")
    signals = all_signals(filings_by_fund, params)
    print(f"      {len(signals)} raw signals "
          f"({sum(s.kind == 'entry' for s in signals)} entries, "
          f"{sum(s.kind == 'exit' for s in signals)} exits)")

    print("[3/5] Mapping CUSIPs to tickers via OpenFIGI...")
    cusips = sorted({s.cusip for s in signals})
    cusip_to_ticker = map_cusips(cusips, api)
    mapped = sum(1 for v in cusip_to_ticker.values() if v)
    print(f"      mapped {mapped}/{len(cusips)} CUSIPs")

    print("[4/5] Running backtest...")
    prices = PriceClient(api)
    result = run_backtest(signals, cusip_to_ticker, prices, params)

    print("[5/5] Writing report...")
    summary = summarize(result)
    breakdown = per_fund_breakdown(result)

    out_dir = OUTPUT_DIR
    prefix = args.output_prefix
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.DataFrame([t.__dict__ for t in result.trades]).to_csv(
        out_dir / f"{prefix}_trades.csv", index=False,
    )
    pd.DataFrame(result.skipped_entries).to_csv(
        out_dir / f"{prefix}_skipped.csv", index=False,
    )
    result.nav_curve.to_csv(out_dir / f"{prefix}_nav.csv")
    if result.benchmark_curve is not None:
        result.benchmark_curve.to_csv(out_dir / f"{prefix}_benchmark.csv")
    if not breakdown.empty:
        breakdown.to_csv(out_dir / f"{prefix}_per_fund.csv")

    _print_summary(summary, breakdown)
    print(f"\nArtifacts written under: {out_dir}")
    return 0


def _print_summary(summary: dict, breakdown: pd.DataFrame) -> None:
    print("\n=== Strategy Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<26} {v:>12.4f}")
        else:
            print(f"  {k:<26} {v}")
    if not breakdown.empty:
        print("\n=== Per-Fund Breakdown ===")
        print(breakdown.to_string())


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    api = APIConfig()
    if not api.eodhd_api_key:
        print("ERROR: EODHD_API_KEY env var not set.", file=sys.stderr)
        return 2

    if args.cmd == "verify-funds":
        return cmd_verify_funds(api)
    if args.cmd == "backtest":
        return cmd_backtest(args, api)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
