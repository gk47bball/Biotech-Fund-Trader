"""Portfolio backtest.

Logic:
  1. For each entry signal, on the next trading day after the 13F filing date:
       - Reference price  = adjusted close on `report_date` (quarter-end)
       - Entry price      = next-trading-day open (or close if open missing)
       - Skip if |entry / reference - 1| > drift_threshold
       - Allocate position_size_pct * NAV; cap at available cash.
  2. Each (fund, ticker) lot is independent. If the same ticker is entered by
     multiple funds, each lot is tracked separately.
  3. For each exit signal from the originating fund, sell the corresponding
     open lot at next-trading-day open after the filing date.
  4. NAV = cash + sum(open lots marked at adjusted close).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from .config import StrategyParams
from .prices import PriceClient
from .signals import Signal


@dataclass
class Lot:
    fund: str
    cusip: str
    ticker: str
    issuer: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    cost: float
    report_date: str
    ref_price: float
    drift: float


@dataclass
class Trade:
    fund: str
    cusip: str
    ticker: str
    issuer: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp | None
    exit_price: float | None
    shares: float
    cost: float
    proceeds: float | None
    pnl: float | None
    return_pct: float | None
    holding_days: int | None
    report_date: str
    ref_price: float
    drift: float
    exit_reason: str | None = None


@dataclass
class BacktestResult:
    trades: list[Trade]
    skipped_entries: list[dict]
    nav_curve: pd.Series
    benchmark_curve: pd.Series | None
    params: StrategyParams


def _next_trading_day(prices: PriceClient, ticker: str, after: pd.Timestamp) -> tuple[pd.Timestamp, dict] | None:
    df = prices.get_eod(ticker)
    if df.empty:
        return None
    sub = df.loc[df.index > after]
    if sub.empty:
        return None
    return sub.index[0], sub.iloc[0].to_dict()


def run_backtest(
    signals: list[Signal],
    cusip_to_ticker: dict[str, str | None],
    prices: PriceClient,
    params: StrategyParams,
) -> BacktestResult:
    open_lots: dict[tuple[str, str], Lot] = {}
    trades: list[Trade] = []
    skipped: list[dict] = []

    cash = params.starting_capital
    nav_history: dict[pd.Timestamp, float] = {}

    start = pd.Timestamp(params.start_date)
    end = pd.Timestamp(params.end_date)

    # We need to process events in filing-date order. nav is updated only on
    # event days; we'll resample the curve to daily afterwards using mark-to-market.
    for sig in signals:
        ticker = cusip_to_ticker.get(sig.cusip)
        if not ticker:
            skipped.append({"reason": "no_ticker", **sig.__dict__})
            continue

        filing_ts = pd.Timestamp(sig.filing_date)
        if filing_ts < start or filing_ts > end:
            continue

        nxt = _next_trading_day(prices, ticker, filing_ts)
        if nxt is None:
            skipped.append({"reason": "no_price_after_filing", "ticker": ticker, **sig.__dict__})
            continue
        trade_date, bar = nxt
        if trade_date > end:
            continue
        entry_price = bar.get("open") or bar.get("adjusted_close") or bar.get("close")
        if not entry_price or entry_price <= 0:
            skipped.append({"reason": "bad_price", "ticker": ticker, **sig.__dict__})
            continue

        key = (sig.fund, sig.cusip)

        if sig.kind == "entry":
            if key in open_lots:
                # Fund added more on top of existing. We don't pyramid; ignore.
                continue
            ref_price = prices.price_on_or_before(ticker, sig.report_date)
            if not ref_price or ref_price <= 0:
                skipped.append({"reason": "no_ref_price", "ticker": ticker, **sig.__dict__})
                continue
            drift = entry_price / ref_price - 1.0
            if abs(drift) > params.drift_threshold:
                skipped.append({
                    "reason": "drift_filter",
                    "ticker": ticker, "drift": drift, **sig.__dict__,
                })
                continue
            if params.max_positions and len(open_lots) >= params.max_positions:
                skipped.append({"reason": "max_positions", "ticker": ticker, **sig.__dict__})
                continue

            nav = _mark_to_market(cash, open_lots, prices, trade_date)
            target_dollars = params.position_size_pct * nav
            spend = min(target_dollars, cash)
            if spend <= 0:
                skipped.append({"reason": "no_cash", "ticker": ticker, **sig.__dict__})
                continue
            shares = spend / entry_price
            cash -= spend
            open_lots[key] = Lot(
                fund=sig.fund, cusip=sig.cusip, ticker=ticker, issuer=sig.issuer,
                entry_date=trade_date, entry_price=entry_price, shares=shares, cost=spend,
                report_date=sig.report_date, ref_price=ref_price, drift=drift,
            )
            nav_history[trade_date] = _mark_to_market(cash, open_lots, prices, trade_date)

        elif sig.kind == "exit":
            lot = open_lots.pop(key, None)
            if lot is None:
                continue
            proceeds = lot.shares * entry_price
            cash += proceeds
            holding_days = (trade_date - lot.entry_date).days
            trades.append(Trade(
                fund=lot.fund, cusip=lot.cusip, ticker=lot.ticker, issuer=lot.issuer,
                entry_date=lot.entry_date, entry_price=lot.entry_price,
                exit_date=trade_date, exit_price=entry_price,
                shares=lot.shares, cost=lot.cost, proceeds=proceeds,
                pnl=proceeds - lot.cost,
                return_pct=proceeds / lot.cost - 1.0 if lot.cost else None,
                holding_days=holding_days,
                report_date=lot.report_date, ref_price=lot.ref_price, drift=lot.drift,
                exit_reason="fund_exit",
            ))
            nav_history[trade_date] = _mark_to_market(cash, open_lots, prices, trade_date)

    # Force-close any still-open lots at end_date for accounting completeness.
    for key, lot in list(open_lots.items()):
        nxt = _next_trading_day(prices, lot.ticker, end - pd.Timedelta(days=1))
        if nxt is None:
            continue
        d, bar = nxt
        px = bar.get("adjusted_close") or bar.get("close")
        if not px:
            continue
        proceeds = lot.shares * px
        cash += proceeds
        trades.append(Trade(
            fund=lot.fund, cusip=lot.cusip, ticker=lot.ticker, issuer=lot.issuer,
            entry_date=lot.entry_date, entry_price=lot.entry_price,
            exit_date=d, exit_price=px,
            shares=lot.shares, cost=lot.cost, proceeds=proceeds,
            pnl=proceeds - lot.cost,
            return_pct=proceeds / lot.cost - 1.0 if lot.cost else None,
            holding_days=(d - lot.entry_date).days,
            report_date=lot.report_date, ref_price=lot.ref_price, drift=lot.drift,
            exit_reason="end_of_backtest",
        ))
        del open_lots[key]
        nav_history[d] = cash + _mark_to_market(0.0, open_lots, prices, d)

    nav_curve = _build_nav_curve(trades, params, prices)
    bench = _benchmark_curve(prices, params)
    return BacktestResult(
        trades=trades, skipped_entries=skipped,
        nav_curve=nav_curve, benchmark_curve=bench, params=params,
    )


def _mark_to_market(cash: float, lots: dict[tuple[str, str], Lot], prices: PriceClient,
                    on_date: pd.Timestamp) -> float:
    total = cash
    for lot in lots.values():
        px = prices.price_on_or_before(lot.ticker, on_date)
        if px is None:
            px = lot.entry_price
        total += lot.shares * px
    return total


def _build_nav_curve(trades: list[Trade], params: StrategyParams, prices: PriceClient) -> pd.Series:
    """Reconstruct daily NAV: replay trades day-by-day using adjusted closes."""
    if not trades:
        idx = pd.bdate_range(params.start_date, params.end_date)
        return pd.Series(params.starting_capital, index=idx, name="nav")

    # Build lot intervals per ticker, then sum each day.
    tickers = sorted({t.ticker for t in trades})
    price_panel: dict[str, pd.Series] = {}
    for tk in tickers:
        df = prices.get_eod(tk)
        if not df.empty:
            price_panel[tk] = df["adjusted_close"]

    idx = pd.bdate_range(params.start_date, params.end_date)
    cash_series = pd.Series(0.0, index=idx)
    cash_series.iloc[0] = params.starting_capital

    holdings = pd.DataFrame(0.0, index=idx, columns=tickers)
    for t in trades:
        d_in = pd.Timestamp(t.entry_date)
        d_out = pd.Timestamp(t.exit_date) if t.exit_date is not None else idx[-1]
        d_in_eff = max(d_in, idx[0])
        d_out_eff = min(d_out, idx[-1])
        if d_in_eff > d_out_eff:
            continue
        # Subtract cost on entry day, add proceeds on exit day.
        if d_in >= idx[0] and d_in <= idx[-1]:
            cash_series.loc[d_in] -= t.cost
        if t.exit_date is not None and idx[0] <= d_out <= idx[-1] and t.proceeds is not None:
            cash_series.loc[d_out] += t.proceeds
        # Hold shares from d_in inclusive to d_out exclusive (we MTM with close so include d_in).
        mask = (idx >= d_in_eff) & (idx < d_out_eff)
        holdings.loc[mask, t.ticker] += t.shares

    cash_curve = cash_series.cumsum()
    holdings_value = pd.Series(0.0, index=idx)
    for tk, series in price_panel.items():
        s = series.reindex(idx).ffill()
        holdings_value = holdings_value + holdings[tk] * s.fillna(0.0)
    nav = cash_curve + holdings_value
    nav.name = "nav"
    return nav


def _benchmark_curve(prices: PriceClient, params: StrategyParams) -> pd.Series | None:
    df = prices.get_eod(params.benchmark)
    if df.empty:
        return None
    idx = pd.bdate_range(params.start_date, params.end_date)
    s = df["adjusted_close"].reindex(idx).ffill()
    s = s / s.iloc[0] * params.starting_capital
    s.name = "benchmark"
    return s
