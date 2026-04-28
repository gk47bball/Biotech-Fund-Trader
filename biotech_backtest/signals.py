"""Signal generation: diff consecutive 13Fs into entry/exit events.

Entry rule: a (fund, cusip) is an entry when shares went from 0 -> >0 (new),
or grew by more than `add_threshold` quarter-over-quarter (meaningful add).
The entry filter is applied later, in build_trades(), once we know prices.

Exit rule: a (fund, cusip) is an exit when shares went to 0, or shrank by more
than `exit_trim_threshold`.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import StrategyParams
from .sec_edgar import Filing


@dataclass
class Signal:
    fund: str
    cusip: str
    issuer: str
    report_date: str       # quarter-end of the filing producing this signal
    filing_date: str       # SEC accepted date (used for entry timing)
    kind: str              # "entry" or "exit"
    prev_shares: float
    new_shares: float


def _aggregate_share_positions(filing: Filing) -> dict[str, tuple[float, str]]:
    """Sum shares across class lines for the same CUSIP; ignore puts/calls.

    Returns {cusip: (shares, issuer)}.
    """
    pos: dict[str, tuple[float, str]] = {}
    for h in filing.holdings:
        if h.put_call:  # skip option exposure
            continue
        if not h.cusip:
            continue
        prev_shares, prev_name = pos.get(h.cusip, (0.0, h.issuer))
        pos[h.cusip] = (prev_shares + h.shares, prev_name or h.issuer)
    return pos


def diff_filings(prev: Filing | None, curr: Filing, params: StrategyParams) -> list[Signal]:
    prev_pos = _aggregate_share_positions(prev) if prev is not None else {}
    curr_pos = _aggregate_share_positions(curr)
    signals: list[Signal] = []

    all_cusips = set(prev_pos) | set(curr_pos)
    for cusip in all_cusips:
        prev_sh, prev_name = prev_pos.get(cusip, (0.0, ""))
        curr_sh, curr_name = curr_pos.get(cusip, (0.0, ""))
        issuer = curr_name or prev_name

        if prev_sh == 0 and curr_sh > 0:
            signals.append(Signal(
                fund=curr.fund, cusip=cusip, issuer=issuer,
                report_date=curr.report_date, filing_date=curr.filing_date,
                kind="entry", prev_shares=prev_sh, new_shares=curr_sh,
            ))
        elif prev_sh > 0 and curr_sh == 0:
            signals.append(Signal(
                fund=curr.fund, cusip=cusip, issuer=issuer,
                report_date=curr.report_date, filing_date=curr.filing_date,
                kind="exit", prev_shares=prev_sh, new_shares=curr_sh,
            ))
        elif prev_sh > 0 and curr_sh > 0:
            change = (curr_sh - prev_sh) / prev_sh
            if change > params.add_threshold:
                signals.append(Signal(
                    fund=curr.fund, cusip=cusip, issuer=issuer,
                    report_date=curr.report_date, filing_date=curr.filing_date,
                    kind="entry", prev_shares=prev_sh, new_shares=curr_sh,
                ))
            elif -change > params.exit_trim_threshold:
                signals.append(Signal(
                    fund=curr.fund, cusip=cusip, issuer=issuer,
                    report_date=curr.report_date, filing_date=curr.filing_date,
                    kind="exit", prev_shares=prev_sh, new_shares=curr_sh,
                ))
    return signals


def signals_for_fund(filings: list[Filing], params: StrategyParams) -> list[Signal]:
    """Walk a fund's chronologically-sorted filings and emit signals."""
    out: list[Signal] = []
    prev: Filing | None = None
    for f in filings:
        out.extend(diff_filings(prev, f, params))
        prev = f
    return out


def all_signals(filings_by_fund: dict[str, list[Filing]], params: StrategyParams) -> list[Signal]:
    out: list[Signal] = []
    for fund, filings in filings_by_fund.items():
        out.extend(signals_for_fund(filings, params))
    out.sort(key=lambda s: (s.filing_date, s.fund, s.cusip))
    return out
