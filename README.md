# Biotech-Fund-Trader

A backtest engine for copy-trading the 13F filings of top biotech hedge funds:

- Perceptive Advisors
- Baker Bros. Advisors
- RA Capital Management
- RTW Investments
- OrbiMed Advisors
- Fairmount Funds Management
- Avoro Capital Advisors

## Strategy

For each quarter:

1. Pull the 13F-HR filings for the funds above from SEC EDGAR.
2. Diff against the prior quarter to find **new positions** and **meaningful adds** (share count up >25% QoQ).
3. On the **next trading day after the SEC accepted date**, attempt entry.
4. **Drift filter**: skip if the day-after-filing price has moved more than ±15% from the quarter-end close (the latest reference snap-in we have for the fund's likely cost). Tunable.
5. Allocate `position_size_pct` of NAV to each surviving entry (default 2%).
6. **Exit** when the originating fund either fully closes the position or trims by more than 50% in a subsequent 13F. Open lots are force-closed at the end of the backtest window.
7. Each (fund, ticker) is its own lot — if multiple funds enter the same name, you hold multiple lots.

## Setup

```bash
pip install -r requirements.txt
export EODHD_API_KEY=6825543d2b2f23.48334508
export SEC_USER_AGENT="Your Name your.email@example.com"
# Optional, helps OpenFIGI rate limits:
# export OPENFIGI_API_KEY=...
```

The SEC requires a User-Agent identifying the requester; the default works but
substituting a real contact is friendlier.

## Run

```bash
# Confirm fund CIKs and recent activity:
python -m biotech_backtest verify-funds

# Full backtest over the default window (2014-2024):
python -m biotech_backtest backtest

# Customize:
python -m biotech_backtest backtest \
  --start 2016-01-01 --end 2024-12-31 \
  --drift-threshold 0.15 \
  --position-size-pct 0.02 \
  --benchmark XBI \
  --output-prefix tight_drift
```

Outputs land in `data/output/`:

- `<prefix>_summary.json` — top-line metrics (CAGR, Sharpe, max drawdown, hit rate, alpha vs benchmark)
- `<prefix>_trades.csv` — every closed lot with entry/exit/return/drift
- `<prefix>_skipped.csv` — every skipped signal with the reason
- `<prefix>_nav.csv` — daily NAV
- `<prefix>_benchmark.csv` — benchmark NAV
- `<prefix>_per_fund.csv` — per-fund stats (which fund's signals worked)

## Caveats

- 13F is an equities snapshot 45 days stale. The engine does not see options,
  shorts, or intra-quarter trades; this is a known limit of the data, not a bug.
- CUSIP→ticker mapping uses OpenFIGI's free tier (25 req/min). The first run
  takes a few minutes; subsequent runs hit the cache.
- Survivorship: EODHD covers most delisted US tickers but coverage isn't perfect.
  Skipped entries with reason `no_ticker` or `no_price_after_filing` are logged.
- Fund CIKs in `config.py` are best-effort — run `verify-funds` to sanity-check.
