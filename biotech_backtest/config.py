"""Configuration: target funds, strategy parameters, paths."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "output"
FILINGS_CACHE = CACHE_DIR / "filings"
PRICES_CACHE = CACHE_DIR / "prices"
CUSIP_CACHE = CACHE_DIR / "cusip_map.json"

for d in (DATA_DIR, CACHE_DIR, OUTPUT_DIR, FILINGS_CACHE, PRICES_CACHE):
    d.mkdir(parents=True, exist_ok=True)


# Best-effort CIKs for the requested biotech funds. Verify with
# `python -m biotech_backtest.cli verify-funds` before relying on them.
FUNDS: dict[str, str] = {
    "Perceptive Advisors":     "0001224962",
    "Baker Bros. Advisors":    "0001263508",
    "RA Capital Management":   "0001346824",
    "RTW Investments":         "0001493215",
    "OrbiMed Advisors":        "0001055951",
    "Fairmount Funds Mgmt":    "0001825868",
    "Avoro Capital Advisors":  "0001633313",
}


@dataclass
class StrategyParams:
    # Entry filter: skip if |price_at_entry / ref_price - 1| > drift_threshold
    drift_threshold: float = 0.15
    # Adds: copy if shares grew by more than this fraction QoQ
    add_threshold: float = 0.25
    # Exit if fund trims by more than this fraction (1.0 = full exit only)
    exit_trim_threshold: float = 0.50
    # Position sizing as fraction of NAV at entry
    position_size_pct: float = 0.02
    # Hard cap on simultaneously held positions; 0 = no cap
    max_positions: int = 0
    # Starting capital for the simulated portfolio
    starting_capital: float = 1_000_000.0
    # Backtest window
    start_date: str = "2014-01-01"
    end_date: str = "2024-12-31"
    # Benchmark ticker
    benchmark: str = "SPY"


@dataclass
class APIConfig:
    eodhd_api_key: str = field(default_factory=lambda: os.environ.get("EODHD_API_KEY", ""))
    sec_user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "SEC_USER_AGENT",
            "Biotech-Fund-Trader research@example.com",
        )
    )
    openfigi_api_key: str = field(default_factory=lambda: os.environ.get("OPENFIGI_API_KEY", ""))
