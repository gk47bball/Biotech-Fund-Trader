from dataclasses import dataclass, field
from typing import List, Optional

# 30 liquid biotech names with price history back to 2019
BIOTECH_UNIVERSE: List[str] = [
    "AMGN", "GILD", "BIIB", "REGN", "VRTX",
    "ILMN", "BMRN", "ALNY", "INCY", "IONS",
    "NBIX", "SRPT", "UTHR", "EXEL", "JAZZ",
    "RARE", "HALO", "ARWR", "BCRX", "FOLD",
    "ITCI", "INSM", "PTCT", "MRNA", "RCKT",
    "SAGE", "SMMT", "VCEL", "ZLAB", "ACAD",
]


@dataclass
class BacktestConfig:
    # ── Time period ──────────────────────────────────────────────────────────
    start_date: str = "2019-01-01"
    end_date: str = "2024-12-31"

    # ── Capital ──────────────────────────────────────────────────────────────
    initial_capital: float = 10_000_000.0   # $10 M fund

    # ── Transaction costs ────────────────────────────────────────────────────
    commission_pct: float = 0.001           # 10 bps commission
    slippage_pct: float = 0.0005            # 5 bps slippage

    # ── Portfolio limits ─────────────────────────────────────────────────────
    max_position_pct: float = 0.08          # 8 % cap per position
    min_position_pct: float = 0.01          # 1 % floor (below → skip)
    max_positions: int = 20
    top_n: int = 15                         # long-list trimmed to top-N signals

    # ── Risk management ──────────────────────────────────────────────────────
    stop_loss_pct: float = 0.25             # 25 % trailing stop per position
    portfolio_stop_pct: float = 0.20        # 20 % max portfolio drawdown

    # ── Signal parameters ────────────────────────────────────────────────────
    momentum_lookback: int = 252            # 12-month return window
    momentum_skip: int = 21                 # skip most-recent month (reversal)
    reversion_lookback: int = 20            # 20-day mean-reversion window
    signal_blend: float = 0.70             # weight on momentum (1-blend → mean-rev)

    # ── Rebalancing ──────────────────────────────────────────────────────────
    rebalance_frequency: str = "monthly"    # "daily" | "weekly" | "monthly"

    # ── Benchmark ────────────────────────────────────────────────────────────
    benchmark: str = "XBI"

    # ── Risk-free rate (annualised) ──────────────────────────────────────────
    risk_free_rate: float = 0.045

    # ── Universe ─────────────────────────────────────────────────────────────
    universe: List[str] = field(default_factory=lambda: list(BIOTECH_UNIVERSE))
