"""
VISION_FUTUR §1 - L'ORGANISATION AUTONOME (desks + marché interne du capital).

- specialized desks by style/asset family, each with its own realized P&L
- INTERNAL CAPITAL MARKET: a Thompson bandit allocates the risk budget between
  desks - good desks get more capital, bad desks lose it (healthy competition)
- daily mean-variance/CVaR-style allocation between desks
- crisis tightening: when stress correlation is high, total exposure shrinks
"""
import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger("Organization")

# strategy family -> desk
DESK_MAP = {
    "Trend Following": "macro",
    "Momentum": "crypto_momentum",
    "Cross-Sectional Momentum": "crypto_momentum",
    "Mean Reversion": "meanrev",
    "Volatility Breakout": "vol",
    "Market Making": "mm",
    "Statistical Arbitrage": "statarb",
    "Inter-Exchange Arbitrage": "statarb",
    "Grid Trading": "mm",
    "Scalping": "scalp",
    "Carry": "carry",
    "Multi-Timeframe": "macro",
    "META_MODEL": "macro",
    "COPY_MIRROR": "macro",
    "WEBHOOK": "macro",
    "POSITION_PROTECTION_STOP_LOSS": "risk",
    "POSITION_PROTECTION_TAKE_PROFIT": "risk",
}


class Desk:
    def __init__(self, name: str):
        self.name = name
        self.pnl: List[float] = []
        self.orders = 0
        self.capital_share = 1.0

    def record(self, pnl: float, notional: float = 0.0):
        self.pnl.append(float(pnl))
        self.orders += 1
        if len(self.pnl) > 200:
            self.pnl = self.pnl[-200:]

    def realized_vol(self) -> float:
        return float(np.std(self.pnl[-40:])) if len(self.pnl) >= 5 else 0.0

    def recent_return(self) -> float:
        return float(np.sum(self.pnl[-20:])) if self.pnl else 0.0

    def status(self) -> dict:
        return {
            "name": self.name, "orders": self.orders,
            "pnl_recent": round(self.recent_return(), 4),
            "vol": round(self.realized_vol(), 6),
            "capital_share": round(self.capital_share, 4),
        }


class Organization:
    """The bot as a mini fund: desks compete for a risk budget (Thompson bandit)."""

    DESKS = ["crypto_momentum", "macro", "meanrev", "vol", "mm", "statarb", "scalp", "carry"]

    def __init__(self, state: dict):
        self.state = state
        self.desks = {name: Desk(name) for name in self.DESKS}
        self.state.setdefault("desk_allocations", {n: 1.0 / len(self.DESKS) for n in self.DESKS})
        self.state.setdefault("desk_pnl", {})

    def desk_of(self, strategy: str) -> str:
        return DESK_MAP.get(strategy, "macro")

    def record_trade(self, strategy: str, pnl: float, notional: float = 0.0):
        desk = self.desk_of(strategy)
        self.desks[desk].record(pnl, notional)
        self.state["desk_pnl"][desk] = self.desks[desk].recent_return()

    def reallocate(self, stress_correlation: float = 0.5, min_share: float = 0.03,
                   max_share: float = 0.35) -> Dict[str, float]:
        """
        INTERNAL CAPITAL MARKET: Thompson sampling over desk recent P&L.
        Good desks get more capital; bad desks lose it. Crisis dampening.
        """
        alpha = np.array([1.0 + max(self.desks[d].recent_return(), -0.5) * 4.0 for d in self.DESKS])
        beta = np.array([1.0 + max(-self.desks[d].recent_return(), 0.0) * 4.0 for d in self.DESKS])
        samples = np.random.beta(alpha, beta)
        weights = samples / samples.sum()
        weights = np.clip(weights, min_share, max_share)
        weights = weights / weights.sum()
        # crisis tightening: reduce overall risk budget
        crisis_factor = 1.0 if stress_correlation <= 0.6 else (1.0 - (stress_correlation - 0.6) * 0.6)
        alloc = {d: round(float(w), 4) for d, w in zip(self.DESKS, weights)}
        self.state["desk_allocations"] = alloc
        self.state["desk_crisis_factor"] = round(float(crisis_factor), 4)
        for d, w in zip(self.DESKS, weights):
            self.desks[d].capital_share = float(w)
        return alloc

    def confidence_factor(self, symbol: str) -> float:
        """Size multiplier from the desk that owns this symbol's strategy context."""
        # uses the market-state crisis factor as the portfolio-level scaler
        return float(self.state.get("desk_crisis_factor", 1.0))

    def status(self) -> dict:
        return {
            "desks": {d: self.desks[d].status() for d in self.DESKS},
            "allocations": self.state.get("desk_allocations", {}),
            "crisis_factor": self.state.get("desk_crisis_factor", 1.0),
        }
