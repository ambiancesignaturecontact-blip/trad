"""
Live CVaR-Constrained Portfolio Optimization (LOT 33)
Constrains new trades so that the portfolio Conditional Value at Risk stays within limits.
"""
import logging

import numpy as np
import pandas as pd

from core.config import settings

logger = logging.getLogger("CVaROptimizer")

class CVaRPortfolioOptimizer:
    """
    Live CVaR Portfolio Optimizer.
    Uses the existing covariance engine to compute portfolio CVaR
    and limits position sizing accordingly.
    """

    def __init__(self, covariance_engine,
                 cvar_limit_pct: float = None):
        self.covariance_engine = covariance_engine
        # P1-8 (audit §3) : défaut branché sur core/config.py (config.yaml)
        if cvar_limit_pct is None:
            cvar_limit_pct = settings.get_float("execution", "cvar_limit_pct", 0.025)
        self.cvar_limit_pct = cvar_limit_pct  # e.g. 2.5% daily CVaR limit

    def calculate_current_cvar(self, positions: list[dict],
                               returns_dict: dict[str, np.ndarray],
                               corr_matrix: pd.DataFrame) -> float:
        """
        Calculates current portfolio CVaR using the covariance engine.
        """
        try:
            var_metrics = self.covariance_engine.calculate_portfolio_var_cvar(
                active_positions=positions,
                corr_matrix=corr_matrix,
                assets_returns_dict=returns_dict
            )
            return float(var_metrics.get("portfolio_cvar_pct", 0.02))
        except Exception as e:
            logger.warning(f"CVaR calculation failed: {e}")
            return 0.02

    def get_cvar_constrained_size(self, current_cvar: float,
                                  proposed_size: float,
                                  symbol: str) -> float:
        """
        Returns a constrained position size so that adding this trade
        does not push portfolio CVaR above the limit.
        """
        if current_cvar >= self.cvar_limit_pct:
            # Already at limit → strongly reduce size
            constrained = proposed_size * 0.25
            logger.info(f"CVaR limit reached ({current_cvar:.2%}). Reducing {symbol} size to 25%.")
            return constrained

        # Room left
        headroom = (self.cvar_limit_pct - current_cvar) / self.cvar_limit_pct
        multiplier = max(0.3, min(1.0, headroom * 1.5))

        constrained_size = proposed_size * multiplier

        if multiplier < 0.9:
            logger.info(f"CVaR constraint applied on {symbol}: {multiplier:.2f}x size (current CVaR={current_cvar:.2%})")

        return constrained_size

    def should_block_trade(self, current_cvar: float) -> bool:
        """Returns True if we should completely block new trades."""
        return current_cvar > (self.cvar_limit_pct * 1.15)
