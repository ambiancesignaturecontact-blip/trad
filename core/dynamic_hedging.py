"""
Dynamic Hedging + Strategy Correlation Matrix (LOT 34)
Calculates correlation between active strategies and dynamically reduces exposure
or suggests hedges when correlations become too high.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("DynamicHedging")

class DynamicHedgingEngine:
    """
    Dynamic Hedging Engine.
    - Builds a correlation matrix between strategies based on recent signals/returns.
    - If average correlation exceeds threshold, reduces overall portfolio exposure.
    """

    def __init__(self, max_strategy_correlation: float = 0.72):
        self.max_strategy_correlation = max_strategy_correlation
        self.strategy_returns_history = {}  # Stores recent signals per strategy

    def update_strategy_returns(self, strategy_name: str, signal: float):
        """Update recent signal history for a strategy"""
        if strategy_name not in self.strategy_returns_history:
            self.strategy_returns_history[strategy_name] = []

        self.strategy_returns_history[strategy_name].append(signal)

        # Keep only last 60 signals
        if len(self.strategy_returns_history[strategy_name]) > 60:
            self.strategy_returns_history[strategy_name].pop(0)

    def calculate_strategy_correlation_matrix(self) -> float:
        """
        Calculates average absolute correlation between all active strategies.
        Returns the average correlation.
        """
        if len(self.strategy_returns_history) < 2:
            return 0.0

        # Build return matrix
        min_len = min(len(v) for v in self.strategy_returns_history.values())
        if min_len < 20:
            return 0.0

        data = {}
        for name, returns in self.strategy_returns_history.items():
            data[name] = returns[-min_len:]

        df = pd.DataFrame(data)
        corr_matrix = df.corr().abs()

        # Average off-diagonal correlation
        mask = ~np.eye(len(corr_matrix), dtype=bool)
        avg_corr = corr_matrix.values[mask].mean()

        return float(avg_corr)

    def get_hedging_multiplier(self, avg_correlation: float) -> float:
        """
        Returns a multiplier to apply to overall exposure.
        If correlation is too high → reduce exposure.
        """
        if avg_correlation > self.max_strategy_correlation:
            reduction = (avg_correlation - self.max_strategy_correlation) / (1 - self.max_strategy_correlation)
            multiplier = max(0.55, 1.0 - reduction * 0.45)
            logger.info(f"Dynamic Hedging: High strategy correlation ({avg_correlation:.2f}) → {multiplier:.2f}x exposure")
            return multiplier
        return 1.0

    def should_hedge(self, avg_correlation: float) -> bool:
        return avg_correlation > (self.max_strategy_correlation * 1.1)
