"""
LOT 50: Dynamic Capital Allocator
Auto-scales capital allocation using Kelly Criterion + Risk Parity.
"""

import logging

import numpy as np

logger = logging.getLogger("DynamicCapitalAllocator")

class DynamicCapitalAllocator:
    """
    LOT 50: Automatically adjusts how much capital to deploy based on:
    - Kelly Criterion (per strategy)
    - Risk Parity across assets/strategies
    - Recent performance & regime
    """

    def __init__(self, base_exposure: float = 0.65, max_exposure: float = 0.92, min_exposure: float = 0.25):
        self.base_exposure = base_exposure
        self.max_exposure = max_exposure
        self.min_exposure = min_exposure
        self.current_exposure = base_exposure
        self.strategy_weights = {}
        self.update_count = 0

    def compute_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Simple Kelly fraction"""
        if avg_loss == 0:
            return 0.0
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        kelly = (b * p - q) / b
        return max(0.0, min(kelly, 1.0))

    def compute_risk_parity_weights(self, volatilities: dict[str, float]) -> dict[str, float]:
        """Risk Parity allocation (inverse volatility)"""
        inv_vol = {k: 1.0 / max(v, 0.001) for k, v in volatilities.items()}
        total = sum(inv_vol.values())
        return {k: v / total for k, v in inv_vol.items()}

    def update_allocation(self,
                          strategy_performance: dict[str, dict],
                          asset_volatilities: dict[str, float],
                          regime_id: int) -> dict:
        """
        Main function called periodically.
        Returns recommended exposure and per-strategy weights.
        """
        # 1. Kelly-based sizing per strategy
        kelly_weights = {}
        for strat, perf in strategy_performance.items():
            wr = perf.get("win_rate", 0.5)
            aw = perf.get("avg_win", 0.01)
            al = perf.get("avg_loss", 0.01)
            kelly = self.compute_kelly_fraction(wr, aw, al)
            kelly_weights[strat] = kelly * 0.6  # conservative Kelly

        # 2. Risk Parity across assets
        rp_weights = self.compute_risk_parity_weights(asset_volatilities)

        # 3. Dynamic total exposure
        avg_kelly = np.mean(list(kelly_weights.values())) if kelly_weights else 0.5
        regime_factor = {0: 0.85, 1: 0.65, 2: 1.0, 3: 0.55}.get(regime_id, 0.8)

        target_exposure = self.base_exposure * avg_kelly * regime_factor
        target_exposure = np.clip(target_exposure, self.min_exposure, self.max_exposure)

        self.current_exposure = 0.7 * self.current_exposure + 0.3 * target_exposure

        # Combine weights
        combined_weights = {}
        for strat in kelly_weights:
            combined_weights[strat] = kelly_weights[strat] * self.current_exposure

        self.strategy_weights = combined_weights
        self.update_count += 1

        logger.info(f"LOT 50: Capital exposure updated → {self.current_exposure*100:.1f}% | Regime factor: {regime_factor}")

        return {
            "total_exposure": round(self.current_exposure, 3),
            "strategy_weights": {k: round(v, 3) for k, v in combined_weights.items()},
            "risk_parity_weights": {k: round(v, 3) for k, v in rp_weights.items()}
        }

    def get_current_exposure(self) -> float:
        return self.current_exposure
