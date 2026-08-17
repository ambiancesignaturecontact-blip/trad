"""
LOT 57: Advanced Almgren-Chriss Market Impact & Liquidity Model
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger("AlmgrenChrissAdvanced")

class AdvancedAlmgrenChrissModel:
    """
    LOT 57: Advanced Almgren-Chriss model with:
    - Temporary + Permanent market impact
    - Optimal execution trajectory
    - Liquidity-adjusted parameters
    - Risk aversion
    """

    def __init__(self, 
                 gamma: float = 0.1,      # Permanent impact coefficient
                 eta: float = 0.05,       # Temporary impact coefficient
                 lambda_risk: float = 0.5):  # Risk aversion
        self.gamma = gamma
        self.eta = eta
        self.lambda_risk = lambda_risk

    def estimate_market_impact(self, 
                               symbol: str,
                               order_size: float,
                               avg_daily_volume: float,
                               volatility: float,
                               current_price: float,
                               side: str = "BUY") -> Dict:
        """
        Estimate total execution cost including market impact.
        """
        # Participation rate
        participation = order_size / avg_daily_volume if avg_daily_volume > 0 else 0.1

        # Temporary impact (linear in participation)
        temp_impact_bps = self.eta * participation * 10000

        # Permanent impact (linear in order size relative to ADV)
        perm_impact_bps = self.gamma * participation * 10000

        # Total expected slippage in basis points
        total_slippage_bps = temp_impact_bps + perm_impact_bps

        # Risk term (Almgren-Chriss style)
        risk_term = self.lambda_risk * volatility * np.sqrt(participation)

        total_cost_bps = total_slippage_bps + risk_term * 10000

        # Estimated execution price
        if side.upper() == "BUY":
            exec_price = current_price * (1 + total_cost_bps / 10000)
        else:
            exec_price = current_price * (1 - total_cost_bps / 10000)

        result = {
            "symbol": symbol,
            "order_size": order_size,
            "participation_rate": round(participation, 4),
            "temporary_impact_bps": round(temp_impact_bps, 2),
            "permanent_impact_bps": round(perm_impact_bps, 2),
            "risk_term_bps": round(risk_term * 10000, 2),
            "total_slippage_bps": round(total_cost_bps, 2),
            "estimated_execution_price": round(exec_price, 4),
            "side": side
        }

        logger.info(f"LOT 57: {symbol} | Participation: {participation:.2%} | Total Impact: {total_cost_bps:.1f} bps")
        return result

    def optimal_execution_trajectory(self, 
                                     total_shares: float,
                                     horizon_steps: int = 10,
                                     urgency: float = 0.5) -> np.ndarray:
        """
        Returns the optimal trading trajectory (Almgren-Chriss optimal schedule).
        urgency: 0 = slow, 1 = aggressive
        """
        # Simplified Almgren-Chriss optimal trajectory
        t = np.linspace(0, 1, horizon_steps)
        trajectory = total_shares * (1 - (1 - t) ** (1 + urgency * 2))

        return np.diff(np.concatenate([[0], trajectory]))

    def get_liquidity_score(self, 
                            avg_daily_volume: float, 
                            spread_bps: float,
                            volatility: float) -> float:
        """Returns a normalized liquidity score (0 = illiquid, 1 = very liquid)"""
        vol_factor = 1 / (1 + volatility * 50)
        spread_factor = 1 / (1 + spread_bps / 5)
        adv_factor = min(avg_daily_volume / 5_000_000, 1.0)  # Normalize around 5M ADV

        score = (adv_factor * 0.5) + (spread_factor * 0.3) + (vol_factor * 0.2)
        return float(np.clip(score, 0.05, 1.0))
