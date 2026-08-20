"""
Regime-Switching Allocation (LOT 22)
Ajuste dynamiquement les poids des stratégies selon le régime de marché détecté.
"""
import logging

import numpy as np

logger = logging.getLogger("RegimeSwitching")

class RegimeSwitchingAllocator:
    """
    Regime-Switching Allocation.
    Change la dominance et les poids des stratégies selon le régime HMM.
    """

    def __init__(self):
        # Poids de base par régime (peuvent être ajustés)
        self.regime_weights = {
            0: {  # Bull Trend (Low Vol)
                "Trend Following": 0.45,
                "Momentum": 0.25,
                "Volatility Breakout": 0.15,
                "Mean Reversion": 0.05,
                "Scalping": 0.05,
                "Grid Trading": 0.05
            },
            1: {  # Bear Trend (High Vol)
                "Mean Reversion": 0.35,
                "Scalping": 0.25,
                "Trend Following": 0.15,
                "Volatility Breakout": 0.15,
                "Momentum": 0.05,
                "Grid Trading": 0.05
            },
            2: {  # Mean-Reverting Range
                "Mean Reversion": 0.40,
                "Grid Trading": 0.25,
                "Scalping": 0.15,
                "Trend Following": 0.10,
                "Momentum": 0.05,
                "Volatility Breakout": 0.05
            },
            3: {  # Erratic High Volatility
                "Scalping": 0.35,
                "Volatility Breakout": 0.25,
                "Mean Reversion": 0.20,
                "Trend Following": 0.10,
                "Momentum": 0.05,
                "Grid Trading": 0.05
            }
        }

    def get_regime_weights(self, regime_id: int) -> dict[str, float]:
        """Retourne les poids normalisés pour un régime donné"""
        weights = self.regime_weights.get(regime_id, self.regime_weights[2])

        # Normalisation
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def apply_regime_switching(self, base_signal: float, regime_id: int,
                               strategy_name: str) -> float:
        """
        Ajuste le signal d'une stratégie selon le régime.
        """
        weights = self.get_regime_weights(regime_id)
        regime_weight = weights.get(strategy_name, 0.1)

        # Le signal est boosté ou réduit selon le poids du régime
        adjusted = base_signal * (0.7 + regime_weight * 1.5)
        return np.clip(adjusted, -1.0, 1.0)
