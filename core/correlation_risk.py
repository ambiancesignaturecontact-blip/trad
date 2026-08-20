"""
Correlation Risk Engine (LOT 26)
Réduit automatiquement l'exposition si la corrélation entre positions devient trop élevée.
"""
import logging

import numpy as np

logger = logging.getLogger("CorrelationRisk")

class CorrelationRiskEngine:
    """
    Correlation Risk Engine.
    Calcule la corrélation réelle entre les positions et réduit le sizing si trop élevée.
    """

    def __init__(self, max_correlation: float = 0.75):
        self.max_correlation = max_correlation

    def calculate_position_correlation(self, positions: list[dict],
                                       price_history: dict[str, np.ndarray]) -> float:
        """
        Calcule la corrélation moyenne entre les positions actuelles.
        """
        if len(positions) < 2:
            return 0.0

        returns = {}
        for pos in positions:
            sym = pos['symbol']
            if sym in price_history and len(price_history[sym]) > 20:
                returns[sym] = np.diff(np.log(price_history[sym][-30:]))

        if len(returns) < 2:
            return 0.0

        # Matrice de corrélation
        symbols = list(returns.keys())
        corr_matrix = np.corrcoef([returns[s] for s in symbols])

        # Moyenne des corrélations hors diagonale
        mask = ~np.eye(len(symbols), dtype=bool)
        avg_corr = np.abs(corr_matrix[mask]).mean()

        return float(avg_corr)

    def get_correlation_adjustment(self, avg_correlation: float) -> float:
        """
        Retourne un multiplicateur de taille (0.5 à 1.0).
        """
        if avg_correlation > self.max_correlation:
            reduction = (avg_correlation - self.max_correlation) / (1 - self.max_correlation)
            multiplier = max(0.5, 1.0 - reduction * 0.5)
            logger.info(f"Correlation Risk: High correlation ({avg_correlation:.2f}) → {multiplier:.2f}x sizing")
            return multiplier
        return 1.0
