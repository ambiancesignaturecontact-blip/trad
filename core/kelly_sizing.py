"""
Dynamic Kelly Fraction Sizing (LOT 27)
Kelly fractionnel dynamique **par stratégie + par régime de marché**.
"""
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger("KellySizing")

class DynamicKellySizer:
    """
    Kelly Criterion Fractionnel Dynamique (amélioré LOT 27).
    Prend en compte à la fois les performances récentes **et** le régime de marché.
    """
    
    def __init__(self, base_fraction: float = 0.15):
        self.base_fraction = base_fraction
        self.strategy_kelly = {}
        
        # Kelly base par régime (plus agressif en bull, plus conservateur en bear/high vol)
        self.regime_base = {
            0: 0.18,   # Bull Trend (Low Vol)     → plus agressif
            1: 0.10,   # Bear Trend (High Vol)    → très conservateur
            2: 0.14,   # Mean-Reverting Range     → neutre
            3: 0.08    # Erratic High Volatility  → très conservateur
        }

    def calculate_kelly_fraction(self, strategy_name: str, recent_scores: list, 
                                 regime_id: int = 2) -> float:
        """
        Calcule le Kelly fractionnel en tenant compte du régime.
        """
        if not recent_scores or len(recent_scores) < 10:
            return self.regime_base.get(regime_id, self.base_fraction)

        wins = sum(1 for s in recent_scores if s > 0)
        total = len(recent_scores)
        p = wins / total

        R = 1.5
        kelly = (p * R - (1 - p)) / R
        kelly = max(0.02, min(kelly, 0.40))

        # Ajustement par régime
        regime_mult = self.regime_base.get(regime_id, 0.12) / 0.12
        fractional_kelly = kelly * 0.25 * regime_mult

        self.strategy_kelly[strategy_name] = fractional_kelly
        return fractional_kelly

    def get_position_size_multiplier(self, strategy_name: str, recent_scores: list, 
                                     regime_id: int = 2) -> float:
        """Retourne le multiplicateur de taille (Kelly + Régime)"""
        kelly = self.calculate_kelly_fraction(strategy_name, recent_scores, regime_id)
        base = self.regime_base.get(regime_id, self.base_fraction)
        return kelly / base

    def get_all_kelly_fractions(self) -> Dict:
        return self.strategy_kelly.copy()