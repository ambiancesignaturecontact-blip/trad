"""
Regime-Switching Allocation (LOT 22 + Axe 2 mission intelligence)
Ajuste dynamiquement les poids des stratégies selon le régime de marché détecté.
Axe 2 : les poids statiques servent d'A PRIORI ; la performance RÉELLE par
(régime, stratégie) les ajuste en ligne (non-stationnarité) — mais de façon
BORNÉE : jamais plus de ±30 % d'écart au prior, pour éviter la sur-réaction
au bruit (leçon des systèmes morts en production par sur-optimisation).
"""
import logging

import numpy as np

logger = logging.getLogger("RegimeSwitching")

# Ajustement max des poids par rapport à l'a priori statique (borné).
# 0.30 = une stratégie peut gagner ou perdre au plus 30 % de son poids de
# base à cause de la performance récente — le reste est le jugement de régime
# statique (économie de marché), qui ne bouge pas avec le bruit à court terme.
MAX_ADAPTIVE_SHIFT = 0.30
# Lissage EMA de la performance par (régime, stratégie) : 0.2 = la mémoire
# dure ~5 observations récentes. Assez lent pour ne pas réagir à un trade,
# assez rapide pour suivre un changement de régime en quelques jours.
PERF_ALPHA = 0.20


class RegimeSwitchingAllocator:
    """
    Regime-Switching Allocation.
    Change la dominance et les poids des stratégies selon le régime HMM,
    ajustés en ligne par la performance réelle (borné).
    """

    def __init__(self):
        # A PRIORI : poids de base par régime (économie de marché, statique)
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
        # POSTERIOR : performance lissée par (régime, stratégie) — apprise en
        # ligne à partir du PnL réel clôturé (aucune donnée simulée).
        self.regime_perf: dict[int, dict[str, float]] = {}
        # Compteur d'observations par (régime, stratégie) — on n'ajuste que
        # si on a vu la stratégie assez souvent dans ce régime.
        self.regime_perf_count: dict[int, dict[str, int]] = {}
        self.min_observations = 5

    def update_regime_performance(self, regime_id: int, strategy: str,
                                  pnl_pct: float) -> None:
        """
        Axe 2 : met à jour la performance lissée (EMA) de (régime, stratégie)
        avec le PnL RÉEL d'un trade clôturé. Appelé par la boucle live à
        chaque trade clôturé (jamais de donnée simulée).
        """
        r = int(regime_id)
        self.regime_perf.setdefault(r, {})
        self.regime_perf_count.setdefault(r, {})
        prev = self.regime_perf[r].get(strategy)
        # EMA : alpha sur la nouvelle observation, sinon initialisation
        new = (PERF_ALPHA * float(pnl_pct) +
               (1.0 - PERF_ALPHA) * prev) if prev is not None else float(pnl_pct)
        self.regime_perf[r][strategy] = new
        self.regime_perf_count[r][strategy] = self.regime_perf_count[r].get(strategy, 0) + 1

    def _adaptive_shift(self, regime_id: int, strategy: str) -> float:
        """
        Multiplicateur adaptatif borné par la performance réelle.
        Retourne un facteur dans [1-MAX_SHIFT, 1+MAX_SHIFT].
        Performance positive -> poids boosté (au plus +30 %) ;
        négative -> réduit (au plus -30 %) ; pas d'échantillon -> neutre 1.0.
        """
        r = int(regime_id)
        if self.regime_perf_count.get(r, {}).get(strategy, 0) < self.min_observations:
            return 1.0
        perf = self.regime_perf.get(r, {}).get(strategy, 0.0)
        # pnl de 1% -> +~10% de poids ; pnl de -1% -> -~10% (borné ±30%)
        shift = np.clip(perf * 10.0, -MAX_ADAPTIVE_SHIFT, MAX_ADAPTIVE_SHIFT)
        return 1.0 + float(shift)

    def get_regime_weights(self, regime_id: int) -> dict[str, float]:
        """Poids normalisés pour un régime : a priori × shift adaptatif (borné)."""
        prior = self.regime_weights.get(regime_id, self.regime_weights[2])
        adjusted = {}
        for strat, w in prior.items():
            adjusted[strat] = w * self._adaptive_shift(regime_id, strat)
        # Normalisation
        total = sum(adjusted.values())
        return {k: v / total for k, v in adjusted.items()}

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
