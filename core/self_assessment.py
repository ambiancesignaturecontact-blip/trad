"""
VISION §7 - SE CONNAÎTRE: honest self-assessment.

- backtest<->live divergence: simulated vs realized slippage gap -> when the
  simulation lies, the bot shrinks itself
- meta-attribution: which top-5 reasons actually predicted winning trades
  (weekly analysis of the decision journal) -> confidence recalibration
- honesty component for the health score
"""
import logging
import time
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("SelfAssessment")


def simulation_divergence(modeled_slippage_bps: float, realized_slippage_bps: float) -> float:
    """
    VISION §7a: divergence = (realized - modeled) / max(modeled, 1).
    >0 means the simulation is optimistic (real slippage higher than modeled).
    """
    if modeled_slippage_bps <= 0:
        return 0.0
    return float((realized_slippage_bps - modeled_slippage_bps) / modeled_slippage_bps)


def honesty_factor(divergence: float, max_divergence: float = 1.0) -> float:
    """
    VISION §7a: factor applied to sizes when the simulation is lying.
    divergence 0 -> 1.0 (trust); divergence >= max -> min_factor.
    """
    if divergence <= 0:
        return 1.0
    return float(max(0.3, 1.0 - divergence / max_divergence))


def meta_attribution(decision_log: List[dict]) -> Dict[str, Dict]:
    """
    VISION §7b: analyze logged decisions (top-5 reasons + realized PnL) and
    compute per-reason effectiveness (win rate + avg contribution).
    decision_log: [{reasons: [str], pnl: float}]
    """
    stats: Dict[str, Dict] = {}
    for d in decision_log:
        pnl = float(d.get("pnl", 0.0))
        for reason in d.get("reasons", []):
            s = stats.setdefault(reason, {"n": 0, "wins": 0, "sum_pnl": 0.0})
            s["n"] += 1
            s["sum_pnl"] += pnl
            if pnl > 0:
                s["wins"] += 1
    for r, s in stats.items():
        s["win_rate"] = round(s["wins"] / max(s["n"], 1), 3)
        s["avg_pnl"] = round(s["sum_pnl"] / max(s["n"], 1), 4)
    return stats




def reason_weight_from_attribution(attribution: Dict[str, Dict],
                                   base_weight: float = 1.0,
                                   min_weight: float = 0.3,
                                   min_samples: int = 5) -> Dict[str, float]:
    """
    LOT 7 (PDF Pilier K) : boucle la méta-attribution (quelles raisons
    gagnent ?) vers une RÉDUCTION AUTOMATIQUE du poids des mauvaises raisons.

    Une raison avec un win rate < 50% et une contribution négative sur un
    échantillon suffisant voit son poids réduit (jusqu'à min_weight). Les
    raisons sans échantillon restent à base_weight (jamais de pénalité sans
    preuve — mentalité n°20).

    Retourne {reason: weight_factor}.
    """
    weights: Dict[str, float] = {}
    for reason, stats in attribution.items():
        n = int(stats.get("n", 0))
        if n < min_samples:
            weights[reason] = base_weight
            continue
        wr = float(stats.get("win_rate", 0.5))
        avg_pnl = float(stats.get("avg_pnl", 0.0))
        if wr < 0.5 and avg_pnl <= 0:
            # mauvaise raison prouvée -> réduction (proportionnelle à la preuve)
            factor = base_weight * (0.5 + 0.5 * (wr / 0.5))
            weights[reason] = max(min_weight, factor)
        elif wr >= 0.55:
            # bonne raison -> léger bonus (jamais plus de 1.2x)
            weights[reason] = min(1.2, base_weight * 1.1)
        else:
            weights[reason] = base_weight
    return weights

def health_honesty_component(divergence: float, base_score: int) -> int:
    """
    VISION §7c: reduce the health score when the simulation is lying.
    """
    if divergence <= 0.15:
        return base_score
    penalty = int(min(20, (divergence - 0.15) * 20))
    return max(0, base_score - penalty)
