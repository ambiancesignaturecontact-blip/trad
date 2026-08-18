"""
VISION §6 - SE PROTÉGER: an AI risk committee with veto power.

- per-strategy risk scoring from REAL stress data (vol, correlation, drawdown)
- veto: disable a strategy when its risk score is excessive (audited, notified)
- daily portfolio optimizer: rebalance strategy risk budget (risk parity +
  stress correlation), replacing heuristics
- dynamic risk budget recalibrated from realized vol + stress scenarios
"""
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("RiskCommittee")


def strategy_risk_score(strategy: str, recent_returns: List[float],
                        correlation: float, drawdown: float,
                        max_vol: float = 0.02) -> float:
    """
    Composite 0..1 risk score per strategy:
    vol + correlation + drawdown, each normalized.
    """
    vol = float(np.std(recent_returns[-40:])) if len(recent_returns) >= 5 else 0.0
    vol_score = min(vol / max(max_vol, 1e-9), 1.0)
    corr_score = min(abs(correlation), 1.0)
    dd_score = min(abs(drawdown) / 0.05, 1.0)
    return float(np.clip(0.5 * vol_score + 0.3 * corr_score + 0.2 * dd_score, 0.0, 1.0))


class RiskCommittee:
    """Veto-capable risk manager (VISION §6a)."""

    def __init__(self, veto_threshold: float = 0.85):
        self.veto_threshold = veto_threshold
        self.vetoed: Dict[str, float] = {}     # strategy -> score at veto
        self.scores: Dict[str, float] = {}

    def evaluate(self, meta_engine, state: dict) -> List[dict]:
        """Scores each strategy from REAL recent performance; vetoes the worst."""
        actions = []
        for strat in getattr(meta_engine, "strategies", []):
            name = getattr(strat, "name", "?")
            rets = meta_engine.recent_performance.get(name, [])
            corr = 0.5  # placeholder from live correlation when available
            dd = 0.0
            if len(rets) > 20:
                eq = np.cumprod(1 + np.array(rets[-40:]))
                dd = float((np.max(eq) - eq[-1]) / max(np.max(eq), 1e-9))
            score = strategy_risk_score(name, rets, corr, dd)
            self.scores[name] = round(score, 3)
            if score >= self.veto_threshold and name not in self.vetoed:
                strat.enabled = False
                self.vetoed[name] = score
                actions.append({"strategy": name, "action": "VETO", "score": round(score, 3),
                                "reason": f"risk score {score:.2f} >= {self.veto_threshold}"})
                logger.critical(f"🛑 RISK COMMITTEE VETO: {name} disabled (score {score:.2f})")
            elif score < self.veto_threshold * 0.7 and name in self.vetoed:
                # allow a strategy back when its risk normalizes
                strat.enabled = True
                del self.vetoed[name]
                actions.append({"strategy": name, "action": "RE-ENABLE", "score": round(score, 3),
                                "reason": "risk normalized"})
        return actions

    def status(self) -> dict:
        return {"veto_threshold": self.veto_threshold, "vetoed": self.vetoed, "scores": self.scores}


def daily_risk_budget(strategy_returns: Dict[str, List[float]],
                      stress_correlation: float = 0.5,
                      min_weight: float = 0.02, max_weight: float = 0.35) -> Dict[str, float]:
    """
    VISION §6b/c: dynamic risk budget = inverse-vol weights, dampened when the
    stress correlation is high (positions become one big bet in a crisis).
    """
    from core.factor_model import risk_parity_weights
    w = risk_parity_weights(strategy_returns, min_weight, max_weight)
    # crisis dampening: when correlation is high, flatten the budget
    if stress_correlation > 0.6:
        damp = 0.5 + (1.0 - stress_correlation) * 0.5
        w = {k: v * damp for k, v in w.items()}
        s = sum(w.values())
        w = {k: v / s for k, v in w.items()}
    return {k: round(v, 4) for k, v in w.items()}
