"""
VISION §5 - EXÉCUTER: the last mile becomes intelligent.

- learned execution-style agent: a bandit that learns which style
  (market/limit/twap) yields the lowest slippage per (symbol, vol-regime)
- tradability filter: if realized slippage for a symbol/strategy is too high,
  the bot reduces size or abstains (capacity awareness)
- per-strategy execution attribution (slippage by strategy)
"""
import logging
import time
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("ExecutionAgent")

STYLES = ["market", "limit", "twap"]


class ExecutionStyleBandit:
    """
    Epsilon-greedy bandit over execution styles, fed by real slippage data.

    LOT 3 (PDF Pilier H-2) : epsilon ADAPTATIF décroissant — le bandit
    explore au début (données réelles rares) puis exploite de plus en plus
    (plancher 2 %). Jamais d'exploration excessive sur l'argent réel
    (mentalité n°2 : chaque trade a un coût).
    """

    def __init__(self, epsilon: float = 0.15, min_epsilon: float = 0.02,
                 decay_per_obs: float = 0.01):
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_per_obs = decay_per_obs
        self._n_obs = 0
        self.counts: Dict[str, Dict[str, int]] = {}
        self.sums: Dict[str, Dict[str, float]] = {}

    def _key(self, symbol: str, vol_regime: str) -> str:
        return f"{symbol}|{vol_regime}"

    def choose_style(self, symbol: str, vol_regime: str, spread_bps: float, urgency: float) -> str:
        key = self._key(symbol, vol_regime)
        c = self.counts.setdefault(key, {s: 0 for s in STYLES})
        self._n_obs += 1
        # epsilon décroît avec le nombre d'observations réelles
        self.epsilon = max(self.min_epsilon,
                           self.epsilon - self.decay_per_obs * max(0, self._n_obs - 20))
        if sum(c.values()) < 5 or np.random.random() < self.epsilon:
            # explore, but respect obvious constraints
            if urgency < 0.3 and spread_bps > 15:
                return "limit"
            if urgency > 0.7:
                return "market"
            return np.random.choice(STYLES)
        # exploit: style with lowest average slippage
        avg = {s: self.sums[key].get(s, 0.0) / max(c[s], 1) for s in STYLES}
        return min(STYLES, key=lambda s: avg[s])

    def observe(self, symbol: str, vol_regime: str, style: str, slippage_bps: float) -> None:
        key = self._key(symbol, vol_regime)
        self.counts.setdefault(key, {s: 0 for s in STYLES})[style] += 1
        self.sums.setdefault(key, {s: 0.0 for s in STYLES})[style] += max(slippage_bps, 0.0)

    def status(self) -> dict:
        return {
            "samples": {k: sum(v.values()) for k, v in self.counts.items()},
            "avg_slippage": {
                k: {s: round(self.sums[k].get(s, 0.0) / max(v.get(s, 1), 1), 2) for s in STYLES}
                for k, v in self.counts.items()
            },
        }


def tradability_factor(avg_slippage_bps: float, max_slippage_bps: float = 15.0,
                       min_factor: float = 0.3) -> float:
    """
    VISION §5b: capacity/tradability filter.
    Realized slippage above the threshold reduces the position size (or kills it).
    """
    if avg_slippage_bps <= 0:
        return 1.0
    if avg_slippage_bps >= max_slippage_bps:
        return min_factor
    return float(1.0 - (avg_slippage_bps / max_slippage_bps) * (1.0 - min_factor))


class StrategyExecutionAttribution:
    """VISION §5c: slippage by strategy - tells signal vs execution apart."""

    def __init__(self):
        self.by_strategy: Dict[str, Dict] = {}

    def record(self, strategy: str, slippage_bps: float, style: str) -> None:
        s = self.by_strategy.setdefault(strategy, {"n": 0, "sum_bps": 0.0, "last": 0.0})
        s["n"] += 1
        s["sum_bps"] += slippage_bps
        s["last"] = slippage_bps
        s["avg_bps"] = round(s["sum_bps"] / s["n"], 2)

    def report(self) -> dict:
        return {k: {kk: vv for kk, vv in v.items()} for k, v in self.by_strategy.items()}
