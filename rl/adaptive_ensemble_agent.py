"""
LOT 46: Adaptive Ensemble Agent with Online Model Selection
"""
import logging
from typing import Any

import numpy as np

logger = logging.getLogger("AdaptiveEnsembleAgent")

class AdaptiveEnsembleAgent:
    def __init__(self, model_selector, base_ensemble=None):
        self.selector = model_selector
        self.base_ensemble = base_ensemble
        self.decision_count = 0

    def decide(self, symbol: str, market_data: dict,
               model_signals: dict[str, float]) -> dict[str, Any]:

        active_weights = self.selector.get_active_weights()

        if not active_weights:
            final_signal = 0.0
        else:
            total_w = sum(active_weights.values())
            final_signal = sum(
                model_signals.get(name, 0.0) * (w / total_w)
                for name, w in active_weights.items()
            )

        final_signal = float(np.clip(final_signal, -1.0, 1.0))

        self.decision_count += 1
        if self.decision_count % 25 == 0:
            logger.info(f"LOT 46 [{symbol}] | Active: {list(active_weights.keys())} | Signal: {final_signal:.3f}")

        return {
            "final_signal": final_signal,
            "active_weights": active_weights,
            "active_models": list(active_weights.keys())
        }
