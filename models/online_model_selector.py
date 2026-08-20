"""
LOT 46: Online Model Selection & Adaptive Ensemble Pruning
"""

import logging
from collections import deque

import numpy as np

logger = logging.getLogger("OnlineModelSelector")

class OnlineModelSelector:
    def __init__(self, model_names: list[str],
                 window_size: int = 80,
                 min_weight: float = 0.07,
                 decay: float = 0.93):
        self.model_names = model_names
        self.window_size = window_size
        self.min_weight = min_weight
        self.decay = decay

        self.performance = {name: deque(maxlen=window_size) for name in model_names}
        self.weights = {name: 1.0 / len(model_names) for name in model_names}
        self.active_models: set[str] = set(model_names)
        self.update_count = 0

        logger.info(f"LOT 46: OnlineModelSelector initialized ({len(model_names)} models)")

    def update_performance(self, model_name: str, pnl: float, confidence: float = 1.0):
        if model_name not in self.performance:
            return
        score = pnl * confidence
        self.performance[model_name].append(score)
        self._recalculate_weights()

    def _recalculate_weights(self):
        scores = {}
        for name in self.model_names:
            if len(self.performance[name]) >= 5:
                recent = list(self.performance[name])[-25:]
                scores[name] = np.mean(recent)
            else:
                scores[name] = 0.01

        exp_scores = {k: np.exp(v * 4.0) for k, v in scores.items()}
        total = sum(exp_scores.values()) or 1.0

        new_weights = {}
        for name in self.model_names:
            w = exp_scores[name] / total
            new_weights[name] = max(w, self.min_weight)

        total_w = sum(new_weights.values())
        self.weights = {k: v / total_w for k, v in new_weights.items()}

        self.active_models = {
            name for name, w in self.weights.items()
            if w > self.min_weight * 1.3
        }

        self.update_count += 1
        if self.update_count % 20 == 0:
            logger.info(f"LOT 46: Active models updated → {list(self.active_models)}")

    def get_active_weights(self) -> dict[str, float]:
        return {name: self.weights[name] for name in self.active_models}

    def get_status(self) -> dict:
        return {
            "active_models": list(self.active_models),
            "weights": {k: round(v, 3) for k, v in self.weights.items()},
            "samples": {k: len(v) for k, v in self.performance.items()}
        }
