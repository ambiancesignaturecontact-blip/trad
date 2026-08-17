"""
LOT 44: Adaptive Online Ensemble Controller
"""
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from collections import deque

logger = logging.getLogger("EnsembleController")

class AdaptiveEnsembleController:
    def __init__(self, model_names: List[str], decay: float = 0.92, min_weight: float = 0.05):
        self.model_names = model_names
        self.n_models = len(model_names)
        self.decay = decay
        self.min_weight = min_weight
        self.performance = {name: 0.0 for name in model_names}
        self.counts = {name: 0 for name in model_names}
        self.weights = np.ones(self.n_models) / self.n_models
        self.weight_history = deque(maxlen=200)
        self.performance_history = deque(maxlen=500)
        logger.info(f"LOT 44: AdaptiveEnsembleController initialized with {self.n_models} models")

    def update_performance(self, model_name: str, pnl: float, confidence: float = 1.0):
        if model_name not in self.performance: return
        old_score = self.performance[model_name]
        new_score = old_score * self.decay + pnl * confidence * (1 - self.decay)
        self.performance[model_name] = new_score
        self.counts[model_name] += 1
        self._recalculate_weights()
        self.performance_history.append({"model": model_name, "pnl": pnl, "score": new_score})

    def _recalculate_weights(self):
        scores = np.array([self.performance[name] for name in self.model_names])
        scores = scores - np.min(scores) + 0.01
        exp_scores = np.exp(scores * 2.5)
        weights = exp_scores / np.sum(exp_scores)
        weights = np.maximum(weights, self.min_weight)
        weights = weights / np.sum(weights)
        self.weights = weights
        self.weight_history.append(dict(zip(self.model_names, self.weights)))

    def get_blended_signal(self, signals: Dict[str, float], confidences: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        if confidences is None:
            confidences = {name: 1.0 for name in self.model_names}
        blended = 0.0
        contributions = {}
        for i, name in enumerate(self.model_names):
            if name in signals:
                w = self.weights[i]
                conf = confidences.get(name, 1.0)
                contrib = signals[name] * w * conf
                blended += contrib
                contributions[name] = {"signal": round(signals[name], 3), "weight": round(w, 3), "contribution": round(contrib, 3)}
        blended = float(np.clip(blended, -1.0, 1.0))
        return {
            "final_signal": blended,
            "contributions": contributions,
            "weights": dict(zip(self.model_names, [round(w, 3) for w in self.weights])),
            "ensemble_confidence": float(np.mean(list(confidences.values())))
        }

    def get_status(self) -> dict:
        return {
            "models": self.model_names,
            "current_weights": dict(zip(self.model_names, [round(w, 3) for w in self.weights])),
            "performance_scores": {k: round(v, 4) for k, v in self.performance.items()},
            "total_updates": sum(self.counts.values())
        }
