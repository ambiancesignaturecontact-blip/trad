"""
LOT 44: Ensemble + RL Hybrid Agent
"""
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger("EnsembleHybridAgent")

class EnsembleHybridAgent:
    def __init__(self, ensemble_controller, multi_agent_rl=None):
        self.ensemble = ensemble_controller
        self.multi_agent_rl = multi_agent_rl
        self.decision_log = []
        logger.info("LOT 44: EnsembleHybridAgent initialized")

    def decide(self, symbol: str, market_data: dict, model_signals: Dict[str, float],
               model_confidences: Dict[str, float] = None, regime_id: int = 2) -> Dict[str, Any]:
        ensemble_result = self.ensemble.get_blended_signal(signals=model_signals, confidences=model_confidences)
        final_signal = ensemble_result["final_signal"]
        if self.multi_agent_rl and hasattr(self.multi_agent_rl, 'get_action'):
            try:
                rl_signal = self.multi_agent_rl.get_action(symbol, market_data)
                final_signal = 0.85 * final_signal + 0.15 * rl_signal
                final_signal = float(np.clip(final_signal, -1.0, 1.0))
            except:
                pass
        explanation = {
            "ensemble_weights": ensemble_result["weights"],
            "model_contributions": ensemble_result["contributions"],
            "ensemble_confidence": round(ensemble_result["ensemble_confidence"], 3),
            "final_signal": round(final_signal, 3)
        }
        self.decision_log.append({"symbol": symbol, "final": final_signal, "weights": ensemble_result["weights"]})
        if len(self.decision_log) % 25 == 0:
            logger.info(f"LOT 44 [{symbol}] | Ensemble decision: {explanation['final_signal']}")
        return {"final_signal": final_signal, "explanation": explanation, "ensemble_result": ensemble_result}

    def update_model_performance(self, model_name: str, pnl: float, confidence: float = 1.0):
        self.ensemble.update_performance(model_name, pnl, confidence)
