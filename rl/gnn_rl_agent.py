"""
LOT 43: GNN + RL Hybrid Agent
"""
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger("GNN_RL_Agent")

class GNNRLHybridAgent:
    def __init__(self, multi_agent_rl, gnn_engine, gnn_weight: float = 0.32):
        self.multi_agent_rl = multi_agent_rl
        self.gnn = gnn_engine
        self.gnn_weight = gnn_weight
        self.signal_history = []
        logger.info("LOT 43: GNNRLHybridAgent initialized")

    def decide(self, symbol: str, market_data: dict, gnn_output: dict, base_signal: float, regime_id: int) -> Dict[str, Any]:
        if symbol not in gnn_output:
            return {"final_signal": base_signal, "gnn_boost": 0.0, "risk_mult": 1.0}
        g = gnn_output[symbol]
        risk_mult = g.get("risk_multiplier", 1.0)
        alpha_boost = g.get("alpha_boost", 0.0)
        rl_signal = base_signal
        gnn_adjusted = rl_signal * risk_mult + alpha_boost
        final = (1 - self.gnn_weight) * rl_signal + self.gnn_weight * gnn_adjusted
        final = float(np.clip(final, -1.0, 1.0))
        explanation = {
            "base_signal": round(rl_signal, 3),
            "risk_multiplier": round(risk_mult, 3),
            "alpha_boost": round(alpha_boost, 3),
            "gnn_weight": self.gnn_weight,
            "final": round(final, 3)
        }
        self.signal_history.append({"symbol": symbol, "final": final})
        if len(self.signal_history) > 300:
            self.signal_history.pop(0)
        return {"final_signal": final, "explanation": explanation, "risk_multiplier": risk_mult}

    def get_recent_performance(self) -> float:
        if not self.signal_history: return 0.0
        return float(np.mean([s["final"] for s in self.signal_history[-50:]]))
