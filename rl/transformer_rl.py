"""
LOT 42: Transformer + RL Hybrid Agent
Combines Transformer forecasts with Multi-Agent RL for final decision.
"""

import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger("TransformerRL")

class TransformerRLAgent:
    """
    LOT 42 Hybrid Agent:
    - Uses Transformer forecast
    - Blends with MultiAgentRLSystem
    - Produces final actionable signal
    """
    
    def __init__(self, multi_agent_rl, transformer_engine, blend_weight: float = 0.35):
        self.multi_agent_rl = multi_agent_rl
        self.transformer = transformer_engine
        self.blend_weight = blend_weight
        self.performance_log = []
        
        logger.info("LOT 42: TransformerRLAgent initialized (Hybrid)")

    def decide(self, symbol: str, market_data: dict, transformer_pred: dict, regime_id: int) -> Dict[str, Any]:
        """
        Returns blended signal + explanation
        """
        # Get RL action
        rl_action = self.multi_agent_rl.get_action(symbol, market_data) if hasattr(self.multi_agent_rl, 'get_action') else 0.0
        
        # Transformer prediction
        t_price = transformer_pred.get("price_delta", 0.0)
        t_conf = transformer_pred.get("confidence", 0.5)
        t_regime = transformer_pred.get("regime", regime_id)
        
        # Blend logic
        transformer_signal = np.tanh(t_price * 8.0) * t_conf   # scale to [-1,1]
        
        # Weighted average
        final_signal = (1 - self.blend_weight) * rl_action + self.blend_weight * transformer_signal
        
        # Regime adjustment
        if t_regime != regime_id:
            final_signal *= 0.65
        
        final_signal = float(np.clip(final_signal, -1.0, 1.0))
        
        explanation = {
            "transformer_delta": round(t_price, 4),
            "transformer_conf": round(t_conf, 3),
            "rl_action": round(rl_action, 3),
            "blend_weight": self.blend_weight,
            "final": round(final_signal, 3)
        }
        
        return {
            "final_signal": final_signal,
            "explanation": explanation,
            "transformer_regime": t_regime
        }

    def update_performance(self, symbol: str, pnl: float):
        self.performance_log.append({"symbol": symbol, "pnl": pnl})
        if len(self.performance_log) > 200:
            self.performance_log.pop(0)
