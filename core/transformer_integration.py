"""
LOT 42: Integration helper for Transformer + RL
"""

import logging
from models.transformer_forecaster import LiveTransformerEngine
from rl.transformer_rl import TransformerRLAgent

logger = logging.getLogger("LOT42_Integration")

def create_lot42_components(multi_agent_rl, blend_weight: float = 0.38):
    """Factory function for LOT 42"""
    transformer_engine = LiveTransformerEngine(seq_len=28)
    hybrid_agent = TransformerRLAgent(
        multi_agent_rl=multi_agent_rl,
        transformer_engine=transformer_engine,
        blend_weight=blend_weight
    )
    
    logger.info("LOT 42: Transformer + Hybrid RL components created")
    return transformer_engine, hybrid_agent
