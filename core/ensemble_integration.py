"""
LOT 44: Ensemble Integration helper
"""
import logging
from models.ensemble_controller import AdaptiveEnsembleController
from rl.ensemble_hybrid_agent import EnsembleHybridAgent

logger = logging.getLogger("LOT44_Integration")

def create_lot44_components(multi_agent_rl=None):
    """Factory for LOT 44 Adaptive Ensemble"""
    model_names = [
        "transformer", "gnn", "meta_labeling", "multi_agent_rl",
        "bayesian_risk", "causal_filter", "regime_switcher"
    ]
    
    ensemble_controller = AdaptiveEnsembleController(
        model_names=model_names, decay=0.93, min_weight=0.06
    )
    
    hybrid_agent = EnsembleHybridAgent(
        ensemble_controller=ensemble_controller,
        multi_agent_rl=multi_agent_rl
    )
    
    logger.info("LOT 44: Adaptive Ensemble + Hybrid Agent created")
    return ensemble_controller, hybrid_agent
