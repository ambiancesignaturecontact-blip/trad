"""
LOT 46: Integration helper
"""
import logging

from models.online_model_selector import OnlineModelSelector
from rl.adaptive_ensemble_agent import AdaptiveEnsembleAgent

logger = logging.getLogger("LOT46_Integration")

def create_lot46_components(model_names=None):
    if model_names is None:
        model_names = ["transformer", "gnn", "meta_labeling", "multi_agent_rl", "bayesian_risk"]

    selector = OnlineModelSelector(model_names)
    agent = AdaptiveEnsembleAgent(selector)

    logger.info("LOT 46: Online Model Selector + Adaptive Ensemble created")
    return selector, agent
