"""
LOT 43: GNN Integration helper
"""
import logging
from models.gnn_dependency import LiveGNNEngine
from rl.gnn_rl_agent import GNNRLHybridAgent

logger = logging.getLogger("LOT43_Integration")

def create_lot43_components(multi_agent_rl, asset_list, gnn_weight: float = 0.32):
    """Factory for LOT 43"""
    gnn_engine = LiveGNNEngine(asset_list=asset_list)
    hybrid_agent = GNNRLHybridAgent(
        multi_agent_rl=multi_agent_rl,
        gnn_engine=gnn_engine,
        gnn_weight=gnn_weight
    )
    logger.info("LOT 43: GNN + Hybrid RL components created")
    return gnn_engine, hybrid_agent
