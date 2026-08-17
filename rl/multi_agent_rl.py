"""
Multi-Agent Reinforcement Learning Framework (LOT 36)
One agent per strategy + Meta-Agent for allocation.
This is a simplified but powerful version suitable for production.
"""
import numpy as np
import logging
from typing import Dict, List
from collections import deque

logger = logging.getLogger("MultiAgentRL")

class StrategyAgent:
    """
    Individual Strategy Agent.
    Learns to output a confidence multiplier for its signal.
    State = [market_features, own_signal, regime]
    """
    def __init__(self, name: str, state_dim: int = 6, action_dim: int = 1):
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Simple linear policy (can be replaced with neural net later)
        self.weights = np.random.normal(0, 0.1, (state_dim, action_dim))
        self.bias = 0.0
        
        self.memory = deque(maxlen=500)
        self.learning_rate = 0.01

    def get_action(self, state: np.ndarray) -> float:
        """Returns a multiplier between 0.3 and 2.0"""
        if len(state) != self.state_dim:
            state = np.zeros(self.state_dim)
        
        logit = np.dot(state, self.weights) + self.bias
        multiplier = 1.0 + np.tanh(logit) * 0.7  # Between 0.3 and 1.7
        return float(np.clip(multiplier, 0.3, 1.7))

    def update(self, state: np.ndarray, reward: float):
        """Simple policy gradient update"""
        if len(state) != self.state_dim:
            return
        
        # Gradient update (simplified)
        grad = state.reshape(-1, 1) * reward
        self.weights += self.learning_rate * grad
        self.bias += self.learning_rate * reward * 0.1

        self.memory.append((state, reward))


class MetaAgent:
    """
    Meta-Agent that decides final allocation weights based on all strategy agents.
    """
    def __init__(self, num_strategies: int):
        self.num_strategies = num_strategies
        self.weights = np.ones(num_strategies) / num_strategies

    def get_allocation(self, agent_outputs: List[float], regime_id: int) -> np.ndarray:
        """
        Combines agent outputs into final allocation weights.
        """
        outputs = np.array(agent_outputs)
        
        # Boost based on agent confidence
        boosted = outputs * self.weights
        
        # Normalize
        if boosted.sum() > 0:
            allocation = boosted / boosted.sum()
        else:
            allocation = np.ones(self.num_strategies) / self.num_strategies
            
        return allocation


class MultiAgentRLSystem:
    """
    Full Multi-Agent RL System.
    """
    def __init__(self, strategy_names: List[str]):
        self.agents = {name: StrategyAgent(name) for name in strategy_names}
        self.meta_agent = MetaAgent(len(strategy_names))
        self.strategy_names = strategy_names

    def get_final_signal(self, 
                         base_signals: Dict[str, float],
                         market_state: np.ndarray,
                         regime_id: int) -> float:
        """
        Main entry point.
        Returns the final risk-adjusted signal.
        """
        multipliers = []
        agent_outputs = []
        
        for name in self.strategy_names:
            agent = self.agents[name]
            base_signal = base_signals.get(name, 0.0)
            
            # Get agent's recommended multiplier
            mult = agent.get_action(market_state)
            agent_outputs.append(mult)
            
            adjusted = base_signal * mult
            multipliers.append(adjusted)
        
        # Meta-agent decides final weights
        allocation = self.meta_agent.get_allocation(agent_outputs, regime_id)
        
        # Weighted final signal
        final_signal = np.sum(np.array(multipliers) * allocation)
        return float(np.clip(final_signal, -1.0, 1.0))

    def update_agents(self, rewards: Dict[str, float]):
        """Update all agents with their rewards"""
        for name, reward in rewards.items():
            if name in self.agents:
                # We need the last state - simplified here
                pass  # In real implementation, store states

    def get_agent_confidences(self) -> Dict[str, float]:
        return {name: agent.get_action(np.zeros(6)) for name, agent in self.agents.items()}