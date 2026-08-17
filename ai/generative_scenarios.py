"""
Generative Scenario Engine (LOT 38)
Regime-Conditioned Scenario Generation with realistic stress injection.
Generates extreme but plausible market scenarios for stress testing and CVaR.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from scipy.stats import multivariate_normal

logger = logging.getLogger("GenerativeScenarios")

class RegimeConditionedScenarioGenerator:
    """
    Generates realistic extreme market scenarios conditioned on the current regime.
    Uses:
    - Regime-specific return distributions
    - Correlation structure preservation
    - Stress injection (fat tails + jumps)
    """
    
    def __init__(self, n_scenarios: int = 5000):
        self.n_scenarios = n_scenarios
        self.regime_params = {}  # Will store mean/cov per regime

    def fit(self, returns_dict: Dict[str, np.ndarray], regimes: np.ndarray):
        """
        Fit regime-specific distributions from historical data.
        returns_dict: {symbol: returns_array}
        regimes: array of regime labels aligned with returns
        """
        unique_regimes = np.unique(regimes)
        
        for regime in unique_regimes:
            mask = regimes == regime
            regime_returns = []
            
            for symbol, ret in returns_dict.items():
                if len(ret) == len(regimes):
                    regime_returns.append(ret[mask])
            
            if regime_returns:
                regime_matrix = np.column_stack(regime_returns)
                self.regime_params[regime] = {
                    'mean': np.mean(regime_matrix, axis=0),
                    'cov': np.cov(regime_matrix, rowvar=False) + np.eye(regime_matrix.shape[1]) * 1e-6
                }
        
        logger.info(f"Generative Scenario Engine fitted on {len(unique_regimes)} regimes")

    def generate_scenarios(self, current_regime: int, n_assets: int, 
                          stress_level: float = 1.0) -> np.ndarray:
        """
        Generate n_scenarios of returns for the given regime.
        stress_level: 1.0 = normal, >1.0 = stressed (fat tails + jumps)
        """
        if current_regime not in self.regime_params:
            current_regime = list(self.regime_params.keys())[0]
        
        params = self.regime_params[current_regime]
        mean = params['mean']
        cov = params['cov']
        
        # Base multivariate normal
        scenarios = np.random.multivariate_normal(mean, cov, self.n_scenarios)
        
        # Stress injection
        if stress_level > 1.0:
            # Fat tails (scale volatility)
            scenarios *= stress_level
            
            # Add jumps (occasional large moves)
            jump_prob = 0.03 * stress_level
            jump_mask = np.random.rand(self.n_scenarios, n_assets) < jump_prob
            jump_size = np.random.choice([-1, 1], size=(self.n_scenarios, n_assets)) * \
                       np.random.exponential(0.025, size=(self.n_scenarios, n_assets)) * stress_level
            
            scenarios[jump_mask] += jump_size[jump_mask]
        
        return scenarios

    def generate_stress_scenarios(self, current_regime: int, n_assets: int,
                                 n_stress: int = 1000) -> np.ndarray:
        """
        Generate specifically stressed scenarios (worst 5% tail).
        """
        scenarios = self.generate_scenarios(current_regime, n_assets, stress_level=2.5)
        
        # Keep only the worst scenarios (lowest portfolio returns)
        portfolio_returns = scenarios.mean(axis=1)
        worst_idx = np.argsort(portfolio_returns)[:n_stress]
        
        return scenarios[worst_idx]