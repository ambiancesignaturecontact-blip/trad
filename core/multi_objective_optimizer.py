"""
LOT 56: Multi-Objective Portfolio Optimizer
Optimizes for Sharpe Ratio + CVaR + Maximum Drawdown
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import minimize

logger = logging.getLogger("MultiObjectiveOptimizer")

class MultiObjectivePortfolioOptimizer:
    """
    LOT 56: Portfolio optimizer balancing multiple objectives.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.last_weights = None
        self.last_objectives = None

    def _portfolio_stats(self, weights: np.ndarray, returns: pd.DataFrame) -> Dict:
        """Calculate key portfolio metrics"""
        port_return = np.sum(returns.mean() * weights) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        # CVaR approximation (historical)
        port_returns = returns @ weights
        var = np.percentile(port_returns, 5)
        cvar = port_returns[port_returns <= var].mean() if len(port_returns[port_returns <= var]) > 0 else var

        # Max Drawdown
        cum_returns = (1 + port_returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()

        return {
            "return": port_return,
            "volatility": port_vol,
            "sharpe": sharpe,
            "cvar": cvar,
            "max_drawdown": max_dd
        }

    def optimize(self, 
                 returns: pd.DataFrame, 
                 target_return: Optional[float] = None,
                 weights_sharpe: float = 0.4,
                 weights_cvar: float = 0.35,
                 weights_drawdown: float = 0.25) -> Dict:
        """
        Multi-objective optimization using weighted sum.
        """
        n_assets = returns.shape[1]
        bounds = tuple((0, 1) for _ in range(n_assets))
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        def objective(weights):
            stats = self._portfolio_stats(weights, returns)
            
            # Normalize objectives (lower is better for risk, higher for return)
            sharpe_score = -stats["sharpe"] * weights_sharpe
            cvar_score = stats["cvar"] * weights_cvar * 10   # scale
            dd_score = stats["max_drawdown"] * weights_drawdown * 5

            return sharpe_score + cvar_score + dd_score

        # Initial guess: equal weight
        init_weights = np.array([1.0 / n_assets] * n_assets)

        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False, 'maxiter': 300}
        )

        if result.success:
            optimal_weights = result.x
            stats = self._portfolio_stats(optimal_weights, returns)

            self.last_weights = optimal_weights
            self.last_objectives = stats

            logger.info(f"LOT 56: Multi-objective optimization completed. Sharpe: {stats['sharpe']:.2f}, CVaR: {stats['cvar']:.4f}, MaxDD: {stats['max_drawdown']:.2%}")

            return {
                "weights": {col: round(w, 4) for col, w in zip(returns.columns, optimal_weights)},
                "sharpe": round(stats["sharpe"], 3),
                "cvar": round(stats["cvar"], 4),
                "max_drawdown": round(stats["max_drawdown"], 4),
                "expected_return": round(stats["return"], 4),
                "volatility": round(stats["volatility"], 4)
            }
        else:
            logger.warning("LOT 56: Optimization failed")
            return {"error": "Optimization failed"}

    def get_last_weights(self) -> Optional[np.ndarray]:
        return self.last_weights
