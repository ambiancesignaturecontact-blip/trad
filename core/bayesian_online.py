"""
Bayesian Online Learning Engine (LOT 40)
Updates beliefs about returns and volatility in real-time using conjugate priors.
"""
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger("BayesianOnline")

class BayesianReturnEstimator:
    """
    Bayesian estimator for asset/strategy returns.
    Uses Normal-Inverse-Gamma conjugate prior for mean and variance.
    """
    
    def __init__(self, prior_mean: float = 0.0, prior_var: float = 0.01, 
                 prior_nu: float = 5.0, prior_lambda: float = 1.0):
        self.mu = prior_mean          # Prior mean
        self.var = prior_var          # Prior variance
        self.nu = prior_nu            # Degrees of freedom
        self.lam = prior_lambda       # Precision parameter
        
        self.n_updates = 0

    def update(self, new_return: float):
        """Bayesian update with new observation"""
        self.n_updates += 1
        
        # Update precision (lambda)
        self.lam = self.lam + 1
        
        # Update mean
        old_mu = self.mu
        self.mu = (self.lam * self.mu + new_return) / (self.lam + 1)
        
        # Update variance (simplified)
        self.var = ((self.nu * self.var) + (self.lam * (old_mu - self.mu)**2) + (new_return - self.mu)**2) / (self.nu + 1)
        
        # Update degrees of freedom
        self.nu += 1

    def get_posterior_mean(self) -> float:
        return self.mu

    def get_posterior_variance(self) -> float:
        return self.var / self.lam

    def get_predictive_distribution(self) -> Tuple[float, float]:
        """Returns mean and variance of the predictive distribution"""
        mean = self.mu
        variance = self.var * (1 + 1/self.lam) * (self.nu / (self.nu - 2)) if self.nu > 2 else self.var
        return mean, variance


class BayesianOnlineRiskManager:
    """
    Manages Bayesian estimates for all assets/strategies.
    """
    
    def __init__(self):
        self.estimators = {}  # symbol/strategy -> BayesianReturnEstimator

    def update(self, symbol: str, new_return: float):
        if symbol not in self.estimators:
            self.estimators[symbol] = BayesianReturnEstimator()
        
        self.estimators[symbol].update(new_return)

    def get_bayesian_kelly_fraction(self, symbol: str, win_rate: float = 0.55, 
                                    reward_risk: float = 1.5) -> float:
        """
        Computes a Bayesian-adjusted Kelly fraction.
        """
        if symbol not in self.estimators:
            return 0.12  # Default conservative value
        
        est = self.estimators[symbol]
        mean, var = est.get_predictive_distribution()
        
        # Adjust win rate based on Bayesian mean
        adjusted_win_rate = 0.5 + (mean / (2 * np.sqrt(var))) if var > 0 else win_rate
        
        # Bayesian Kelly
        kelly = (adjusted_win_rate * reward_risk - (1 - adjusted_win_rate)) / reward_risk
        return max(0.03, min(kelly * 0.25, 0.25))  # Fractional Kelly

    def get_all_estimates(self) -> Dict:
        result = {}
        for symbol, est in self.estimators.items():
            mean, var = est.get_predictive_distribution()
            result[symbol] = {
                "posterior_mean": round(mean, 6),
                "posterior_variance": round(var, 6),
                "updates": est.n_updates
            }
        return result