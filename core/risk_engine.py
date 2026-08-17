"""
Risk Engine Module
Centralise les calculs de risque, CVaR, corrélation, etc.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict

logger = logging.getLogger("RiskEngine")

class RiskEngine:
    def __init__(self, risk_manager, covariance_engine):
        self.risk_manager = risk_manager
        self.covariance = covariance_engine

    def calculate_portfolio_metrics(self, positions: list, returns_dict: dict) -> Dict:
        """Calcule VaR/CVaR et réduction de risque"""
        try:
            corr_df = self.covariance.calculate_correlation_matrix(returns_dict)
            var_metrics = self.covariance.calculate_portfolio_var_cvar(
                active_positions=positions,
                corr_matrix=corr_df,
                assets_returns_dict=returns_dict
            )
            return var_metrics
        except Exception as e:
            logger.warning(f"Portfolio risk calculation failed: {e}")
            return {"portfolio_cvar_pct": 0.02, "tripped": False}

    def apply_risk_scaling(self, target_qty: float, news_scale: float, macro_scale: float, 
                          onchain_risk: float, corr_reduction: float) -> float:
        """Applique tous les facteurs de réduction de risque"""
        qty = target_qty
        qty *= news_scale
        qty *= macro_scale
        
        if onchain_risk > 0.75:
            qty *= 0.5
            logger.info("ON-CHAIN WARNING: Scaling down position size")
            
        qty *= corr_reduction
        return max(0.0, qty)