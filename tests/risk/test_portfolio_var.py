import pytest
import pandas as pd
import numpy as np
from models.risk_covariance import RiskCovarianceEngine

def test_correlation_and_var_calculations():
    engine = RiskCovarianceEngine(max_correlation_threshold=0.70, max_portfolio_var_pct=0.05)
    
    # Generate mock asset returns
    returns = {
        "BTCUSDT": np.array([0.01, -0.005, 0.002, 0.015, -0.008, 0.01]),
        "ETHUSDT": np.array([0.012, -0.004, 0.003, 0.014, -0.007, 0.009])
    }
    
    corr_matrix = engine.calculate_correlation_matrix(returns)
    assert corr_matrix.loc["BTCUSDT", "ETHUSDT"] > 0.80 # highly correlated
    
    # Calculate VaR
    active_positions = [
        {"symbol": "BTCUSDT", "qty": 0.1, "avg_price": 60000.0},
        {"symbol": "ETHUSDT", "qty": 1.0, "avg_price": 2500.0}
    ]
    
    var_res = engine.calculate_portfolio_var_cvar(
        active_positions=active_positions,
        corr_matrix=corr_matrix,
        assets_returns_dict=returns
    )
    
    assert "portfolio_var_pct" in var_res
    assert "portfolio_cvar_pct" in var_res
    assert var_res["portfolio_var_pct"] >= 0.0
