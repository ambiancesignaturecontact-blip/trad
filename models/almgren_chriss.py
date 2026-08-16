import numpy as np
import logging

logger = logging.getLogger("AlmgrenChriss")

class AlmgrenChrissExecutionOptimizer:
    """
    Almgren-Chriss Optimal Execution Transaction Cost Model.
    Solves the tradeoff between execution speed (market impact) and delay (price risk).
    Enables adaptive VWAP/TWAP order slicing.
    """
    def __init__(self, risk_aversion=0.1, temporary_impact_coeff=1e-6, permanent_impact_coeff=1e-7):
        self.risk_aversion = risk_aversion
        self.eta = temporary_impact_coeff # Temporary price impact coefficient
        self.gamma = permanent_impact_coeff # Permanent price impact coefficient

    def calculate_optimal_trajectory(self, total_shares_to_sell: float, time_steps=5, volatility=0.02) -> list:
        """
        Calculates the mathematically optimal list of trade sizes (shares per slice)
        minimizing total expected transaction costs and volatility risks.
        """
        if total_shares_to_sell <= 0 or time_steps <= 1:
            return [total_shares_to_sell]
            
        # Almgren-Chriss closed form scale factor lambda:
        # lambda = sqrt( (risk_aversion * volatility^2) / (temporary_impact_coeff) )
        lam = np.sqrt((self.risk_aversion * (volatility**2)) / (self.eta + 1e-12))
        
        # Hyperbolic sine / cosine optimal trajectory approximation
        optimal_shares_slice = []
        remaining = total_shares_to_sell
        
        for k in range(time_steps):
            t_k = k / time_steps
            # Calculate optimal path fraction: sinh(lambda * (T - t)) / sinh(lambda * T)
            numerator = np.sinh(lam * (1.0 - t_k))
            denominator = np.sinh(lam) + 1e-12
            
            target_pos = total_shares_to_sell * (numerator / denominator)
            slice_qty = remaining - target_pos
            slice_qty = max(0.0, min(slice_qty, remaining))
            
            optimal_shares_slice.append(float(slice_qty))
            remaining -= slice_qty
            
        # Ensure total is fully allocated
        if remaining > 0 and len(optimal_shares_slice) > 0:
            optimal_shares_slice[-1] += remaining
            
        logger.info(f"ALMGREN-CHRISS OPTIMIZATION: Optimal execution path generated: {optimal_shares_slice}")
        return optimal_shares_slice


def calculate_cvar_constrained_sizing(capital: float, current_price: float, cvar_pct: float, max_loss_usd: float) -> float:
    """
    Sizing sensible au risque: CVaR-constrained Position Sizing.
    Limits trade size so that the expected loss in the worst 5% of cases (CVaR)
    never exceeds a maximum allowed absolute dollar loss limit.
    """
    if current_price <= 0 or cvar_pct <= 0:
        return 0.0
        
    # Maximum allowed trade size in USD = Max_Loss_USD / CVaR_pct
    max_trade_size_usd = max_loss_usd / (cvar_pct + 1e-8)
    
    # Capital exposure constraint (max 50% of capital in one asset)
    max_trade_size_usd = min(max_trade_size_usd, capital * 0.50)
    
    qty = max_trade_size_usd / current_price
    return float(qty)
