import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("RiskCovariance")

class RiskCovarianceEngine:
    """
    Institutional Portfolio Risk Covariance Engine.
    Calculates rolling covariance and Pearson correlation matrices between multi-assets,
    restricting trade size if total joint portfolio correlation risk is too high.
    """
    def __init__(self, max_correlation_threshold=0.75):
        self.max_correlation_threshold = max_correlation_threshold

    def calculate_correlation_matrix(self, assets_returns_dict: dict) -> pd.DataFrame:
        """
        Computes the Pearson correlation matrix given a dictionary of assets returns.
        assets_returns_dict: dict of symbol -> list/array of percentage returns
        """
        # Align lengths and create a DataFrame
        min_len = min(len(r) for r in assets_returns_dict.values()) if assets_returns_dict else 0
        if min_len < 5:
            # Fallback identity matrix
            symbols = list(assets_returns_dict.keys())
            return pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols)
            
        aligned_data = {symbol: r[-min_len:] for symbol, r in assets_returns_dict.items()}
        df_rets = pd.DataFrame(aligned_data)
        
        # Pearson correlation matrix
        corr_matrix = df_rets.corr()
        return corr_matrix

    def evaluate_portfolio_concentration_risk(self, symbol: str, active_positions: list, corr_matrix: pd.DataFrame) -> float:
        """
        Calculates a scale factor [0.0 to 1.0] to restrict order sizes if the proposed asset
        is highly correlated with already active portfolio exposures.
        """
        if symbol not in corr_matrix.index or not active_positions:
            return 1.0 # No risk
            
        high_corr_exposure_value = 0.0
        total_portfolio_value = 0.0
        
        for pos in active_positions:
            pos_symbol = pos['symbol']
            pos_value = pos['qty'] * pos['avg_price']
            total_portfolio_value += pos_value
            
            if pos_symbol in corr_matrix.index:
                correlation = corr_matrix.loc[symbol, pos_symbol]
                # If correlation is high (e.g. > 0.70), we count this position as highly correlated risk
                if correlation >= self.max_correlation_threshold:
                    high_corr_exposure_value += pos_value
                    
        if total_portfolio_value == 0:
            return 1.0
            
        concentration_ratio = high_corr_exposure_value / total_portfolio_value
        
        # Sizing reduction: if over 40% of the portfolio is in highly correlated assets,
        # we scale down the trade size of the proposed asset proportionally!
        if concentration_ratio > 0.40:
            reduction_factor = 1.0 - (concentration_ratio - 0.40)
            reduction_factor = max(0.20, reduction_factor) # Keep a minimum floor of 20%
            logger.info(f"CORRELATION RISK: {symbol} is highly correlated with active positions. Restricting trade size by factor {reduction_factor:.2f}")
            return float(reduction_factor)
            
        return 1.0
