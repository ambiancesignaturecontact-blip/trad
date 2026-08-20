import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("RiskCovariance")

class RiskCovarianceEngine:
    """
    Institutional Portfolio Risk Covariance Engine.
    Calculates rolling covariance and Pearson correlation matrices between multi-assets,
    restricting trade size if total joint portfolio correlation risk is too high.

    Includes Portfolio-level Value at Risk (VaR) and Conditional Value at Risk (CVaR)
    to calculate joint covariance circuit breakers!
    """
    def __init__(self, max_correlation_threshold=0.70, max_portfolio_var_pct=0.05):
        self.max_correlation_threshold = max_correlation_threshold
        self.max_portfolio_var_pct = max_portfolio_var_pct # 5% max daily VaR threshold

    def calculate_correlation_matrix(self, assets_returns_dict: dict) -> pd.DataFrame:
        min_len = min(len(r) for r in assets_returns_dict.values()) if assets_returns_dict else 0
        if min_len < 5:
            symbols = list(assets_returns_dict.keys())
            return pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols)

        aligned_data = {symbol: r[-min_len:] for symbol, r in assets_returns_dict.items()}
        df_rets = pd.DataFrame(aligned_data)

        corr_matrix = df_rets.corr()
        return corr_matrix

    def calculate_portfolio_var_cvar(self, active_positions: list, corr_matrix: pd.DataFrame, assets_returns_dict: dict, confidence_level=0.95) -> dict:
        """
        Calculates joint covariance Portfolio Value at Risk (VaR) and Conditional Value at Risk (CVaR)
        using Parametric (Variance-Covariance) and Historical Simulation methods.
        """
        if not active_positions or corr_matrix.empty:
            return {"portfolio_var_pct": 0.0, "portfolio_cvar_pct": 0.0, "tripped": False}

        # Get active positions and their relative weights
        position_values = []
        symbols = []
        for pos in active_positions:
            val = pos['qty'] * pos['avg_price']
            if val > 0:
                position_values.append(val)
                symbols.append(pos['symbol'])

        total_portfolio_value = sum(position_values)
        if total_portfolio_value == 0:
            return {"portfolio_var_pct": 0.0, "portfolio_cvar_pct": 0.0, "tripped": False}

        weights = np.array(position_values) / total_portfolio_value

        # Calculate covariance matrix of active symbols
        min_len = min(len(assets_returns_dict[sym]) for sym in symbols if sym in assets_returns_dict) if assets_returns_dict else 0
        if min_len < 5:
            return {"portfolio_var_pct": 0.0, "portfolio_cvar_pct": 0.0, "tripped": False}

        active_returns = {sym: assets_returns_dict[sym][-min_len:] for sym in symbols if sym in assets_returns_dict}
        df_rets = pd.DataFrame(active_returns)

        cov_matrix = df_rets.cov()

        # Portfolio Variance = w^T * Cov * w
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_std_dev = np.sqrt(port_variance)

        # 1. Parametric VaR (assuming normality): VaR = z * std_dev
        z_score = 1.645 if confidence_level == 0.95 else 2.33
        portfolio_var_pct = z_score * port_std_dev

        # 2. Historical Simulation VaR and CVaR
        # Simulate portfolio returns historically
        simulated_port_returns = np.dot(df_rets.values, weights)
        simulated_port_returns = np.sort(simulated_port_returns)

        cutoff_idx = int((1.0 - confidence_level) * len(simulated_port_returns))
        # CVaR is the average of returns below the VaR cutoff
        hist_cvar_pct = abs(np.mean(simulated_port_returns[:cutoff_idx])) if cutoff_idx > 0 else portfolio_var_pct

        # Evaluate Joint Covariance Circuit Breaker
        # If daily VaR exceeds our threshold (e.g. 5%), we trigger a risk freeze!
        tripped = False
        reason = ""
        if portfolio_var_pct >= self.max_portfolio_var_pct:
            tripped = True
            reason = f"PORTFOLIO COVARIANCE RISK BREACHED: Daily VaR is {portfolio_var_pct*100:.2f}% (limit: {self.max_portfolio_var_pct*100:.2f}%)."
            logger.warning(reason)

        return {
            "portfolio_var_pct": float(portfolio_var_pct),
            "portfolio_cvar_pct": float(hist_cvar_pct),
            "tripped": tripped,
            "reason": reason
        }

    def evaluate_portfolio_concentration_risk(self, symbol: str, active_positions: list, corr_matrix: pd.DataFrame) -> float:
        if symbol not in corr_matrix.index or not active_positions:
            return 1.0

        high_corr_exposure_value = 0.0
        total_portfolio_value = 0.0

        for pos in active_positions:
            pos_symbol = pos['symbol']
            pos_value = pos['qty'] * pos['avg_price']
            total_portfolio_value += pos_value

            if pos_symbol in corr_matrix.index:
                correlation = corr_matrix.loc[symbol, pos_symbol]
                if correlation >= self.max_correlation_threshold:
                    high_corr_exposure_value += pos_value

        if total_portfolio_value == 0:
            return 1.0

        concentration_ratio = high_corr_exposure_value / total_portfolio_value

        if concentration_ratio > 0.40:
            reduction_factor = 1.0 - (concentration_ratio - 0.40)
            reduction_factor = max(0.20, reduction_factor)
            logger.info(f"CORRELATION RISK: {symbol} is highly correlated with active positions. Restricting trade size by factor {reduction_factor:.2f}")
            return float(reduction_factor)

        return 1.0
