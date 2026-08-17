"""
Causal Inference Engine (LOT 37)
Determines whether a trading signal is likely causal or merely correlated.
Uses Granger Causality + Partial Correlation for robustness.
"""
import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Tuple

logger = logging.getLogger("CausalInference")

class CausalSignalFilter:
    """
    Causal Inference Filter for Trading Signals.
    - Uses Granger Causality to test if the signal "causes" future price movements.
    - Uses Partial Correlation to remove confounding effects.
    - Significantly reduces false positives from spurious correlations.
    """
    
    def __init__(self, max_lag: int = 5, significance_level: float = 0.05):
        self.max_lag = max_lag
        self.significance_level = significance_level

    def granger_causality_test(self, signal: np.ndarray, price: np.ndarray) -> bool:
        """
        Returns True if the signal Granger-causes price movements.
        """
        if len(signal) < 50 or len(price) < 50:
            return True  # Not enough data → conservative (allow)

        try:
            # Prepare data for Granger test
            data = pd.DataFrame({
                'signal': signal[-100:],
                'price': price[-100:]
            }).dropna()

            if len(data) < 30:
                return True

            # Test if signal causes price
            result = grangercausalitytests(data[['price', 'signal']], maxlag=self.max_lag, verbose=False)
            
            p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, self.max_lag + 1)]
            min_p = min(p_values)
            
            is_causal = min_p < self.significance_level
            return is_causal
            
        except Exception as e:
            logger.warning(f"Granger Causality test failed: {e}")
            return True  # Conservative fallback

    def partial_correlation(self, signal: np.ndarray, price: np.ndarray, 
                           control: np.ndarray = None) -> float:
        """
        Computes partial correlation between signal and future price.
        """
        try:
            if control is None:
                control = np.zeros_like(signal)
            
            # Simple partial correlation using residuals
            from scipy.stats import pearsonr
            
            # Future price
            future_price = np.roll(price, -3)[:-3]
            sig = signal[:-3]
            
            if len(future_price) < 20:
                return 0.0
            
            # Remove linear effect of control variable
            if len(control) == len(sig):
                res_sig = sig - np.dot(control[:-3], np.linalg.lstsq(control[:-3].reshape(-1,1), sig, rcond=None)[0])
                res_price = future_price - np.dot(control[:-3], np.linalg.lstsq(control[:-3].reshape(-1,1), future_price, rcond=None)[0])
            else:
                res_sig = sig
                res_price = future_price
            
            corr, _ = pearsonr(res_sig, res_price)
            return float(corr)
            
        except Exception as e:
            logger.warning(f"Partial correlation failed: {e}")
            return 0.0

    def is_signal_causal(self, signal_series: np.ndarray, 
                         price_series: np.ndarray,
                         control_series: np.ndarray = None) -> Tuple[bool, float]:
        """
        Main method.
        Returns (is_causal, causality_score)
        """
        if len(signal_series) < 40 or len(price_series) < 40:
            return True, 0.5  # Not enough data

        # Granger Causality
        granger_ok = self.granger_causality_test(signal_series, price_series)
        
        # Partial Correlation (future price)
        partial_corr = self.partial_correlation(signal_series, price_series, control_series)
        
        # Combined score
        causality_score = (0.6 if granger_ok else 0.2) + (partial_corr * 0.4)
        causality_score = max(0.0, min(1.0, causality_score))
        
        is_causal = causality_score > 0.45
        
        if not is_causal:
            logger.info(f"Causal Inference: Signal likely spurious (score={causality_score:.3f})")
        
        return is_causal, causality_score