import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("MicrostructureEdge")

class MicrostructureEdgeEngine:
    """
    Advanced Microstructure Edge Engine.
    Calculates Volume-Synchronized Probability of Informed Trading (VPIN)
    and Kyle's Lambda (price impact per unit of trade volume).
    """
    def __init__(self, bucket_size_volume=50.0):
        self.bucket_size_volume = bucket_size_volume

    def calculate_vpin(self, df_ticks: pd.DataFrame, num_buckets=50) -> float:
        """
        Calculates VPIN (Volume-Synchronized Probability of Informed Trading).
        Measures the probability that order flow is driven by informed toxic traders.
        """
        if df_ticks.empty or 'volume' not in df_ticks.columns or 'close' not in df_ticks.columns:
            return 0.5 # Neutral fallback
            
        prices = df_ticks['close'].values
        volumes = df_ticks['volume'].values
        
        # Calculate tick direction (Buy vs Sell volume proxy using tick sign)
        price_diffs = np.diff(prices)
        tick_directions = np.sign(price_diffs)
        # Pad first diff
        tick_directions = np.insert(tick_directions, 0, 0.0)
        
        # Partition ticks into equal-volume buckets
        buy_volume_buckets = []
        sell_volume_buckets = []
        
        current_buy_vol = 0.0
        current_sell_vol = 0.0
        current_total_vol = 0.0
        
        for idx in range(len(volumes)):
            vol = volumes[idx]
            direction = tick_directions[idx]
            
            # Classify volume based on tick direction
            if direction >= 0:
                current_buy_vol += vol
            else:
                current_sell_vol += vol
                
            current_total_vol += vol
            
            if current_total_vol >= self.bucket_size_volume:
                buy_volume_buckets.append(current_buy_vol)
                sell_volume_buckets.append(current_sell_vol)
                
                # Reset for next bucket
                current_buy_vol = 0.0
                current_sell_vol = 0.0
                current_total_vol = 0.0
                
        if len(buy_volume_buckets) < 5:
            return 0.5
            
        # VPIN Formula = Sum(|V_buy - V_sell|) / (N_buckets * Bucket_Volume)
        abs_imbalances = np.abs(np.array(buy_volume_buckets) - np.array(sell_volume_buckets))
        total_imbalance = np.sum(abs_imbalances)
        total_volume = len(buy_volume_buckets) * self.bucket_size_volume
        
        vpin = total_imbalance / total_volume if total_volume > 0 else 0.5
        return float(vpin)

    def calculate_kyles_lambda(self, df_bars: pd.DataFrame) -> float:
        """
        Calculates Kyle's Lambda (price impact per unit of volume traded).
        Lambda = Cov(Price_Change, Volume_Imbalance) / Var(Volume_Imbalance)
        """
        if len(df_bars) < 10:
            return 1e-5 # Tiny default
            
        price_changes = df_bars['close'].diff().dropna().values
        volumes = df_bars['volume'].values[-len(price_changes):]
        
        # Simple proxy: imbalance volume is signed volume based on price changes
        imbalances = np.sign(price_changes) * volumes
        
        cov = np.cov(price_changes, imbalances)
        if cov.ndim > 1:
            cov_val = cov[0, 1]
            var_val = cov[1, 1] + 1e-8
            kyles_lambda = cov_val / var_val
        else:
            kyles_lambda = 1e-5
            
        return max(1e-9, float(kyles_lambda))
