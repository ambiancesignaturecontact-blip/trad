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

        FIX P0-4 (audit §2.1 / logs prod) : le bucket_size_volume était FIXE (50),
        alors que les barres ont des volumes de 250-1000+ unités -> chaque barre
        formait son propre bucket et le dénominateur (N * 50) devenait minuscule :
        VPIN mesuré à 6 988 465 sur BTC (une probabilité bornée [0,1] !). Le
        paramètre num_buckets était même ignoré.
        Définition standard : on découpe le flux en num_buckets buckets de VOLUME
        ÉGAL (bucket_size = volume_total / num_buckets), puis
        VPIN = sum(|V_buy - V_sell|) / volume_total  -> mathématiquement borné [0,1].
        """
        if df_ticks is None or df_ticks.empty or 'volume' not in df_ticks.columns or 'close' not in df_ticks.columns:
            return 0.5  # Neutral fallback

        prices = df_ticks['close'].values
        volumes = df_ticks['volume'].values
        volumes = np.asarray(volumes, dtype=float)

        total_volume_all = float(np.nansum(volumes))
        if total_volume_all <= 0 or num_buckets < 2:
            return 0.5

        # Direction du tick : signe de la variation de prix (proxy buy/sell)
        price_diffs = np.diff(prices)
        tick_directions = np.sign(price_diffs)
        tick_directions = np.insert(tick_directions, 0, 0.0)

        bucket_size = total_volume_all / num_buckets

        # Partition en buckets de volume égal
        buy_volume_buckets = []
        sell_volume_buckets = []
        current_buy_vol = 0.0
        current_sell_vol = 0.0
        current_total_vol = 0.0

        for idx in range(len(volumes)):
            vol = volumes[idx]
            direction = tick_directions[idx]

            if direction >= 0:
                current_buy_vol += vol
            else:
                current_sell_vol += vol

            current_total_vol += vol

            if current_total_vol >= bucket_size:
                buy_volume_buckets.append(current_buy_vol)
                sell_volume_buckets.append(current_sell_vol)
                current_buy_vol = 0.0
                current_sell_vol = 0.0
                current_total_vol = 0.0

        # Dernier bucket résiduel (partiel) : inclus pour ne pas perdre du volume
        if current_total_vol > 0 and len(buy_volume_buckets) > 0:
            buy_volume_buckets.append(current_buy_vol)
            sell_volume_buckets.append(current_sell_vol)

        if len(buy_volume_buckets) < 5:
            return 0.5

        # VPIN = sum(|V_buy - V_sell|) / volume_total  -> borné [0,1] par construction
        abs_imbalances = np.abs(np.array(buy_volume_buckets) - np.array(sell_volume_buckets))
        vpin = float(np.sum(abs_imbalances)) / total_volume_all if total_volume_all > 0 else 0.5
        # garde finale : jamais hors [0,1] (défense en profondeur)
        return float(min(1.0, max(0.0, vpin)))

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
