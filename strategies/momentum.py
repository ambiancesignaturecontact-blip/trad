"""
Momentum Strategy - Nouvelle stratégie ajoutée (LOT 7)
"""
import numpy as np
import pandas as pd
from strategies.engine import BaseStrategy

class MomentumStrategy(BaseStrategy):
    """
    Stratégie de Momentum pure :
    - Rate of Change (ROC)
    - RSI Momentum
    - Volume confirmation
    """
    def __init__(self, params=None):
        default = {
            'roc_period': 14,
            'rsi_period': 14,
            'volume_ma': 20,
            'min_momentum': 0.012
        }
        default.update(params or {})
        super().__init__("Momentum", default)

    def generate_signal(self, market_data):
        df = market_data.get('df')
        if df is None or len(df) < max(self.params['roc_period'], self.params['rsi_period']) + 5:
            return 0.0, 0.0

        close = df['close'].values
        volume = df['volume'].values

        # Rate of Change
        roc = (close[-1] - close[-self.params['roc_period']]) / close[-self.params['roc_period']]

        # RSI
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.params['rsi_period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        # Volume confirmation
        vol_ma = pd.Series(volume).rolling(self.params['volume_ma']).mean().iloc[-1]
        vol_ratio = volume[-1] / vol_ma if vol_ma > 0 else 1.0

        momentum_score = 0.0

        if roc > self.params['min_momentum']:
            momentum_score += 0.55
        elif roc < -self.params['min_momentum']:
            momentum_score -= 0.55

        if rsi > 65:
            momentum_score += 0.25
        elif rsi < 35:
            momentum_score -= 0.25

        if vol_ratio > 1.4:
            momentum_score *= 1.15

        signal = np.clip(momentum_score, -1.0, 1.0)
        confidence = min(1.0, abs(roc) * 8 + (vol_ratio - 1) * 0.5)

        return float(signal), float(confidence)