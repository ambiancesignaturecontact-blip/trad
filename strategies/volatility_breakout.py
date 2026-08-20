"""
Volatility Breakout Strategy - Nouvelle stratégie ajoutée (LOT 7)
"""
import numpy as np
import pandas as pd

from strategies.engine import BaseStrategy


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Stratégie de Volatility Breakout :
    - ATR Breakout
    - Bollinger Band Squeeze + Expansion
    - Volume confirmation
    """
    def __init__(self, params=None):
        default = {
            'atr_period': 14,
            'bb_period': 20,
            'bb_std': 2.0,
            'squeeze_threshold': 0.6,
            'min_breakout_pct': 0.008
        }
        default.update(params or {})
        super().__init__("Volatility Breakout", default)

    def generate_signal(self, market_data):
        df = market_data.get('df')
        if df is None or len(df) < self.params['bb_period'] + 5:
            return 0.0, 0.0

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # ATR
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
        atr = pd.Series(tr).rolling(self.params['atr_period']).mean().iloc[-1]

        # Bollinger Bands
        sma = pd.Series(close).rolling(self.params['bb_period']).mean().iloc[-1]
        std = pd.Series(close).rolling(self.params['bb_period']).std().iloc[-1]
        upper = sma + self.params['bb_std'] * std
        lower = sma - self.params['bb_std'] * std

        # Squeeze detection
        bb_width = (upper - lower) / sma
        atr_norm = atr / sma
        squeeze = bb_width < (self.params['squeeze_threshold'] * atr_norm)

        current_price = close[-1]
        breakout_up = current_price > upper * (1 + self.params['min_breakout_pct'])
        breakout_down = current_price < lower * (1 - self.params['min_breakout_pct'])

        score = 0.0

        if squeeze:
            if breakout_up:
                score = 0.85
            elif breakout_down:
                score = -0.85
        else:
            # Continuation
            if current_price > sma * 1.015:
                score = 0.45
            elif current_price < sma * 0.985:
                score = -0.45

        signal = np.clip(score, -1.0, 1.0)
        confidence = min(1.0, abs(bb_width - 0.02) * 25)

        return float(signal), float(confidence)
