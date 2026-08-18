"""
Institutional signal strategies (VISION_NIVEAU_MONDIAL §5).

- CarryStrategy            : funding-rate carry (delta-neutral-ish signal)
- CrossSectionalMomentum   : rank assets against each other (long strong / short weak)
- MultiTimeframeWrapper    : adapts MultiTimeframeConsensus to the BaseStrategy API
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from strategies.engine import BaseStrategy


class CarryStrategy(BaseStrategy):
    """Scores the funding-rate carry: positive funding on a long position earns
    carry; extreme funding also signals crowdedness (contrarian dampen)."""

    def __init__(self, params=None):
        super().__init__("Carry", params or {"funding_cap": 0.0005, "carry_weight": 0.6})

    def generate_signal(self, market_data):
        funding = market_data.get("funding_rate_8h")
        if funding is None:
            return 0.0, 0.0
        funding = float(funding)
        cap = float(self.params["funding_cap"])
        # Normalize funding into [-1, 1]
        score = float(np.clip(funding / max(cap, 1e-9), -1.0, 1.0))
        confidence = min(abs(funding) / max(cap, 1e-9), 1.0) * 0.6 + 0.2
        return score, confidence


class CrossSectionalMomentumStrategy(BaseStrategy):
    """
    Cross-sectional momentum (VISION §5): compares this asset's return against a
    reference basket (market average). Positive spread = relative strength -> long.
    """

    def __init__(self, params=None):
        super().__init__("Cross-Sectional Momentum",
                         params or {"roc_period": 24, "market_returns": None})

    def generate_signal(self, market_data):
        df = market_data.get("df")
        if df is None or len(df) < self.params["roc_period"] + 5:
            return 0.0, 0.0
        close = df["close"].values
        roc = (close[-1] - close[-self.params["roc_period"]]) / max(close[-self.params["roc_period"]], 1e-9)
        market_ret = self.params.get("market_returns") or market_data.get("market_avg_return")
        if market_ret is None:
            market_ret = 0.0
        spread = roc - float(market_ret)
        score = float(np.clip(spread / 0.05, -1.0, 1.0))
        return score, 0.5


class MultiTimeframeWrapperStrategy(BaseStrategy):
    """Adapts the existing MultiTimeframeConsensus to the BaseStrategy API
    (VISION §5: 'brancher les stratégies mortes')."""

    def __init__(self, db=None, params=None):
        super().__init__("Multi-Timeframe", params or {})
        self.db = db

    def generate_signal(self, market_data):
        df = market_data.get("df")
        if df is None:
            return 0.0, 0.0
        symbol = market_data.get("symbol")
        if not symbol or self.db is None:
            return 0.0, 0.0
        try:
            from strategies.multi_timeframe import MultiTimeframeConsensus
            mtf = MultiTimeframeConsensus()
            # Base signal: short-term momentum direction from the live frame;
            # the MTF consensus then validates it across 1H/4H/1D/1W.
            close = df["close"].values
            roc = (close[-1] - close[-13]) / max(close[-13], 1e-9)
            base_signal = float(np.clip(np.tanh(roc * 20.0), -1.0, 1.0))
            res = mtf.check_consensus(
                symbol=symbol,
                current_price=float(df["close"].iloc[-1]),
                strategy_signal=base_signal,
                db=self.db,
            )
            signal = float(res.get("adjusted_signal", base_signal) or 0.0)
            agreements = int(res.get("agreements", 0) or 0)
            confidence = min(0.3 + 0.2 * agreements, 1.0)
            return signal, confidence
        except Exception:
            return 0.0, 0.0
