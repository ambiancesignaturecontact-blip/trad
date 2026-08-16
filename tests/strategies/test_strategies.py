import pytest
import pandas as pd
import numpy as np
from strategies.engine import TrendFollowingStrategy, MeanReversionStrategy, MetaAllocationEngine

def test_trend_following_signal():
    strat = TrendFollowingStrategy()
    
    # Check fallback on empty data
    sig, conf = strat.generate_signal({})
    assert sig == 0.0
    assert conf == 0.0

def test_meta_allocation_dominance():
    # Setup mock strategies
    trend = TrendFollowingStrategy()
    rev = MeanReversionStrategy()
    meta = MetaAllocationEngine(strategies=[trend, rev])
    
    # Verify dominant strategy selection based on regime
    # regime_state_id = 0 (Bull) -> Trend Following gets a boost (+0.40), making it dominate
    res = meta.allocate(market_data={}, regime_state_id=0, ml_prediction=0.0, ppo_action=0.0)
    assert "Trend Following" in res["contributions"]
    assert "Mean Reversion" in res["contributions"]
    assert res["contributions"]["Trend Following"]["weight"] >= 0.40
    assert res["contributions"]["Mean Reversion"]["weight"] <= 0.60
