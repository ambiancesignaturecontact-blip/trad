"""
Basic strategy tests
"""
import numpy as np
import pandas as pd

from strategies.engine import TrendFollowingStrategy


def test_all_strategies_enabled():
    """Verify all 7 strategies are enabled by default"""
    from main import strategies_list

    enabled = [s.enabled for s in strategies_list]
    assert all(enabled), "All strategies must be enabled for institutional trading"

def test_trend_following_signal():
    df = pd.DataFrame({
        'close': np.linspace(100, 120, 50),
        'high': np.linspace(101, 121, 50),
        'low': np.linspace(99, 119, 50),
        'volume': np.random.uniform(10, 50, 50)
    })

    strat = TrendFollowingStrategy()
    market_data = {'df': df}
    signal, conf = strat.generate_signal(market_data)

    assert isinstance(signal, float)
    assert -1.0 <= signal <= 1.0

def test_meta_engine_allocation():
    from main import meta_engine

    df = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 60000,
        'high': np.random.randn(100).cumsum() + 60001,
        'low': np.random.randn(100).cumsum() + 59999,
        'volume': np.random.uniform(10, 100, 100)
    })

    market_data = {
        'df': df,
        'price_primary': 60000,
        'price_secondary': 60005,
        'bids': [[59990, 1.2]],
        'asks': [[60010, 0.8]],
        'inventory': 0.0,
        'max_inventory': 0.01
    }

    result = meta_engine.allocate(market_data, 2, 0.001, 0.0)

    assert "final_signal" in result
    assert isinstance(result["final_signal"], float)
