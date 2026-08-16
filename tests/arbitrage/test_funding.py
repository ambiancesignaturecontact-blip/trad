import pytest
from models.funding_arbitrage import FundingRateArbitrageEngine

def test_funding_arbitrage_entry_success():
    engine = FundingRateArbitrageEngine(min_funding_threshold=0.0005) # 0.05% threshold
    
    # Profitable opportunity: funding rate is high, perp is trading at premium
    res = engine.analyze_funding_opportunities(
        symbol="BTCUSDT",
        spot_bid=59990.0,
        spot_ask=60000.0,
        perp_bid=60100.0,
        perp_ask=60110.0,
        mark_price=60050.0,
        index_price=60000.0,
        funding_rate_8h=0.0012 # 0.12% per 8h
    )
    
    assert res["action"] == "ENTER_ARBITRAGE"
    assert res["symbol"] == "BTCUSDT"
    assert res["spot_action"] == "BUY"
    assert res["perp_action"] == "SELL_SHORT"
    assert res["real_spread_pct"] > 0

def test_funding_arbitrage_missing_parameters():
    engine = FundingRateArbitrageEngine()
    
    # Missing parameters should immediately trigger fail-safe hold
    res = engine.analyze_funding_opportunities(
        symbol="BTCUSDT",
        spot_bid=59990.0,
        spot_ask=None, # Missing!
        perp_bid=60100.0,
        perp_ask=60110.0,
        mark_price=60050.0,
        index_price=60000.0,
        funding_rate_8h=0.0012
    )
    
    assert res["action"] == "HOLD"
    assert "Data incomplete" in res["reason"]

def test_funding_arbitrage_simplified_signature():
    engine = FundingRateArbitrageEngine(min_funding_threshold=0.0005)
    
    # Check simplified call signature (from main loop)
    res = engine.analyze_funding_opportunities(
        symbol="BTCUSDT",
        spot_price=60000.0,
        perp_price=60100.0,
        funding_rate_8h=0.0012
    )
    
    assert res["action"] == "ENTER_ARBITRAGE"
    assert res["symbol"] == "BTCUSDT"
    assert res["spot_action"] == "BUY"
    assert res["perp_action"] == "SELL_SHORT"
    assert res["real_spread_pct"] > 0
    
pre_init = True
