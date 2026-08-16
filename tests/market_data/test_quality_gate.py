import pytest
import time
from market_data.quality import MarketDataQuality
from market_data.models import MarketTick
from market_data.base import DataQualityGate

def test_data_quality_gate_valid_tick():
    gate = DataQualityGate(stale_timeout_seconds=5.0)
    
    tick = MarketTick(
        symbol="BTCUSDT",
        exchange="Binance",
        timestamp=time.time(),
        bid=59900.0,
        ask=60100.0,
        last=60000.0,
        volume=1.5
    )
    
    # Valid LIVE tick in DEMO mode should succeed perfectly
    assert gate.validate_tick(tick, "DEMO") is True
    
    # Valid LIVE tick in REAL mode should succeed perfectly
    assert gate.validate_tick(tick, "REAL") is True

def test_data_quality_gate_stale_tick():
    gate = DataQualityGate(stale_timeout_seconds=2.0)
    
    tick = MarketTick(
        symbol="BTCUSDT",
        exchange="Binance",
        timestamp=time.time(),
        bid=59900.0,
        ask=60100.0,
        last=60000.0,
        volume=1.5
    )
    
    # Simulate a stale delay (e.g. wait 2.5 seconds)
    time.sleep(2.5)
    
    # Stale tick should be rejected
    assert gate.validate_tick(tick, "DEMO") is False
    assert tick.quality == MarketDataQuality.STALE
