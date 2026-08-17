import pytest
from core.tax_compliance import TaxComplianceEngine
from datetime import datetime, timedelta

def test_fifo_realized_pnl():
    engine = TaxComplianceEngine(journal_path=":memory:")
    
    # Buy 2 BTC @ 60000
    engine.record_trade("BTCUSDT", "BUY", 2.0, 60000, datetime.now())
    
    # Sell 1 BTC @ 65000
    engine.record_trade("BTCUSDT", "SELL", 1.0, 65000, datetime.now())
    
    pnl = engine.get_realized_pnl("BTCUSDT")
    assert pnl == 5000.0

def test_cost_basis_calculation():
    engine = TaxComplianceEngine(journal_path=":memory:")
    
    engine.record_trade("ETHUSDT", "BUY", 5.0, 3000, datetime.now())
    engine.record_trade("ETHUSDT", "BUY", 3.0, 3200, datetime.now())
    
    cost = engine.get_cost_basis("ETHUSDT")
    assert cost == 24600.0  # (5*3000) + (3*3200)
