"""
Institutional Unit Tests - RiskManager
Run with: pytest tests/test_risk_manager.py -v
"""
import pytest
from risk.risk_manager import RiskManager

def test_position_size_micro_capital():
    """Test that micro capital (50$) can still generate trades"""
    risk = RiskManager()
    risk.set_initial_capital(50.0)
    
    qty = risk.calculate_position_size(
        capital=50.0,
        atr=100.0,
        current_price=60000.0
    )
    
    # Should return a positive quantity even with tiny capital
    assert qty > 0, "Micro capital should still allow trading"
    assert qty * 60000 > 2.0, "Trade notional should be > $2"

def test_circuit_breaker_daily_drawdown():
    risk = RiskManager()
    risk.set_initial_capital(1000.0)
    
    # Simulate 3% daily loss (should trigger with 2.5% limit)
    tripped, msg = risk.check_circuit_breaker(970.0)
    
    assert tripped is True
    assert "DAILY DRAWDOWN" in msg

def test_validate_order_safety():
    risk = RiskManager()
    
    ok, reason = risk.validate_order_safety(
        order_price=60000,
        mid_market_price=60000,
        order_qty=0.001,
        capital_available=1000
    )
    
    assert ok is True

def test_micro_exposure_limit():
    """Micro accounts should be allowed higher exposure"""
    risk = RiskManager()
    risk.set_initial_capital(80.0)
    
    qty = risk.calculate_position_size(80.0, 50.0, 3000.0)
    notional = qty * 3000.0
    
    # Should allow up to ~80% exposure for micro accounts
    assert notional > 50.0, "Micro accounts should get decent exposure"