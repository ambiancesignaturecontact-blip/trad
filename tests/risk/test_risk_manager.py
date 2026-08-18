import pytest
import numpy as np
from risk.risk_manager import RiskManager

def test_risk_manager_initialization():
    risk = RiskManager()
    assert risk.circuit_breaker_active is False
    assert risk.peak_equity == 100000.0
    assert risk.daily_start_equity == 100000.0

def test_circuit_breakers():
    risk = RiskManager(params={
        'max_daily_drawdown_pct': 0.10, # 10% limit
        'max_total_drawdown_pct': 0.20, # 20% limit
        'max_exposure_per_asset_pct': 0.30,
        'fractional_kelly_multiplier': 0.15
    })
    
    # Set starting values
    risk.set_initial_capital(100.0)  # micro-account -> 18% daily limit

    # 1. Check within normal bounds
    tripped, msg = risk.check_circuit_breaker(95.0) # 5% drawdown
    assert tripped is False

    # 2. Check daily drawdown breach (19% > 18% micro limit)
    tripped, msg = risk.check_circuit_breaker(81.0)
    assert tripped is True
    assert "DAILY DRAWDOWN BREACHED" in msg

def test_position_sizing_safeguards():
    risk = RiskManager()
    
    # Test sizing under standard conditions
    qty = risk.calculate_position_size(
        capital=10000.0,
        atr=200.0,
        current_price=1000.0
    )
    # Volatility size: (10000 * 0.01) / (200 / 1000) = 100 / 0.20 = 500 USD = 0.50 Qty
    assert qty > 0
    assert qty <= (10000.0 * 0.25 / 1000.0) # Enforces maximum exposure limit cap of 25%

def test_order_safety_sanity_checks():
    risk = RiskManager()
    
    # 1. Valid order
    ok, reason = risk.validate_order_safety(1000.0, 1000.0, 1.0, 10000.0)
    assert ok is True
    
    # 2. Price deviation fat-finger rejection
    ok, reason = risk.validate_order_safety(1100.0, 1000.0, 1.0, 10000.0) # 10% deviation (limit is 5%)
    assert ok is False
    assert "Price deviation too high" in reason
    
    # 3. Insufficient capital rejection
    ok, reason = risk.validate_order_safety(1000.0, 1000.0, 20.0, 10000.0) # $20k order on $10k capital
    assert ok is False
    assert "Insufficient capital available" in reason
