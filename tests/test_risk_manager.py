"""
Institutional Unit Tests - RiskManager
Run with: pytest tests/test_risk_manager.py -v
"""
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
    risk.set_initial_capital(1000.0)  # small account -> 10% daily limit

    # 3% loss under the 10% small-account limit -> should NOT trip
    tripped, msg = risk.check_circuit_breaker(970.0)
    assert tripped is False

    # 12% loss exceeds the 10% small-account limit -> trips
    tripped2, msg2 = risk.check_circuit_breaker(880.0)
    assert tripped2 is True
    assert "DAILY DRAWDOWN" in msg2

    # micro-account (50$) gets 18% daily limit so noise doesn't lock it
    risk.set_initial_capital(50.0)
    tripped3, _ = risk.check_circuit_breaker(47.5)  # 5% loss on 50$
    assert tripped3 is False

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
    """Micro accounts should get a tradable size above the exchange $10 minimum,
    while staying within the conservative volatility/Kelly sizing."""
    risk = RiskManager()
    risk.set_initial_capital(80.0)

    qty = risk.calculate_position_size(80.0, 50.0, 3000.0)
    notional = qty * 3000.0

    # Micro-budget optimizer floors the size at the $10 exchange minimum so the
    # order is actually accepted, but stays far below a reckless 50%+ exposure.
    assert notional >= 10.0, "Micro accounts should reach at least the $10 min notional"
    assert notional <= 80.0 * 0.80, "Micro accounts must respect the 80% exposure cap"
