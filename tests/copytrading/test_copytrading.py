import pytest
from copytrading.manager import CopyTradingManager

def test_copytrading_unavailable_by_default():
    manager = CopyTradingManager()
    
    # By default, without specific private keys, the real API should fail or be offline,
    # leading strictly to UNAVAILABLE status!
    assert manager.status == "UNAVAILABLE"
    assert manager.status_message == "Real trader data unavailable - No simulation"
    assert manager.get_ranked_traders() == []
    
    # Starting copy should fail
    ok, msg = manager.start_copying("any_id", 1000.0)
    assert ok is False
    assert "Feature Unavailable" in msg
