from copytrading.manager import CopyTradingManager


def test_copytrading_never_simulates_data():
    """
    The module must be either LIVE with REAL traders or UNAVAILABLE - never fake.
    (Hyperliquid public leaderboard is now the default real source.)
    """
    manager = CopyTradingManager()

    # The strict invariant: no simulated/fake profiles, ever.
    assert manager.status in ("LIVE", "UNAVAILABLE")

    if manager.status == "LIVE":
        # Every trader must be a real on-chain address (0x...) from the public leaderboard
        assert manager.get_ranked_traders(), "LIVE must expose real traders"
        for t in manager.get_ranked_traders():
            assert t.trader_id.startswith("0x"), "trader id must be a real chain address"
            assert t.roi_annual != 0.0
    else:
        assert manager.get_ranked_traders() == []
        ok, msg = manager.start_copying("any_id", 1000.0)
        assert ok is False
        assert "Feature Unavailable" in msg
