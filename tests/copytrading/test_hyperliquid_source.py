"""
Copy Trading - real public leaderboard source (Hyperliquid).
Tests the parsing/filtering logic deterministically (mocked HTTP).
"""
import pytest

from copytrading.manager import CopyTradingManager

SAMPLE_LEADERBOARD = {
    "leaderboardRows": [
        {
            "ethAddress": "0x1111111111111111111111111111111111111111",
            "accountValue": "5000000.0",
            "displayName": "WhaleAlpha",
            "windowPerformances": [
                ["day", {"pnl": "10000", "roi": "0.002", "vlm": "5000000"}],
                ["week", {"pnl": "70000", "roi": "0.014", "vlm": "35000000"}],
                ["month", {"pnl": "300000", "roi": "0.06", "vlm": "150000000"}],
                ["allTime", {"pnl": "1200000", "roi": "0.24", "vlm": "600000000"}],
            ],
        },
        {
            "ethAddress": "0x2222222222222222222222222222222222222222",
            "accountValue": "800.0",  # below $10k floor -> must be filtered out
            "displayName": "DustAccount",
            "windowPerformances": [
                ["day", {"pnl": "50", "roi": "0.06", "vlm": "1000"}],
                ["month", {"pnl": "1200", "roi": "1.5", "vlm": "5000"}],
            ],
        },
        {
            "ethAddress": "0x3333333333333333333333333333333333333333",
            "accountValue": "2500000.0",
            "displayName": "RoiAnomaly",
            "windowPerformances": [
                ["day", {"pnl": "250000", "roi": "0.1", "vlm": "1000000"}],
                ["month", {"pnl": "8000000", "roi": "6.2", "vlm": "4000000"}],  # >500% -> outlier, filtered
            ],
        },
        {
            "ethAddress": "0x4444444444444444444444444444444444444444",
            "accountValue": "1200000.0",
            "displayName": "SteadyEddie",
            "windowPerformances": [
                ["day", {"pnl": "-500", "roi": "-0.0004", "vlm": "200000"}],
                ["month", {"pnl": "36000", "roi": "0.03", "vlm": "9000000"}],
            ],
        },
    ]
}


@pytest.fixture
def manager_with_hyperliquid(monkeypatch):
    class FakeResp:
        status_code = 200
        text = __import__("json").dumps(SAMPLE_LEADERBOARD)
        def json(self):
            return SAMPLE_LEADERBOARD

    def fake_get(url, timeout=None):
        assert url.startswith("https://stats-data.hyperliquid.xyz")
        return FakeResp()

    import httpx as _httpx
    monkeypatch.setattr(_httpx, "get", fake_get)
    m = CopyTradingManager()
    return m


def test_hyperliquid_source_loads_real_traders(manager_with_hyperliquid):
    m = manager_with_hyperliquid
    assert m.status == "LIVE"
    # Dust account (<$10k) and ROI anomaly (>500%/month) must be filtered out
    ids = {t.trader_id for t in m.get_ranked_traders()}
    assert "0x1111111111111111111111111111111111111111" in ids
    assert "0x4444444444444444444444444444444444444444" in ids
    assert "0x2222222222222222222222222222222222222222" not in ids  # dust
    assert "0x3333333333333333333333333333333333333333" not in ids  # outlier


def test_hyperliquid_ranking_is_real(manager_with_hyperliquid):
    m = manager_with_hyperliquid
    ranked = m.get_ranked_traders()
    # Both real traders present, sorted by risk-adjusted score
    assert len(ranked) == 2
    # Annualized ROI derived from real monthly ROI (0.06 -> 0.72 = 72%)
    whale = next(t for t in ranked if t.trader_id.startswith("0x1111"))
    assert abs(whale.roi_annual - 0.06 * 12.0) < 1e-6
    assert whale.pnl_month == 300000.0
    assert whale.account_value == 5000000.0
