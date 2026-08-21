"""
LOT 7 (mandat — Execution Intelligence) : Implementation Shortfall consolidé
+ venue quality + comparaison prévision/réalité.

Vérifié ici :
  1. ExecutionIntel.record : IS bps (adverse > 0, BUY/SELL), IS $, forecast
     error (realized − expected), latence/fill ratio optionnels (None si non
     mesurés — jamais inventés), persistance best-effort (db.add_event).
  2. report() : par venue (n, avg IS, latence, fill ratio, forecast error),
     par style, global, dernière mesure.
  3. Convention IS alignée sur ExecutionAlpha (slippage adverse positif).
  4. Câblage main.py : execution_intel.record appelé au fill réel ET au fill
     papier (DEMO == REAL) ; télémétrie execution_intel ; API
     /api/v1/execution-intel.
  5. DÉMO == RÉAL : aucun flag de mode.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from core.execution_intel import ExecutionIntel  # noqa: E402


class FakeDB:
    def __init__(self):
        self.events = []

    def add_event(self, ts, etype, payload):
        self.events.append((etype, json.loads(payload)))


# --------------------------------------------------------------------------- #
# 1. Record : IS, forecast error, persistance
# --------------------------------------------------------------------------- #
class TestRecord:
    def _intel(self):
        return ExecutionIntel()

    def test_buy_adverse_slippage_positive_is(self):
        e = self._intel()
        e.record("Binance", "BTCUSDT", "BUY", 0.1, arrival_price=60000.0,
                 fill_price=60030.0, style="market", expected_slippage_bps=3.0,
                 realized_slippage_bps=5.0)
        s = e.last
        assert s["is_bps"] == pytest.approx((60030 - 60000) / 60000 * 1e4, abs=1e-3)
        assert s["is_usd"] == pytest.approx(30.0 * 0.1, abs=1e-4)
        assert s["forecast_error_bps"] == pytest.approx(2.0, abs=1e-6)  # 5 - 3

    def test_sell_adverse_positive(self):
        e = self._intel()
        e.record("Bybit", "ETHUSDT", "SELL", 1.0, arrival_price=3000.0,
                 fill_price=2985.0, style="market")
        assert e.last["is_bps"] > 0  # vendu plus bas que l'arrival = adverse

    def test_favorable_slippage_negative(self):
        e = self._intel()
        e.record("Binance", "BTCUSDT", "BUY", 0.1, arrival_price=60000.0,
                 fill_price=59970.0, style="limit")
        assert e.last["is_bps"] < 0  # meilleur prix que l'arrival

    def test_latency_and_fill_ratio_optional(self):
        e = self._intel()
        e.record("Binance", "BTCUSDT", "BUY", 0.1, 60000.0, 60010.0, "market")
        assert e.last["latency_ms"] is None
        assert e.last["fill_ratio"] is None
        e.record("Binance", "BTCUSDT", "BUY", 0.1, 60000.0, 60010.0, "market",
                 latency_ms=42.0, fill_ratio=0.5)
        assert e.last["latency_ms"] == 42.0
        assert e.last["fill_ratio"] == 0.5

    def test_persistence_via_db(self):
        e = self._intel()
        db = FakeDB()
        e.record("Binance", "BTCUSDT", "BUY", 0.1, 60000.0, 60030.0, "market",
                 expected_slippage_bps=3.0, realized_slippage_bps=5.0, db=db)
        assert db.events[0][0] == "exec_intel"
        assert db.events[0][1]["is_bps"] > 0
        assert db.events[0][1]["expected_bps"] == 3.0

    def test_never_raises(self):
        e = self._intel()
        e.record(None, "X", "BUY", 0.1, 0.0, 0.0, "market")          # prix 0 -> ignoré
        e.record("V", "X", "BUY", 0.1, None, None, "market")          # None -> ignoré
        e.record("V", "X", "BAD", 0.1, 1.0, 1.1, "market")            # side inconnu
        assert e.report()["n"] >= 0


# --------------------------------------------------------------------------- #
# 2. Rapport : par venue, par style, global
# --------------------------------------------------------------------------- #
class TestReport:
    def test_by_venue_aggregates(self):
        e = ExecutionIntel()
        for i in range(4):
            e.record("Binance", "BTCUSDT", "BUY", 0.1, 60000.0,
                     60000.0 + 10.0 * i, "market", latency_ms=40.0 + i,
                     fill_ratio=1.0)
        for i in range(2):
            e.record("Bybit", "ETHUSDT", "SELL", 1.0, 3000.0,
                     2999.0 + i, "twap")
        r = e.report()
        assert r["n"] == 6
        assert r["by_venue"]["Binance"]["n"] == 4
        assert r["by_venue"]["Binance"]["avg_latency_ms"] == pytest.approx(41.5, abs=0.1)
        assert r["by_venue"]["Binance"]["avg_fill_ratio"] == 1.0
        assert r["by_venue"]["Bybit"]["avg_latency_ms"] is None  # non mesuré

    def test_forecast_error_global(self):
        e = ExecutionIntel()
        for i in range(3):
            e.record("Binance", "BTCUSDT", "BUY", 0.1, 60000.0, 60015.0,
                     "market", expected_slippage_bps=2.0, realized_slippage_bps=5.0)
        r = e.report()
        assert r["avg_forecast_error_bps"] == pytest.approx(3.0, abs=1e-6)

    def test_by_style(self):
        e = ExecutionIntel()
        e.record("V", "S", "BUY", 1.0, 100.0, 100.05, "market")
        e.record("V", "S", "BUY", 1.0, 100.0, 100.05, "twap")
        r = e.report()
        assert r["by_style"]["market"]["n"] == 1
        assert r["by_style"]["twap"]["n"] == 1

    def test_empty_report_honest(self):
        r = ExecutionIntel().report()
        assert r["n"] == 0
        assert r["avg_is_bps"] is None
        assert r["by_venue"] == {}


# --------------------------------------------------------------------------- #
# 3. Câblage main + télémétrie + API
# --------------------------------------------------------------------------- #
class TestWiring:
    def test_main_wiring(self):
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        assert "execution_intel = ExecutionIntel()" in src
        assert "execution_intel.record(_venue" in src          # fill réel
        assert "execution_intel.record(_paper_venue" in src    # fill papier

    def test_telemetry_exposes_execution_intel(self):
        import main  # noqa: F401
        from telemetry import compile_telemetry_data
        tel = compile_telemetry_data()
        assert "execution_intel" in tel
        assert "by_venue" in tel["execution_intel"]

    def test_api_execution_intel(self):
        from fastapi.testclient import TestClient

        import main  # noqa: F401
        with TestClient(main.app) as c:
            r = c.get("/api/v1/execution-intel")
            assert r.status_code == 200
            body = r.json()
            assert "by_venue" in body and "by_style" in body

    def test_no_mode_flag(self):
        import inspect
        src = inspect.getsource(ExecutionIntel.record)
        assert "active_mode" not in src
        assert '"DEMO"' not in src and '"REAL"' not in src

    def test_convention_matches_execution_alpha(self):
        """La convention IS (adverse positif) est alignée sur ExecutionAlpha."""
        from core.execution_router import ExecutionAlpha
        alpha = ExecutionAlpha()
        slip_alpha = alpha.record("S", "BUY", 60000.0, 60030.0, "market")
        e = ExecutionIntel()
        e.record("V", "S", "BUY", 0.1, 60000.0, 60030.0, "market")
        assert e.last["is_bps"] == pytest.approx(slip_alpha, abs=1e-6)
