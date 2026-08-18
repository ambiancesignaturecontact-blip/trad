"""
LOT 5 — Tests « Actualités / Macro réels + machine à états événementielle »
(PROMPT MAÎTRE : Pilier I).

Vérifie :
 1. Phases AVANT / PENDANT / APRÈS des événements macro (mentalité n°4)
 2. HALT demandé pendant un événement HIGH actif
 3. Sentiment : intensificateurs, négation à distance, pondération des sources
 4. Headlines réelles exposées pour l'API
 5. Endpoints /api/v1/news et /api/v1/macro/override
 6. NEWS_SHOCK_HALT_MINUTES branché (durée du HALT pilotable)
"""
import json
import time

import pytest

from models.macro_calendar import (MacroeconomicCalendarEngine, EVENT_ACTIVE_WINDOW,
                                   EVENT_AFTERMATH_WINDOW, PHASE_FACTORS)
from models.sentiment_analyzer import NewsSentimentAnalyzer, SOURCE_WEIGHTS, INTENSIFIERS


# --------------------------------------------------------------------------- #
# 1. PHASES MACRO
# --------------------------------------------------------------------------- #
class TestMacroPhases:
    def _eng_with_event(self, ts_offset: float):
        """Crée un moteur avec UN événement à ts = now + offset (secondes)."""
        eng = MacroeconomicCalendarEngine(calendar_file="__none__")
        eng.scheduled_events = [{
            "event": "FOMC_RATE_DECISION", "timestamp": time.time() + ts_offset,
            "impact": "HIGH", "source": "test",
        }]
        eng.source_status = "file:test"
        return eng

    def test_approaching_phase(self):
        # événement dans 2h -> APPROACHING, facteur 0.40 (HIGH)
        eng = self._eng_with_event(7200)
        res = eng.check_upcoming_macro_shocks(warning_window_seconds=14400)
        assert res["phase"] == "APPROACHING"
        assert res["scale_reduction_factor"] == PHASE_FACTORS["APPROACHING"]["HIGH"]
        assert res["request_halt"] is False

    def test_active_phase_requests_halt_for_high(self):
        # événement EN COURS (dans 5 min) -> ACTIVE, HALT demandé pour HIGH
        eng = self._eng_with_event(5 * 60)
        res = eng.check_upcoming_macro_shocks(warning_window_seconds=14400)
        assert res["phase"] == "ACTIVE"
        assert res["scale_reduction_factor"] == PHASE_FACTORS["ACTIVE"]["HIGH"]  # 0.20
        assert res["request_halt"] is True

    def test_aftermath_phase(self):
        # événement passé il y a 30 min -> AFTERMATH, retour progressif 0.60
        eng = self._eng_with_event(-30 * 60)
        res = eng.check_upcoming_macro_shocks(warning_window_seconds=14400)
        assert res["phase"] == "AFTERMATH"
        assert res["scale_reduction_factor"] == PHASE_FACTORS["AFTERMATH"]["HIGH"]
        assert res["request_halt"] is False

    def test_no_phase_far_from_event(self):
        # événement dans 3 jours -> pas de phase active
        eng = self._eng_with_event(3 * 86400)
        res = eng.check_upcoming_macro_shocks(warning_window_seconds=14400)
        assert res["phase"] == "NONE"
        assert res["upcoming_shock"] is False
        assert res["scale_reduction_factor"] == 1.0

    def test_medium_impact_active_not_halt(self):
        eng = MacroeconomicCalendarEngine(calendar_file="__none__")
        eng.scheduled_events = [{
            "event": "US_RETAIL_SALES", "timestamp": time.time() + 300,
            "impact": "MEDIUM", "source": "test",
        }]
        res = eng.check_upcoming_macro_shocks(warning_window_seconds=14400)
        assert res["phase"] == "ACTIVE"
        assert res["request_halt"] is False  # seul HIGH demande HALT


# --------------------------------------------------------------------------- #
# 2. SENTIMENT ROBUSTE
# --------------------------------------------------------------------------- #
class TestSentimentRobust:
    def test_negation_at_distance(self):
        an = NewsSentimentAnalyzer()
        # "reports say no crash" -> crash est nié à distance (3 mots avant)
        pos = an.analyze_semantic_context("reports say no crash expected")
        assert pos > 0  # négation à distance transforme le négatif en positif

    def test_negation_adjacent_still_works(self):
        an = NewsSentimentAnalyzer()
        assert an.analyze_semantic_context("crash avoided") > 0

    def test_intensifier_amplifies(self):
        an = NewsSentimentAnalyzer()
        base = an.analyze_semantic_context("bitcoin rally")
        intense = an.analyze_semantic_context("bitcoin record rally")
        assert abs(intense) >= abs(base)  # "record" amplifie

    def test_source_weights_defined(self):
        assert SOURCE_WEIGHTS["google_news"] >= SOURCE_WEIGHTS["reddit"]
        assert set(SOURCE_WEIGHTS) >= {"cryptocompare", "google_news", "alpha_vantage", "reddit"}

    def test_intensifiers_defined(self):
        assert "record" in INTENSIFIERS
        assert INTENSIFIERS["record"] > 1.0

    @pytest.mark.asyncio
    async def test_weighted_average(self, monkeypatch):
        an = NewsSentimentAnalyzer()
        async def fake_crypto():
            return ["Bitcoin rally"] * 5       # poids 0.30
        async def fake_reddit():
            return ["Bitcoin crash"] * 5       # poids 0.15
        async def fake_alpha():
            return []
        async def fake_google():
            return []
        monkeypatch.setattr(an, "fetch_cryptocompare_news", fake_crypto)
        monkeypatch.setattr(an, "fetch_reddit_news", fake_reddit)
        monkeypatch.setattr(an, "fetch_alpha_vantage_news", fake_alpha)
        monkeypatch.setattr(an, "fetch_google_news_rss", fake_google)
        res = await an.get_market_sentiment_index()
        assert res["available"] is True
        # rally (+0.8) pondéré 0.30 vs crash (-0.9) pondéré 0.15 -> positif
        assert res["sentiment_index"] > 0

    @pytest.mark.asyncio
    async def test_headlines_exposed(self, monkeypatch):
        an = NewsSentimentAnalyzer()
        async def fake_google():
            return ["Record rally in crypto markets"]
        async def fake_others():
            return []
        monkeypatch.setattr(an, "fetch_google_news_rss", fake_google)
        monkeypatch.setattr(an, "fetch_cryptocompare_news", fake_others)
        monkeypatch.setattr(an, "fetch_reddit_news", fake_others)
        monkeypatch.setattr(an, "fetch_alpha_vantage_news", fake_others)
        await an.get_market_sentiment_index()
        hl = an.get_recent_headlines()
        assert len(hl) == 1
        assert hl[0]["source"] == "google_news"
        assert "rally" in hl[0]["title"]


# --------------------------------------------------------------------------- #
# 3. INTÉGRATION MAIN.PY
# --------------------------------------------------------------------------- #
class TestMainIntegration:
    def test_state_has_news_fields(self):
        import main
        assert "recent_headlines" in main.STATE
        assert "macro_phase" in main.STATE
        assert "news_shock" in main.STATE

    def test_telemetry_has_news(self):
        import main
        tel = main.compile_telemetry_data()
        assert "recent_headlines" in tel
        assert "macro_phase" in tel
        assert "news_shock" in tel

    def test_macro_override_reduce(self):
        """POST /api/v1/macro/override (reduce) via TestClient."""
        import main
        from fastapi.testclient import TestClient
        from main import app
        with TestClient(app) as c:
            r = c.post("/api/v1/macro/override",
                       json={"action": "reduce", "factor": 0.4})
            assert r.status_code == 200
            body = r.json()
            assert body["ok"] is True
            assert body["factor"] == 0.4
            assert main.STATE["macro_scale_factor_tactile"] == 0.4
            # reset
            r2 = c.post("/api/v1/macro/override", json={"action": "reset"})
            assert r2.json()["ok"] is True
            assert main.STATE["macro_scale_factor_tactile"] == 1.0

    def test_macro_override_halt(self):
        from fastapi.testclient import TestClient
        from main import app, risk_state, RiskStateMachine
        with TestClient(app) as c:
            r = c.post("/api/v1/macro/override", json={"action": "halt"})
            assert r.json()["ok"] is True
            assert risk_state.state == RiskStateMachine.HALT
            # remise à NORMAL
            risk_state.reset(reason="test")

    def test_news_endpoint(self):
        from fastapi.testclient import TestClient
        from main import app
        with TestClient(app) as c:
            r = c.get("/api/v1/news")
            assert r.status_code in (200, 401)  # auth optionnelle
            if r.status_code == 200:
                body = r.json()
                assert "sentiment_index" in body
                assert "headlines" in body
                assert "source_weights" in body

    def test_newshock_halt_minutes_env(self):
        """La durée du HALT news est pilotée par NEWS_SHOCK_HALT_MINUTES."""
        import inspect
        src = inspect.getsource(__import__("main", fromlist=["x"]))
        assert 'os.getenv("NEWS_SHOCK_HALT_MINUTES", "15")' in src
        assert "cooldown_seconds" in src
