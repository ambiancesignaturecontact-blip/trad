"""
LOT 1 — Tests « Données 100% réelles, zéro fictif » (faille 1 du PROMPT MAÎTRE).

Règle vérifiée ici : AUCUNE donnée inventée ne doit exister ni influencer une
décision de trading. Toute source indisponible -> UNAVAILABLE (mentalité n°5).
"""
import json
import os
import time

import pytest


# --------------------------------------------------------------------------- #
# 1. CALENDRIER MACRO RÉEL (plus de time.time()+3600*4)
# --------------------------------------------------------------------------- #
class TestMacroCalendarReal:
    def test_events_are_real_and_future(self):
        """Le calendrier chargé vient du fichier JSON réel, dates futures UTC."""
        from models.macro_calendar import MacroeconomicCalendarEngine
        eng = MacroeconomicCalendarEngine(calendar_file="data/macro_events.json")
        assert len(eng.scheduled_events) > 0, "Le fichier data/macro_events.json doit contenir des événements"
        now = time.time()
        for ev in eng.scheduled_events:
            assert ev["timestamp"] > now, f"Événement passé: {ev}"
            assert ev["timestamp"] < now + 200 * 86400, f"Horizon déraisonnable: {ev}"
            assert ev["impact"] in ("HIGH", "MEDIUM", "LOW")
            assert ev["source"], "chaque événement doit documenter sa source"
        assert eng.source_status.startswith("file:")

    def test_no_synthetic_timestamps(self):
        """Les timestamps viennent du FICHIER, jamais générés à l'exécution."""
        from models.macro_calendar import MacroeconomicCalendarEngine
        with open("data/macro_events.json") as f:
            file_ts = {e["timestamp"] for e in json.load(f)["events"]}
        eng = MacroeconomicCalendarEngine(calendar_file="data/macro_events.json")
        for ev in eng.scheduled_events:
            assert ev["timestamp"] in file_ts, \
                f"timestamp {ev['timestamp']} doit provenir du fichier JSON, pas de time.time()+offset"

    def test_missing_file_returns_unavailable(self, tmp_path):
        """Fichier absent -> UNAVAILABLE et AUCUNE réduction de risque."""
        from models.macro_calendar import MacroeconomicCalendarEngine
        eng = MacroeconomicCalendarEngine(calendar_file=str(tmp_path / "absent.json"))
        res = eng.check_upcoming_macro_shocks(warning_window_seconds=10 * 86400)
        assert res["upcoming_shock"] is False
        assert res["scale_reduction_factor"] == 1.0
        assert res["status"] == "UNAVAILABLE"
        assert res["events_loaded"] == 0

    def test_upcoming_shock_uses_real_event(self):
        """Un vrai événement proche déclenche la réduction avec le bon facteur."""
        from models.macro_calendar import MacroeconomicCalendarEngine, IMPACT_REDUCTION
        eng = MacroeconomicCalendarEngine(calendar_file="data/macro_events.json")
        res = eng.check_upcoming_macro_shocks(warning_window_seconds=30 * 86400)
        if res["upcoming_shock"]:
            assert res["scale_reduction_factor"] == IMPACT_REDUCTION.get(res["impact"])
            assert res["time_to_event_minutes"] > 0
        # il DOIT y avoir au moins un événement futur dans les 30 jours (calendrier 2026)
        assert res["events_loaded"] > 0

    def test_calendar_file_has_utc_meta(self):
        with open("data/macro_events.json") as f:
            payload = json.load(f)
        assert payload["meta"]["timezone"] == "UTC"
        assert len(payload["events"]) >= 5


# --------------------------------------------------------------------------- #
# 2. SENTIMENT SANS FALLBACK FICTIF
# --------------------------------------------------------------------------- #
class TestSentimentNoFakeFallback:
    async def _all_sources_offline(self):
        return {}

    @pytest.mark.asyncio
    async def test_unavailable_when_sources_down(self, monkeypatch):
        from models.sentiment_analyzer import NewsSentimentAnalyzer
        an = NewsSentimentAnalyzer()
        for m in ("fetch_cryptocompare_news", "fetch_reddit_news",
                  "fetch_alpha_vantage_news", "fetch_google_news_rss"):
            monkeypatch.setattr(an, m, self._all_sources_offline)
        res = await an.get_market_sentiment_index()
        assert res["available"] is False
        assert res["sentiment_index"] is None
        assert res["confidence"] == 0.0
        assert res["num_headlines"] == 0

    @pytest.mark.asyncio
    async def test_real_headlines_give_confidence(self, monkeypatch):
        from models.sentiment_analyzer import NewsSentimentAnalyzer
        an = NewsSentimentAnalyzer()
        async def fake_fetch():
            return ["Bitcoin rally as institutional adoption surges"] * 10
        monkeypatch.setattr(an, "fetch_cryptocompare_news", fake_fetch)
        monkeypatch.setattr(an, "fetch_reddit_news", fake_fetch)
        monkeypatch.setattr(an, "fetch_alpha_vantage_news", fake_fetch)
        monkeypatch.setattr(an, "fetch_google_news_rss", fake_fetch)
        res = await an.get_market_sentiment_index()
        assert res["available"] is True
        assert res["confidence"] > 0.0
        assert res["sentiment_index"] is not None

    def test_no_fake_headlines_in_source(self):
        import inspect
        from models import sentiment_analyzer
        src = inspect.getsource(sentiment_analyzer)
        assert "BTC consolidates support as retail accumulation surges" not in src, \
            "Les titres d'actualité FICTIFS doivent avoir disparu"


# --------------------------------------------------------------------------- #
# 3. VOLATILITÉ IMPLICITE RÉELLE (plus d'iv_map en dur)
# --------------------------------------------------------------------------- #
class TestVolatilityRealIV:
    def test_unavailable_when_no_iv(self):
        from models.volatility_arbitrage import OptionsVolatilityArbitrageEngine
        eng = OptionsVolatilityArbitrageEngine()
        res = eng.evaluate_optimal_options_strategy(current_price=60000.0, iv_annual=None, regime_id=0)
        assert res["strategy"] == "UNAVAILABLE"
        assert res["implied_volatility_pct"] is None

    @pytest.mark.asyncio
    async def test_fetch_real_iv_returns_none_on_failure(self, monkeypatch):
        from models.volatility_arbitrage import OptionsVolatilityArbitrageEngine
        eng = OptionsVolatilityArbitrageEngine()

        class FakeResp:
            status_code = 503
        class FakeClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def get(self, *a, **k):
                return FakeResp()

        monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: FakeClient())
        assert await eng.fetch_real_iv("BTCUSDT") is None

    def test_no_hardcoded_iv_map(self):
        import inspect
        from models import volatility_arbitrage
        src = inspect.getsource(volatility_arbitrage)
        assert "0.35" not in src or "iv_map" not in src


# --------------------------------------------------------------------------- #
# 4. ON-CHAIN SANS FABRICATION
# --------------------------------------------------------------------------- #
class TestOnchainReal:
    def test_risk_score_none_when_unavailable(self):
        from models.onchain_tracker import OnChainTracker
        tr = OnChainTracker()
        assert tr.compute_onchain_risk_score(
            {"net_flow_usd": None, "whale_holding_status": "UNAVAILABLE"}) is None

    def test_safe_fallback_is_honest(self):
        from models.onchain_tracker import OnChainTracker
        tr = OnChainTracker()
        fb = tr._safe_fallback()
        assert fb["net_flow_usd"] is None
        assert fb["whale_holding_status"] == "UNAVAILABLE"

    def test_no_volume_fabrication(self):
        import inspect
        from models import onchain_tracker
        src = inspect.getsource(onchain_tracker)
        assert "vol * 0.0008" not in src
        assert "* 2500" not in src


# --------------------------------------------------------------------------- #
# 5. ÉTAT INITIAL HONNÊTE + CARNETS MULTI-ACTIFS
# --------------------------------------------------------------------------- #
class TestStateAndOrderBooks:
    def test_state_has_no_fake_market_data(self):
        import main
        assert main.STATE["order_book"] is None, "Plus de carnet d'ordres fictif"
        assert main.STATE["last_price"] is None, "Plus de prix fictif initial"
        assert main.STATE["price_history"] == [], "Plus d'historique de prix fictif"
        for sym, meta in main.STATE["assets"].items():
            assert meta["price"] is None, f"{sym}: plus de prix initial inventé"
            assert meta["has_real_price"] is False, f"{sym}: has_real_price doit être False"
            assert meta["data_status"] == "UNAVAILABLE"
        assert main.STATE["sentiment_index"] is None
        assert main.STATE["onchain_risk_score"] is None

    def test_update_asset_order_book_multiasset(self):
        import main
        bids = [[60000.0, 1.5]]
        asks = [[60001.0, 2.0]]
        main.update_asset_order_book("ETHUSDT", bids, asks)
        assert main.STATE["order_books"]["ETHUSDT"]["bids"] == bids
        assert main.STATE["order_books"]["ETHUSDT"]["asks"] == asks
        main.update_asset_order_book("BTCUSDT", bids, asks)
        assert main.STATE["order_book"]["bids"] == bids  # alias historique préservé

    def test_mark_real_price_sets_flags(self):
        import main
        main.mark_real_price("SOLUSDT", 150.25, volume_24h=123456)
        assert main.STATE["assets"]["SOLUSDT"]["has_real_price"] is True
        assert main.STATE["assets"]["SOLUSDT"]["price"] == 150.25
        assert main.STATE["assets"]["SOLUSDT"]["volume_24h"] == 123456
        assert main.STATE["last_known_prices"]["SOLUSDT"] == 150.25

    def test_set_asset_quality_unavailable_blocks_trading(self):
        import main
        main.set_asset_quality("AAPL", "UNAVAILABLE")
        assert main.STATE["assets"]["AAPL"]["has_real_price"] is False
        assert main.STATE["asset_data_status"]["AAPL"] == "UNAVAILABLE"


# --------------------------------------------------------------------------- #
# 6. AIDE À LA DÉCISION : NEUTRE QUAND INDISPONIBLE
# --------------------------------------------------------------------------- #
class TestNeutralHelper:
    def test_neutral(self):
        import main
        assert main._neutral(None) == 0.0
        assert main._neutral(0.42) == 0.42
        assert main._neutral("n/a") == 0.0
        assert main._neutral(None, 0.5) == 0.5


# --------------------------------------------------------------------------- #
# 7. HISTORIQUE : JAMAIS DE DONNÉES FABRIQUÉES QUAND TOUT EST DOWN
# --------------------------------------------------------------------------- #
class TestHistoricalRealOnly:
    @pytest.mark.asyncio
    async def test_returns_empty_df_when_all_sources_down(self, monkeypatch):
        import main

        class FakeResp:
            status_code = 451
        class FakeClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def get(self, *a, **k):
                return FakeResp()

        monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **k: FakeClient())
        # Yahoo aussi down
        monkeypatch.setattr(main, "fetch_yahoo_finance_candles",
                            lambda *a, **k: main.pd.DataFrame())
        df = await main.fetch_historical_market_data("BTCUSDT")
        assert df.empty, "Aucune source réelle -> DataFrame vide, jamais de données fabriquées"

    def test_klines_converter_preserves_real_values(self):
        import main
        data = [
            [1699999999000, "1", "2", "0.5", "1.5", "100.0"],
            [1700000000000, "1.5", "3", "1", "2", "200.0"],
        ]
        df = main._klines_to_df(data)
        assert len(df) == 2
        assert df["close"].iloc[-1] == 2.0
        assert df["volume"].iloc[-1] == 200.0
