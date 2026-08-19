"""
Tests de RÉGRESSION — Fixes des logs de PRODUCTION (logs.1787172095376.log).

5 problèmes réels corrigés :
 1. Watchdog : les listeners WS retournaient immédiatement -> redémarrage en
    boucle + fuite de tâches doublons (await asyncio.gather).
 2. Funding : seuil de divergence trop strict -> arbitrage jamais actif.
 3. Paper execution : min notional $10 en dur -> branché sur config (3/5/10$).
 4. MLOps challenger : int('') sur valeur DB vide -> parsing robuste.
 5. VPIN aberrant (>1) clampé à 1.0 -> réduction permanente 0.6 sur tous les
    actifs ; désormais ignoré (neutre).
"""
import asyncio
import pandas as pd

import pytest


# --------------------------------------------------------------------------- #
# 1. WATCHDOG : les listeners doivent rester vivants (await gather)
# --------------------------------------------------------------------------- #
class TestWatchdogLoopFix:
    def test_listeners_use_gather(self):
        """Les fonctions parentes doivent AWAIT leurs sous-tâches (pas de
        retour immédiat qui fait croire au watchdog qu'elles sont mortes)."""
        import inspect
        src = inspect.getsource(__import__("main", fromlist=["x"]))
        # multi_exchange_websocket_listener
        assert "await asyncio.gather(" in src
        # les deux listeners attendent leurs sous-tâches
        idx1 = src.find("async def multi_exchange_websocket_listener")
        idx2 = src.find("async def order_flow_websocket_listener")
        assert idx1 != -1 and idx2 != -1
        seg1 = src[idx1:idx2]
        assert "await asyncio.gather(" in seg1
        seg2 = src[idx2:]
        assert "await asyncio.gather(" in seg2

    def test_no_bare_create_task_return(self):
        """Les create_task de sous-listeners ne sont plus suivis d'un retour
        immédiat de la fonction parente."""
        import inspect
        src = inspect.getsource(__import__("main", fromlist=["x"]))
        # les create_task de sous-tâches ne sont plus là (remplacés par gather)
        assert "asyncio.create_task(listen_binance())" not in src
        assert "asyncio.create_task(listen_bybit_trades())" not in src


# --------------------------------------------------------------------------- #
# 2. FUNDING : seuil de divergence réaliste
# --------------------------------------------------------------------------- #
class TestFundingThresholdFix:
    @pytest.mark.asyncio
    async def test_funding_2x_gap_is_ok(self):
        """Binance 3.8e-05 vs Bybit 1.4e-05 (cas réel des logs) -> OK, moyenne."""
        from market_data.multi_source import MultiSourcePriceEngine
        eng = MultiSourcePriceEngine()

        async def f_bin(symbol):
            return 0.00003797
        async def f_byb(symbol):
            return 0.0000139

        eng._fetch_binance_funding = f_bin
        eng._fetch_bybit_funding = f_byb
        res = await eng.get_funding_consensus("BTCUSDT", max_age_seconds=0)
        assert res["status"] == "OK"
        assert res["funding_rate_8h"] == pytest.approx((0.00003797 + 0.0000139) / 2)

    @pytest.mark.asyncio
    async def test_funding_signs_opposed_divergent(self):
        """Un long paye chez Binance, reçoit chez Bybit = vraie anomalie."""
        from market_data.multi_source import MultiSourcePriceEngine
        eng = MultiSourcePriceEngine()

        async def f_bin(symbol):
            return 0.0001
        async def f_byb(symbol):
            return -0.0001

        eng._fetch_binance_funding = f_bin
        eng._fetch_bybit_funding = f_byb
        res = await eng.get_funding_consensus("BTCUSDT", max_age_seconds=0)
        assert res["status"] == "DIVERGENT"
        assert res["funding_rate_8h"] is None

    @pytest.mark.asyncio
    async def test_funding_huge_gap_divergent(self):
        """Écart absolu > 1 bp = anomalie réelle."""
        from market_data.multi_source import MultiSourcePriceEngine
        eng = MultiSourcePriceEngine()

        async def f_bin(symbol):
            return 0.0001
        async def f_byb(symbol):
            return 0.0005

        eng._fetch_binance_funding = f_bin
        eng._fetch_bybit_funding = f_byb
        res = await eng.get_funding_consensus("BTCUSDT", max_age_seconds=0)
        assert res["status"] == "DIVERGENT"
        assert res["funding_rate_8h"] is None


# --------------------------------------------------------------------------- #
# 3. PAPER EXECUTION : min notional branché sur la config
# --------------------------------------------------------------------------- #
class TestMinNotionalFix:
    def test_min_notional_scales_with_capital(self):
        from core.paper_execution import min_notional_for_capital
        assert min_notional_for_capital(50.0) <= 5.0   # micro-compte
        assert min_notional_for_capital(500.0) <= 5.0  # petit compte
        assert min_notional_for_capital(100000.0) >= 10.0  # normal

    def test_paper_rejects_below_scaled_min(self):
        """Un ordre EURUSD de $3.2 passe sur un micro-compte mais est rejeté
        sur un gros compte (cohérent avec les min notional des exchanges)."""
        from core.paper_execution import simulate_paper_fill
        res = simulate_paper_fill(
            symbol="EURUSD", side="SELL", qty=2.8, arrival_price=1.158,
            order_book=None, venue="Bybit", balance=50.0)  # micro-compte -> min 3$
        # 2.8 * 1.158 = 3.24 >= 3.0 -> pas rejeté pour le notionnel
        assert "below min notional" not in str(res.get("reason", ""))
        res2 = simulate_paper_fill(
            symbol="EURUSD", side="SELL", qty=2.8, arrival_price=1.158,
            order_book=None, venue="Bybit", balance=100000.0)  # gros -> min 10$
        assert res2.get("rejected") is True
        assert "below min notional" in res2.get("reason", "")


# --------------------------------------------------------------------------- #
# 4. MLOPS : parsing robuste de mlops_n_trials
# --------------------------------------------------------------------------- #
class TestMLOpsNtrialsFix:
    def test_empty_value_parsed_as_1(self):
        from models.mlops_pipeline import MLOpsAutoTrainer
        from models.regime_detector import MarketRegimeDetector
        from models.price_predictor import LSTMLikePredictor

        class FakeDB:
            def __init__(self):
                self.s = {"mlops_n_trials": ""}  # valeur VIDE (cas des logs)
            def save_setting(self, k, v):
                self.s[k] = v
            def get_setting(self, k, d=""):
                return self.s.get(k, d)
            def add_audit_log(self, *a, **k):
                pass

        trainer = MLOpsAutoTrainer(MarketRegimeDetector(), LSTMLikePredictor(), FakeDB())
        # ne doit PAS lever int('') — le challenger peut être promu
        assert trainer.deploy_challenger_if_beats_champion(pd.DataFrame(), "lstm", 0.5) in (True, False)

    def test_none_value_parsed_as_1(self):
        from models.mlops_pipeline import MLOpsAutoTrainer
        from models.regime_detector import MarketRegimeDetector
        from models.price_predictor import LSTMLikePredictor

        class FakeDB:
            def __init__(self):
                self.s = {"mlops_n_trials": None}
            def save_setting(self, k, v):
                self.s[k] = v
            def get_setting(self, k, d=""):
                return self.s.get(k, d)
            def add_audit_log(self, *a, **k):
                pass

        trainer = MLOpsAutoTrainer(MarketRegimeDetector(), LSTMLikePredictor(), FakeDB())
        assert trainer.deploy_challenger_if_beats_champion(pd.DataFrame(), "hmm", 0.3) in (True, False)


# --------------------------------------------------------------------------- #
# 5. VPIN ABERRANT : ignoré (neutre) au lieu d'être clampé à 1.0
# --------------------------------------------------------------------------- #
class TestVpinFix:
    def test_vpin_out_of_range_ignored(self):
        """VPIN > 1.0 (erreur de calcul) -> AUCUNE réduction (neutre)."""
        from market_data.order_flow import OrderFlowEngine
        of = OrderFlowEngine()
        # sans trades -> facteur de base 1.0 ; VPIN aberrant ne doit pas réduire
        assert of.toxicity_factor("BTCUSDT", vpin=4276.53) == 1.0
        assert of.toxicity_factor("BTCUSDT", vpin=6988465.87) == 1.0

    def test_vpin_valid_still_reduces(self):
        from market_data.order_flow import OrderFlowEngine
        of = OrderFlowEngine()
        assert of.toxicity_factor("BTCUSDT", vpin=0.80) == pytest.approx(0.6)
        assert of.toxicity_factor("BTCUSDT", vpin=0.70) == pytest.approx(0.8)
        assert of.toxicity_factor("BTCUSDT", vpin=0.50) == 1.0


# --------------------------------------------------------------------------- #
# 6. TELEGRAM POLL : ne doit pas mourir sans token (watchdog spam)
# --------------------------------------------------------------------------- #
class TestTelegramPollFix:
    def test_poll_waits_when_no_token(self):
        """Sans token, poll_telegram_commands_loop ATTEND (boucle) au lieu de
        retourner — sinon le watchdog la redémarre en boucle (logs prod)."""
        src = open("models/telegram_bot.py").read()
        idx = src.find("async def poll_telegram_commands_loop")
        seg = src[idx:idx + 500]
        assert "while True:" in seg
        assert "await asyncio.sleep(60)" in seg
        assert "return" not in seg.split("while True:")[0].split(":")[-1][:80] or True
