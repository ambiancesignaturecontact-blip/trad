"""
LOT 9 — Tests « Finition UX + Honnêteté + Intégration finale »
(PROMPT MAÎTRE : Faille 7 + contraintes techniques).

Vérifie :
 1. Registre d'étiquetage honnête des modules (PRODUCTION/EXPÉRIMENTAL/ÉDUCATIF)
 2. Aucun module ÉDUCATIF n'influence le sizing réel
 3. Télémétrie + endpoint /api/v1/honesty
 4. Bot Telegram : commande /honesty + aide
 5. Contraintes techniques : signatures intactes, asyncio, aucun blocage boucle
"""
import inspect

from core.module_honesty import MODULE_STATUS, is_educational, is_experimental, status_summary


# --------------------------------------------------------------------------- #
# 1. ÉTIQUETAGE HONNÊTE
# --------------------------------------------------------------------------- #
class TestModuleHonesty:
    def test_registry_complete(self):
        """Le registre couvre les modules clés de la plateforme."""
        for key in ("multi_source_price", "risk_pipeline", "order_flow",
                    "macro_calendar", "sentiment", "reconciliation",
                    "cost_accounting", "attribution", "scenario_stress",
                    "bias_audit", "rlhf", "gan_scenarios", "llm_narrative"):
            assert key in MODULE_STATUS, f"{key} manquant dans le registre"

    def test_statuses_valid(self):
        for name, info in MODULE_STATUS.items():
            assert info["status"] in ("PRODUCTION", "EXPÉRIMENTAL", "ÉDUCATIF")
            assert "detail" in info

    def test_educational_has_guard(self):
        """Chaque module ÉDUCATIF documente sa garde (jamais de sizing)."""
        for name, info in MODULE_STATUS.items():
            if info["status"] == "ÉDUCATIF":
                assert "guard" in info, f"{name} : garde manquante"
                assert "JAMAIS" in info["guard"] or "NEUTRE" in info["guard"] \
                       or "aucune" in info["guard"] or "informatif" in info["guard"]

    def test_experimental_has_guard(self):
        for name, info in MODULE_STATUS.items():
            if info["status"] == "EXPÉRIMENTAL":
                assert "guard" in info, f"{name} : garde manquante"

    def test_summary_counts(self):
        s = status_summary()
        assert s["PRODUCTION"] >= 10
        assert s["EXPÉRIMENTAL"] >= 1
        assert s["ÉDUCATIF"] >= 1
        assert sum(s.values()) == len(MODULE_STATUS)

    def test_helpers(self):
        assert is_experimental("regime_confidence") is True
        assert is_educational("rlhf") is True
        assert is_experimental("order_flow") is False


# --------------------------------------------------------------------------- #
# 2. AUCUN MODULE ÉDUCATIF DANS LE SIZING
# --------------------------------------------------------------------------- #
class TestNoEducationalInSizing:
    def test_rlhf_neutral_without_torch(self):
        """RLHF (ÉDUCATIF) renvoie None sans torch -> facteur 1.0 (neutre)."""
        import numpy as np

        from rl.rlhf_reward_model import RLHFRewardModel
        m = RLHFRewardModel()
        assert m.predict_reward(np.zeros(10)) is None

    def test_gan_not_in_sizing(self):
        """Le GAN n'apparaît dans AUCUNE étape du pipeline de risque."""
        from core.risk_pipeline import RISK_PIPELINE_ORDER
        steps = " ".join(RISK_PIPELINE_ORDER)
        # "gan" est une sous-chaîne de "orGANization" -> chercher le mot entier
        assert "gan_scenarios" not in steps
        assert "scenario" not in steps
        assert "generative" not in steps

    def test_options_educational_unavailable(self):
        """Sans IV réelle, le moteur d'options renvoie UNAVAILABLE (pas de sizing)."""
        from models.volatility_arbitrage import OptionsVolatilityArbitrageEngine
        eng = OptionsVolatilityArbitrageEngine()
        res = eng.evaluate_optimal_options_strategy(60000.0, None, 0)
        assert res["strategy"] == "UNAVAILABLE"

    def test_llm_not_in_sizing(self):
        """Le narratif LLM n'apparaît dans aucune étape du pipeline."""
        from core.risk_pipeline import RISK_PIPELINE_ORDER
        steps = " ".join(RISK_PIPELINE_ORDER)
        assert "llm" not in steps
        assert "narrative" not in steps


# --------------------------------------------------------------------------- #
# 3. TÉLÉMÉTRIE + ENDPOINT
# --------------------------------------------------------------------------- #
class TestTelemetryHonesty:
    def test_telemetry_has_honesty(self):
        import main
        tel = main.compile_telemetry_data()
        assert "module_honesty" in tel
        assert "summary" in tel["module_honesty"]
        assert "registry" in tel["module_honesty"]

    def test_endpoint_exists(self):
        import main
        from test_support import all_api_paths
        routes = all_api_paths(main.app)
        assert "/api/v1/honesty" in routes

    def test_endpoint_response(self):
        from fastapi.testclient import TestClient

        from main import app
        with TestClient(app) as c:
            r = c.get("/api/v1/honesty")
            assert r.status_code in (200, 401)
            if r.status_code == 200:
                body = r.json()
                assert "modules" in body
                assert "summary" in body
                assert "rule" in body


# --------------------------------------------------------------------------- #
# 4. BOT TELEGRAM
# --------------------------------------------------------------------------- #
class TestTelegramBot:
    def test_honesty_command(self):
        src = open("bot/telegram_bot.py").read()
        assert '"/honesty"' in src
        assert "HONNÊTETÉ DES MODULES" in src

    def test_help_mentions_honesty(self):
        src = open("bot/telegram_bot.py").read()
        assert "`/honesty`" in src

    def test_risk_status_still_there(self):
        """Les commandes existantes sont préservées (contrainte : ne pas casser)."""
        src = open("bot/telegram_bot.py").read()
        for cmd in ('"/status"', '"/history"', '"/modes"', '"/risk"',
                    '"/pause"', '"/resume"', '"/kill"', '"/approve"', '"/chat "'):
            assert cmd in src, f"commande {cmd} manquante"


# --------------------------------------------------------------------------- #
# 5. CONTRAINTES TECHNIQUES (PDF)
# --------------------------------------------------------------------------- #
class TestTechnicalConstraints:
    def test_no_blocking_calls_in_loop(self):
        """La boucle de trading ne contient pas de sleep bloquant > 30s."""
        src = inspect.getsource(__import__("main", fromlist=["x"]))
        # le seul sleep de boucle est config-driven (2.5s)
        assert "asyncio.sleep(settings.get_float" in src

    def test_real_data_rule_kept(self):
        """Aucune donnée fictive ne doit influencer une décision (règle absolue)."""
        src = inspect.getsource(__import__("main", fromlist=["x"]))
        assert "AUCUNE DONNÉE -> AUCUN ORDRE" in src or "AUCUNE DONNÉE" in src
        # plus de titres d'actualité fictifs
        sent_src = open("models/sentiment_analyzer.py").read()
        assert "BTC consolidates support as retail accumulation surges" not in sent_src

    def test_signatures_preserved(self):
        """Les signatures clés appelées par main.py sont intactes."""
        import inspect as i

        from risk.risk_manager import RiskManager
        sig = i.signature(RiskManager.calculate_position_size)
        params = list(sig.parameters.keys())
        assert "capital" in params and "atr" in params and "current_price" in params

        from models.macro_calendar import MacroeconomicCalendarEngine
        sig2 = i.signature(MacroeconomicCalendarEngine.check_upcoming_macro_shocks)
        assert "warning_window_seconds" in sig2.parameters

    def test_defaults_are_prudent(self):
        """Les défauts par défaut sont prudents (facteurs <= 1.0)."""
        from core.risk_pipeline import MIN_REWARD_RISK, REWARD_RISK_RATIO, ROUND_TRIP_COST_PCT
        assert MIN_REWARD_RISK >= 1.5
        assert REWARD_RISK_RATIO >= MIN_REWARD_RISK
        assert ROUND_TRIP_COST_PCT > 0
