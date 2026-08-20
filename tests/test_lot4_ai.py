"""
LOT 4 — Tests « IA hiérarchisée + fusion des doublons + régimes certains »
(PROMPT MAÎTRE : Faille 2 + Piliers B, C, D).

Vérifie :
 1. Fusion ai/ -> models/ (source unique de vérité, shims de dépréciation)
 2. Confiance de régime (vraisemblance + stabilité) + validation multi-actifs
 3. MLOps : challenger vs champion HORS-ÉCHANTILLON (walk-forward Purged K-Fold)
 4. Mixture of Experts : contribution réelle au PnL + mise en sommeil
 5. Causal gate : désactivation des signaux non causaux
 6. Alpha contrefactuel généralisé (tous les trades clôturés)
 7. RLHF : fallback NEUTRE (bug corrigé) + GAN : limites documentées
 8. HypothesisGenerator : budget de recherche + gel anti-overfitting
 9. MetaAllocationEngine : pondération par contribution PnL réelle
"""
import inspect

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# 1. FUSION ai/ -> models/
# --------------------------------------------------------------------------- #
class TestFusion:
    def test_ai_reexports_models(self):
        """Les doublons ai/ réexportent la source unique models/."""
        import ai.price_predictor as aip
        import models.price_predictor as mp
        assert aip.LSTMLikePredictor is mp.LSTMLikePredictor
        assert aip.PPOTRAgent is mp.PPOTRAgent
        import ai.regime_detector as air
        import models.regime_detector as mr
        assert air.MarketRegimeDetector is mr.MarketRegimeDetector
        import ai.sentiment_analyzer as ais
        import models.sentiment_analyzer as ms
        assert ais.NewsSentimentAnalyzer is ms.NewsSentimentAnalyzer
        import ai.mlops_pipeline as aim
        import models.mlops_pipeline as mm
        assert aim.MLOpsAutoTrainer is mm.MLOpsAutoTrainer

    def test_shims_marked_deprecated(self):
        src = open("ai/price_predictor.py").read()
        assert "DÉPRÉCIÉ" in src or "DEPRECATED" in src.upper()

    def test_models_are_singular_source(self):
        """La logique vit dans models/ (pas de code dupliqué dans ai/)."""
        src = open("ai/price_predictor.py").read()
        assert "from models.price_predictor import" in src


# --------------------------------------------------------------------------- #
# 2. CONFIANCE DE RÉGIME
# --------------------------------------------------------------------------- #
class TestRegimeConfidence:
    def _df(self, n=120):
        idx = pd.date_range("2026-01-01", periods=n, freq="h")
        close = np.linspace(100, 110, n) + np.sin(np.arange(n) / 5) * 0.5
        return pd.DataFrame({"close": close, "high": close + 1,
                             "low": close - 1, "volume": np.full(n, 10.0)},
                            index=idx)

    def test_regime_confidence_bounds(self):
        from models.regime_detector import MarketRegimeDetector
        det = MarketRegimeDetector()
        df = self._df()
        rets = df["close"].pct_change().dropna().values[-30:]
        vols = np.abs(rets)
        X = np.column_stack([rets, vols])
        res = det.regime_confidence(X)
        assert 0.0 <= res["confidence"] <= 1.0
        assert res["regime_id"] in (0, 1, 2, 3)
        assert 0.0 <= res["stability"] <= 1.0

    def test_validate_on_asset(self):
        from models.regime_detector import MarketRegimeDetector
        det = MarketRegimeDetector()
        df = self._df()
        res = det.validate_on_asset(df, symbol="BTCUSDT")
        assert res is not None
        assert res["symbol"] == "BTCUSDT"
        assert res["n_samples"] >= 20
        assert res["loglik_mean"] <= 0.0  # log-vraisemblance toujours <= 0
        assert res["stability"] >= 0.0

    def test_validate_insufficient_data(self):
        from models.regime_detector import MarketRegimeDetector
        det = MarketRegimeDetector()
        assert det.validate_on_asset(pd.DataFrame(), "SOLUSDT") is None


# --------------------------------------------------------------------------- #
# 3. MLOPS CHALLENGER vs CHAMPION
# --------------------------------------------------------------------------- #
class TestMLOpsOOS:
    def test_evaluate_oos_walkforward(self):
        from models.mlops_pipeline import MLOpsAutoTrainer
        from models.price_predictor import LSTMLikePredictor
        from models.regime_detector import MarketRegimeDetector

        class FakeDB:
            def __init__(self):
                self.s = {}
            def save_setting(self, k, v):
                self.s[k] = v
            def get_setting(self, k, d=""):
                return self.s.get(k, d)
            def add_audit_log(self, *a, **k):
                pass

        trainer = MLOpsAutoTrainer(MarketRegimeDetector(), LSTMLikePredictor(), FakeDB())
        n = 150
        idx = pd.date_range("2026-01-01", periods=n, freq="h")
        close = np.cumsum(np.random.RandomState(42).normal(0, 0.01, n)) + 100
        df = pd.DataFrame({"close": close, "high": close + 1, "low": close - 1,
                           "volume": np.full(n, 10.0), "open": close - 0.5},
                          index=idx)
        res = trainer.evaluate_oos_walkforward(df, n_splits=3)
        assert res is not None
        assert "oos_sharpe_mean" in res
        assert res["n_folds"] >= 2

    def test_deploy_only_if_beats_champion(self):
        from models.mlops_pipeline import MLOpsAutoTrainer
        from models.price_predictor import LSTMLikePredictor
        from models.regime_detector import MarketRegimeDetector

        class FakeDB:
            def __init__(self):
                self.s = {}
            def save_setting(self, k, v):
                self.s[k] = v
            def get_setting(self, k, d=""):
                return self.s.get(k, d)
            def add_audit_log(self, *a, **k):
                pass

        db = FakeDB()
        trainer = MLOpsAutoTrainer(MarketRegimeDetector(), LSTMLikePredictor(), db)
        df = pd.DataFrame()
        # champion = 1.0 ; challenger 0.5 -> écarté
        db.save_setting("mlops_champion_sharpe_lstm", "1.0")
        assert trainer.deploy_challenger_if_beats_champion(df, "lstm", 0.5) is False
        # challenger 2.0 > champion 1.0 -> promu
        assert trainer.deploy_challenger_if_beats_champion(df, "lstm", 2.0) is True
        # pas de champion -> promu
        assert trainer.deploy_challenger_if_beats_champion(df, "hmm", 0.3) is True

    def test_no_zero_slippage_claim(self):
        src = inspect.getsource(__import__("models.mlops_pipeline", fromlist=["x"]))
        assert "auto-deploy in DEMO" not in src  # la promotion n'est plus inconditionnelle


# --------------------------------------------------------------------------- #
# 4. MIXTURE OF EXPERTS
# --------------------------------------------------------------------------- #
class TestMixtureExperts:
    def test_pnl_contribution_and_sleep(self):
        from core.mixture_experts import MixtureOfExperts
        moe = MixtureOfExperts()
        # 10 trades perdants pour "swing" -> mis en sommeil
        for _ in range(10):
            moe.record_pnl_contribution("swing", -0.01)
        sleepy = moe.sleep_useless_experts(min_samples=10, min_contrib_pct=0.0)
        assert "swing" in sleepy
        assert "swing" in moe.sleeping
        # le gate exclut l'expert endormi
        gate = moe.gate(2, 0.001)
        assert gate["swing"] == 0.0
        assert abs(sum(gate.values()) - 1.0) < 1e-6  # re-normalisé

    def test_no_sleep_without_samples(self):
        from core.mixture_experts import MixtureOfExperts
        moe = MixtureOfExperts()
        moe.record_pnl_contribution("scalping", -0.05)  # 1 seul échantillon
        assert moe.sleep_useless_experts(min_samples=10) == []
        assert moe.sleeping == set()

    def test_contribution_report(self):
        from core.mixture_experts import MixtureOfExperts
        moe = MixtureOfExperts()
        moe.record_pnl_contribution("position", 0.02)
        rep = moe.expert_contribution_report()
        assert rep["position"]["n_trades"] == 1
        assert "sleeping" in rep["position"]


# --------------------------------------------------------------------------- #
# 5. CAUSAL GATE
# --------------------------------------------------------------------------- #
class TestCausalGate:
    def test_factor_logic(self):
        import main
        # pas encore analysé -> neutre
        assert main.causal_signal_factor({"causal_analyzed": False}) == 1.0
        # analysé avec parents -> actif
        assert main.causal_signal_factor({"causal_analyzed": True,
                                          "causal_parents": ["momentum_10"]}) == 1.0
        # analysé SANS parent -> réduction (signaux non causaux désactivés)
        assert main.causal_signal_factor({"causal_analyzed": True,
                                          "causal_parents": []}) == 0.5


# --------------------------------------------------------------------------- #
# 6. ALPHA CONTREFACTUEL GÉNÉRALISÉ
# --------------------------------------------------------------------------- #
class TestCounterfactualAlpha:
    def test_record_closed_trade_logs_alpha(self):
        import main
        main.record_open_position("CAUSALTEST", "Momentum", 100.0)
        main.record_closed_trade("CAUSALTEST", 103.0, "SELL")
        # le trade a été consommé (position supprimée)
        assert main.STATE.get("position_strategies", {}).get("CAUSALTEST") is None
        # le win rate de Momentum a été mis à jour
        assert main.win_tracker.samples("Momentum") >= 1


# --------------------------------------------------------------------------- #
# 7. RLHF FALLBACK NEUTRE + GAN LIMITES
# --------------------------------------------------------------------------- #
class TestRLHFandGAN:
    def test_rlhf_fallback_neutral(self):
        from rl.rlhf_reward_model import RLHFRewardModel
        model = RLHFRewardModel()
        # sans torch ou sans entraînement -> None (neutre)
        assert model.predict_reward(np.zeros(10)) is None

    def test_main_handles_none(self):
        """P2-19 (audit §2.4) : main.py n'appelle PLUS le RLHF (ÉDUCATIF)
        pour le sizing — le facteur du pipeline est la constante 1.0."""
        import inspect
        src = inspect.getsource(__import__("main", fromlist=["x"]))
        assert "rlhf_scale=1.0" in src
        assert "rlhf_reward_model.predict_reward" not in src

    def test_gan_documents_limits(self):
        src = open("ai/generative_extreme_scenarios.py").read()
        assert "LIMITES" in src
        assert "EXPÉRIMENTAL" in src.upper() or "EXPERIMENTAL" in src.upper()


# --------------------------------------------------------------------------- #
# 8. HYPOTHESIS GENERATOR : BUDGET + GEL
# --------------------------------------------------------------------------- #
class TestHypothesisBudget:
    def test_budget_and_freeze(self):
        from core.hypothesis_generator import HypothesisGenerator
        hg = HypothesisGenerator()
        # budget disponible
        assert hg.can_run_research()["allowed"] is True
        hg.consume_budget()
        assert hg.can_run_research()["remaining"] == 19
        # gel après trop de promotions
        for _ in range(3):
            hg.register_promotion()
        gate = hg.can_run_research()
        assert gate["allowed"] is False
        assert "gel" in gate["reason"]

    def test_frozen_cycle_returns_status(self):
        from core.hypothesis_generator import HypothesisGenerator
        hg = HypothesisGenerator()
        for _ in range(5):
            hg.register_promotion()
        res = hg.run_research_cycle(pd.DataFrame(), {})
        assert res["status"] == "FROZEN"


# --------------------------------------------------------------------------- #
# 9. META ALLOCATION : PONDÉRATION PNL RÉELLE
# --------------------------------------------------------------------------- #
class TestMetaAllocationPnL:
    def test_update_pnl_attribution(self):
        from strategies.engine import BaseStrategy, MetaAllocationEngine
        s1 = BaseStrategy("Alpha")
        s2 = BaseStrategy("Beta")
        meta = MetaAllocationEngine(strategies=[s1, s2])
        # Alpha gagne 10 fois, Beta perd 10 fois
        for _ in range(10):
            meta.update_pnl_attribution("Alpha", 0.01)
            meta.update_pnl_attribution("Beta", -0.01)
        w = meta.get_strategy_weights()
        assert w["Alpha"] > w["Beta"]
        # les deux restent admissibles (softmax strictement positif)
        assert w["Alpha"] > 0.0 and w["Beta"] > 0.0


# --------------------------------------------------------------------------- #
# 10. PIPELINE : ÉTAPE REGIME CONFIDENCE
# --------------------------------------------------------------------------- #
class TestPipelineRegimeConfidence:
    def test_regime_confidence_in_pipeline(self):
        from core.risk_pipeline import RISK_PIPELINE_ORDER, apply_risk_pipeline
        assert "regime_confidence" in RISK_PIPELINE_ORDER
        assert RISK_PIPELINE_ORDER.index("correlation") < RISK_PIPELINE_ORDER.index("regime_confidence") < RISK_PIPELINE_ORDER.index("order_flow")
        res = apply_risk_pipeline(
            base_qty=100.0, cvar_qty=1000.0, max_asset_qty=1000.0,
            conviction=1.0, risk_state_scale=1.0, regime_confidence_scale=0.5)
        assert res["qty"] == pytest.approx(50.0)

    def test_main_integration(self):
        import main
        assert "regime_confidence" in main.STATE
        assert "hmm_validation" in main.STATE
        tel = main.compile_telemetry_data()
        assert "regime_confidence" in tel
        assert "expert_contribution" in tel
        assert "research_gate" in tel
