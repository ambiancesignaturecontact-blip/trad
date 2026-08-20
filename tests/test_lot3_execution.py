"""
LOT 3 — Tests « Exécution honnête + Order Flow réel » (PROMPT MAÎTRE : Faille 4 + Pilier H).

Vérifie :
 1. Order flow réel : delta/CVD, OFI, absorption, cascades de liquidation
 2. Exploitation (a) refus d'entrer contre un flux dominant
 3. Exploitation (b) cascade -> ne pas acheter la panique
 4. Exploitation (c) stop jamais dans une zone de stops évidente (stop hunting)
 5. Exploitation (d) toxicité -> réduction de taille (VPIN/Kyle/delta)
 6. Slicer honnête (plus de « 0 % slippage »)
 7. Bandit d'exécution à epsilon décroissant
 8. Pipeline : étape order_flow dans l'ordre documenté
"""
import time

import pytest

from market_data.order_flow import OrderFlowEngine


# --------------------------------------------------------------------------- #
# 1. INDICATEURS ORDER FLOW
# --------------------------------------------------------------------------- #
class TestOrderFlowIndicators:
    def test_delta_and_cvd(self):
        of = OrderFlowEngine()
        # 10 acheteurs agressifs de 1 BTC + 4 vendeurs de 1 BTC
        for _ in range(10):
            of.update_trade("BTCUSDT", 60000.0, 1.0, "buy")
        for _ in range(4):
            of.update_trade("BTCUSDT", 60000.0, 1.0, "sell")
        delta, vol = of.get_delta("BTCUSDT")
        assert delta == pytest.approx(6.0)
        assert vol == pytest.approx(14.0)
        assert of.get_cvd("BTCUSDT") == pytest.approx(6.0)

    def test_delta_window_expiry(self):
        of = OrderFlowEngine()
        of.update_trade("BTCUSDT", 60000.0, 1.0, "buy")
        # vieux trade hors fenêtre
        of.trades["BTCUSDT"].append((time.time() - 300.0, -1.0, 5.0, 59000.0))
        delta, vol = of.get_delta("BTCUSDT")
        assert delta == pytest.approx(1.0)
        assert vol == pytest.approx(1.0)

    def test_ofi(self):
        of = OrderFlowEngine()
        assert of.compute_ofi("BTCUSDT") is None  # pas de carnet -> neutre
        of.update_book("BTCUSDT", 3.0, 1.0)
        ofi = of.compute_ofi("BTCUSDT")
        assert ofi == pytest.approx(0.5)  # (3-1)/(3+1)

    def test_absorption(self):
        of = OrderFlowEngine()
        # 100 trades de 10 BTC à prix quasi identique (> $300k) -> absorption
        for i in range(100):
            of.update_trade("BTCUSDT", 60000.0 + (i % 5) * 0.1, 10.0, "buy")
        assert of.detect_absorption("BTCUSDT") is True

    def test_no_absorption_small_volume(self):
        of = OrderFlowEngine()
        for _ in range(5):
            of.update_trade("BTCUSDT", 60000.0, 0.001, "buy")
        assert of.detect_absorption("BTCUSDT") is False


# --------------------------------------------------------------------------- #
# 2. LIQUIDATIONS / CASCADE
# --------------------------------------------------------------------------- #
class TestLiquidations:
    def test_cascade_detection(self):
        of = OrderFlowEngine()
        for i in range(5):  # 5 liquidations de $200k chacune dans la fenêtre
            of.update_liquidation("BTCUSDT", "sell", 3.0, 60000.0)
        assert of.liquidation_cascade_active("BTCUSDT") is True
        assert of.wait_cascade_end("BTCUSDT")[0] is True

    def test_no_cascade_small(self):
        of = OrderFlowEngine()
        of.update_liquidation("BTCUSDT", "sell", 0.1, 60000.0)
        assert of.liquidation_cascade_active("BTCUSDT") is False
        assert of.wait_cascade_end("BTCUSDT")[0] is False


# --------------------------------------------------------------------------- #
# 3. EXPLOITATION (a) — FLUX DOMINANT
# --------------------------------------------------------------------------- #
class TestAvoidAgainstFlow:
    def test_refuse_buy_against_selling_wave(self):
        of = OrderFlowEngine()
        for _ in range(10):  # 600k USD de volume -> échantillon significatif
            of.update_trade("BTCUSDT", 60000.0, 1.0, "sell")
        avoid, reason = of.should_avoid_entry("BTCUSDT", "BUY")
        assert avoid is True
        assert "flux" in reason
        # vente dans le même sens -> autorisée
        avoid2, _ = of.should_avoid_entry("BTCUSDT", "SELL")
        assert avoid2 is False

    def test_neutral_without_data(self):
        of = OrderFlowEngine()
        assert of.should_avoid_entry("BTCUSDT", "BUY") == (False, "")


# --------------------------------------------------------------------------- #
# 4. EXPLOITATION (c) — STOP HUNTING
# --------------------------------------------------------------------------- #
class TestStopHunting:
    def test_stop_long_moved_out_of_zone(self):
        of = OrderFlowEngine()
        recent_high, recent_low, atr = 61000.0, 59000.0, 500.0
        # stop calculé à 59500 -> DANS la zone de chasse (sous le plus bas 59000 - 250)
        new_stop = of.adjust_stop_against_hunting(
            "BTCUSDT", 59500.0, "long", recent_high, recent_low, atr)
        # déplacé SOUS la zone (zone_low - 1 ATR = 58250)
        assert new_stop < 59000.0 - 0.5 * atr

    def test_stop_outside_zone_unchanged(self):
        of = OrderFlowEngine()
        # stop à 57000, zone basse 58750 -> déjà hors zone -> inchangé
        new_stop = of.adjust_stop_against_hunting(
            "BTCUSDT", 57000.0, "long", 61000.0, 59000.0, 500.0)
        assert new_stop == pytest.approx(57000.0)

    def test_zone_computation(self):
        of = OrderFlowEngine()
        zh, zl = of.stop_hunting_zone("BTCUSDT", 61000.0, 59000.0, 500.0)
        assert zh == pytest.approx(61250.0)  # high + 0.5 ATR
        assert zl == pytest.approx(58750.0)  # low - 0.5 ATR


# --------------------------------------------------------------------------- #
# 5. EXPLOITATION (d) — TOXICITÉ -> RÉDUCTION DE TAILLE
# --------------------------------------------------------------------------- #
class TestToxicity:
    def test_toxic_delta_reduces(self):
        of = OrderFlowEngine()
        # volume > 100k USD requis pour juger la toxicité (10 BTC x 60000)
        for _ in range(10):
            of.update_trade("BTCUSDT", 60000.0, 1.0, "sell")
        f = of.toxicity_factor("BTCUSDT")
        assert f < 1.0
        assert f >= 0.2

    def test_small_sample_not_toxic(self):
        """Échantillon trop petit (bruit) -> jamais de réduction (mentalité n°20)."""
        of = OrderFlowEngine()
        of.update_trade("BTCUSDT", 60000.0, 0.001, "sell")
        assert of.toxicity_factor("BTCUSDT") == 1.0
        assert of.should_avoid_entry("BTCUSDT", "BUY") == (False, "")

    def test_neutral_without_data(self):
        of = OrderFlowEngine()
        assert of.toxicity_factor("BTCUSDT") == 1.0

    def test_high_vpin_reduces(self):
        of = OrderFlowEngine()
        # pas de trades -> facteur = 1 puis VPIN réduit
        f = of.toxicity_factor("BTCUSDT", vpin=0.80)
        assert f == pytest.approx(0.6)


# --------------------------------------------------------------------------- #
# 6. SLICER HONNÊTE
# --------------------------------------------------------------------------- #
class TestSlicerHonesty:
    def test_no_zero_slippage_claim(self):
        import inspect

        from models import execution_slicer
        src = inspect.getsource(execution_slicer)
        assert "0% slippage" not in src
        assert "0 % slippage" not in src
        assert "réduit" in src or "REDUCE" in src or "reduce" in src


# --------------------------------------------------------------------------- #
# 7. BANDIT EPSILON DÉCROISSANT
# --------------------------------------------------------------------------- #
class TestBanditDecay:
    def test_epsilon_decays(self):
        from core.execution_agent import ExecutionStyleBandit
        b = ExecutionStyleBandit(epsilon=0.15, min_epsilon=0.02, decay_per_obs=0.01)
        assert b.epsilon == pytest.approx(0.15)
        for _ in range(30):
            b.choose_style("BTCUSDT", "normal", 5.0, 0.8)  # n_obs++ via choose
        assert b.epsilon < 0.15
        for _ in range(300):
            b.choose_style("BTCUSDT", "normal", 5.0, 0.8)
        assert b.epsilon == pytest.approx(0.02)  # plancher


# --------------------------------------------------------------------------- #
# 8. PIPELINE : ÉTAPE ORDER FLOW
# --------------------------------------------------------------------------- #
class TestPipelineOrderFlow:
    def test_order_flow_in_pipeline(self):
        from core.risk_pipeline import RISK_PIPELINE_ORDER, apply_risk_pipeline
        assert "order_flow" in RISK_PIPELINE_ORDER
        # position entre correlation et confidence
        assert RISK_PIPELINE_ORDER.index("correlation") < RISK_PIPELINE_ORDER.index("order_flow") < RISK_PIPELINE_ORDER.index("confidence")
        res = apply_risk_pipeline(
            base_qty=100.0, cvar_qty=1000.0, max_asset_qty=1000.0,
            conviction=1.0, risk_state_scale=1.0, order_flow_scale=0.5)
        steps = [s["step"] for s in res["steps"]]
        assert "order_flow" in steps
        assert res["qty"] == pytest.approx(50.0)

    def test_main_integration(self):
        import main
        assert hasattr(main, "order_flow")
        tel = main.compile_telemetry_data()
        assert "order_flow" in tel
        # le slicer n'est plus appelé avec une promesse de slippage nul
        import inspect
        src = inspect.getsource(main)
        assert "achieve 0% slippage" not in src
