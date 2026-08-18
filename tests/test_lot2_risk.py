"""
LOT 2 — Tests « RR dynamique unifié + pipeline de risque + NORMAL/CAUTION/HALT »
(PROMPT MAÎTRE : Faille 3 + Piliers F & G).

Vérifie :
 1. UNE seule source de vérité REWARD_RISK_RATIO (sizing = SL/TP = backtest)
 2. Kelly DYNAMIQUE sur win rate réel (plancher 0.45 / plafond 0.65 / lissage)
 3. Filtre d'entrée RR minimal + RR adaptatif volatilité + asymétrie vs coûts
 4. Machine à états NORMAL/CAUTION/HALT (cool-down + redémarrage progressif)
 5. Pipeline de risque UNIFIÉ, ordre documenté, tracé
 6. Backtest cohérent avec le live (plus de valeurs en dur divergentes)
"""
import inspect
import time

import pytest

from core import position_manager
from core.risk_pipeline import (
    REWARD_RISK_RATIO, MIN_REWARD_RISK, MIN_REWARD_RISK_HIGH_VOL,
    ROUND_TRIP_COST_PCT, WIN_RATE_FLOOR, WIN_RATE_CEIL,
    kelly_dynamic, smoothed_win_rate, rr_requirement, rr_net_positive,
    entry_rr_filter, RiskStateMachine, StrategyWinRateTracker,
    apply_risk_pipeline, RISK_PIPELINE_ORDER,
)


# --------------------------------------------------------------------------- #
# 1. SOURCE UNIQUE DE VÉRITÉ DU RR
# --------------------------------------------------------------------------- #
class TestRRUnity:
    def test_position_manager_uses_same_rr(self):
        """SL/TP du position_manager respectent REWARD_RISK_RATIO (plus de
        stops à 1.75-2.0 pendant que le Kelly utilise 1.5)."""
        p = position_manager.PositionProtection("BTCUSDT", 60000.0, 0.1)
        rr_pct = (p.take_price - 60000.0) / (60000.0 - p.stop_price)
        assert rr_pct == pytest.approx(REWARD_RISK_RATIO, rel=1e-6)

        # Version ATR aussi
        p2 = position_manager.PositionProtection("ETHUSDT", 3000.0, 2.0, atr=20.0)
        rr_atr = (p2.take_price - 3000.0) / (3000.0 - p2.stop_price)
        assert rr_atr == pytest.approx(REWARD_RISK_RATIO, rel=1e-6)

    def test_risk_manager_default_uses_same_rr(self):
        from risk.risk_manager import RiskManager
        rm = RiskManager()
        # win_rate=None -> plancher 0.45 ; rr None -> REWARD_RISK_RATIO
        qty = rm.calculate_position_size(100000.0, 200.0, 60000.0)
        assert qty > 0
        # même résultat qu'un appel explicite avec les constantes
        qty2 = rm.calculate_position_size(100000.0, 200.0, 60000.0,
                                          win_rate=WIN_RATE_FLOOR,
                                          reward_risk_ratio=REWARD_RISK_RATIO)
        assert qty == pytest.approx(qty2)

    def test_backtester_uses_same_rr(self):
        """Le backtest mesure la MÊME stratégie que le live (Pilier F, ex. 6)."""
        src = inspect.getsource(__import__("backtester.engine", fromlist=["x"]))
        assert "win_rate=0.55" not in src
        assert "reward_risk_ratio=1.5" not in src
        assert "REWARD_RISK_RATIO" in src
        assert "WIN_RATE_FLOOR" in src


# --------------------------------------------------------------------------- #
# 2. KELLY DYNAMIQUE
# --------------------------------------------------------------------------- #
class TestKellyDynamic:
    def test_floor_and_ceil(self):
        # win rate hors bornes -> ramené à 0.45 / 0.65
        low = kelly_dynamic(0.30)
        assert low <= kelly_dynamic(WIN_RATE_FLOOR)
        high = kelly_dynamic(0.90)
        assert high <= kelly_dynamic(WIN_RATE_CEIL) + 1e-12
        assert kelly_dynamic(0.90) == pytest.approx(kelly_dynamic(0.65))

    def test_none_is_prudent(self):
        assert kelly_dynamic(None) == pytest.approx(kelly_dynamic(0.45))

    def test_monotonic_in_win_rate(self):
        prev = -1
        for wr in (0.40, 0.50, 0.60, 0.70):
            k = kelly_dynamic(wr)
            assert k >= prev
            prev = k

    def test_net_of_costs(self):
        """Kelly réduit du coût aller-retour (mentalité n°2 : edge net des coûts)."""
        k = kelly_dynamic(0.55, reward_risk=1.5)
        assert 0.0 <= k <= 0.15  # fraction max 15 %, jamais de levier fou

    def test_smoothing_converges(self):
        prev = None
        for _ in range(40):
            prev = smoothed_win_rate(prev, 1.0, alpha=0.25)
        assert prev == pytest.approx(1.0, abs=0.02)
        prev = None
        for _ in range(40):
            prev = smoothed_win_rate(prev, 0.0, alpha=0.25)
        assert prev == pytest.approx(0.0, abs=0.02)


# --------------------------------------------------------------------------- #
# 3. FILTRE D'ENTRÉE RR + ADAPTATIF + ASYMÉTRIE
# --------------------------------------------------------------------------- #
class TestRRFilter:
    def test_rr_requirement_adaptive(self):
        assert rr_requirement(None, None) == MIN_REWARD_RISK
        assert rr_requirement(1) == MIN_REWARD_RISK_HIGH_VOL       # bear high vol
        assert rr_requirement(3) == MIN_REWARD_RISK_HIGH_VOL       # erratic
        assert rr_requirement(2) == MIN_REWARD_RISK                # range -> base
        assert rr_requirement(None, 0.05) == MIN_REWARD_RISK_HIGH_VOL  # vol élevée

    def test_rr_net_positive(self):
        # (RR-1) × SL > coûts
        assert rr_net_positive(1.8, 0.03, cost_pct=0.002) is True    # 0.024 > 0.002
        assert rr_net_positive(1.1, 0.01, cost_pct=0.002) is False   # 0.001 < 0.002
        assert rr_net_positive(1.0, 0.05) is False                   # RR <= 1 jamais

    def test_entry_rr_filter(self):
        ok, reason = entry_rr_filter(REWARD_RISK_RATIO, regime_id=2, vol_mean=0.005)
        assert ok
        # en forte volatilité, le RR configuré (1.8) est insuffisant (2.0 requis)
        ok2, reason2 = entry_rr_filter(REWARD_RISK_RATIO, regime_id=1)
        assert ok2 is False
        assert "requis" in reason2


# --------------------------------------------------------------------------- #
# 4. MACHINE À ÉTATS NORMAL / CAUTION / HALT
# --------------------------------------------------------------------------- #
class TestRiskStateMachine:
    def test_transitions(self):
        sm = RiskStateMachine(cooldown_minutes=0.01)
        assert sm.state == "NORMAL"
        assert sm.scale_factor() == 1.0
        sm.enter("CAUTION", "test")
        assert sm.state == "CAUTION"
        assert sm.scale_factor() == 0.5
        sm.enter("HALT", "test2")
        assert sm.state == "HALT"
        assert sm.scale_factor() == 0.0      # HALT = aucun nouvel ordre
        sm.enter("CAUTION", "ignored")       # CAUTION ne dégrade pas un HALT
        assert sm.state == "HALT"

    def test_cooldown_then_progressive_restart(self):
        sm = RiskStateMachine(cooldown_minutes=0.01, restart_stages=[
            (0.25, 0.0), (0.50, 1.0), (1.0, 2.0)])  # étapes en MINUTES
        sm.enter("HALT", "test")
        now = time.time()
        # cool-down pas encore écoulé
        assert sm.tick(now) is False
        assert sm.state == "HALT"
        # cool-down écoulé -> CAUTION, début du redémarrage progressif
        assert sm.tick(now + 2.0) is True
        assert sm.state == "CAUTION"
        assert sm.scale_factor() == pytest.approx(0.25)   # étape 1 (25 %)
        # 1 minute après le restart -> 50 %
        sm.restart_ts = time.time() - 61.0
        assert sm.scale_factor() == pytest.approx(0.50)
        # 2 minutes -> 100 %
        sm.restart_ts = time.time() - 121.0
        assert sm.scale_factor() == pytest.approx(1.0)
        # toutes les étapes écoulées -> retour à NORMAL
        assert sm.tick(now + 2.0 + 180.0) is True
        assert sm.state == "NORMAL"

    def test_reset_operator(self):
        sm = RiskStateMachine()
        sm.enter("HALT", "test")
        assert sm.reset(reason="api") is True
        assert sm.state == "NORMAL"
        assert sm.reset() is False  # déjà NORMAL


# --------------------------------------------------------------------------- #
# 5. PIPELINE UNIFIÉ
# --------------------------------------------------------------------------- #
class TestRiskPipeline:
    def test_order_is_documented(self):
        assert RISK_PIPELINE_ORDER[0] == "cvar_cap"
        assert RISK_PIPELINE_ORDER[1] == "max_asset_cap"
        assert "conviction" in RISK_PIPELINE_ORDER
        assert "risk_state" in RISK_PIPELINE_ORDER
        assert RISK_PIPELINE_ORDER[-1] == "tradability"
        assert len(RISK_PIPELINE_ORDER) >= 10

    def test_caps_first_then_multiplicative(self):
        res = apply_risk_pipeline(
            base_qty=10.0, cvar_qty=5.0, max_asset_qty=8.0, conviction=1.0,
            risk_state_scale=1.0)
        assert res["qty"] == pytest.approx(5.0)  # plafonné par CVaR d'abord
        steps = [s["step"] for s in res["steps"]]
        assert steps.index("cvar_cap") < steps.index("conviction")

    def test_halt_zeroes(self):
        res = apply_risk_pipeline(
            base_qty=10.0, cvar_qty=100.0, max_asset_qty=100.0, conviction=0.5,
            risk_state_scale=0.0)
        assert res["qty"] == 0.0
        assert res["final_scale"] == 0.0

    def test_product_of_factors(self):
        res = apply_risk_pipeline(
            base_qty=100.0, cvar_qty=1000.0, max_asset_qty=1000.0, conviction=0.8,
            risk_state_scale=0.5, news_scale=0.2, macro_scale=0.4,
            onchain_scale=0.5, corr_scale=0.7, confidence_scale=0.9,
            org_scale=0.85, rlhf_scale=0.8, vol_scale=0.75, tradability_scale=0.9)
        expected = 100.0 * 0.8 * 0.5 * 0.2 * 0.4 * 0.5 * 0.7 * 0.9 * 0.85 * 0.8 * 0.75 * 0.9
        assert res["qty"] == pytest.approx(expected)
        assert len(res["steps"]) == len(RISK_PIPELINE_ORDER)

    def test_steps_traced(self):
        res = apply_risk_pipeline(
            base_qty=10.0, cvar_qty=50.0, max_asset_qty=50.0, conviction=0.6,
            risk_state_scale=1.0)
        assert all("step" in s and "qty_after" in s for s in res["steps"])


# --------------------------------------------------------------------------- #
# 6. TRACKER WIN RATE RÉEL
# --------------------------------------------------------------------------- #
class TestWinRateTracker:
    def test_record_and_bounds(self):
        state = {}
        tr = StrategyWinRateTracker(state, alpha=0.5)
        tr.record("Trend Following", +0.03)
        tr.record("Trend Following", -0.02)
        tr.record("Trend Following", +0.01)
        assert tr.samples("Trend Following") == 3
        wr = tr.get("Trend Following")
        assert WIN_RATE_FLOOR <= wr <= WIN_RATE_CEIL
        assert 0.4 <= wr <= 0.7  # 2 gagnants / 3 -> ~0.67 (lissé, borné)

    def test_no_history_is_prudent(self):
        tr = StrategyWinRateTracker({})
        assert tr.get("Inconnue") == WIN_RATE_FLOOR
        assert tr.samples("Inconnue") == 0

    def test_meta_label_warmup(self):
        from core.research_discipline import meta_label_filter
        # sans historique et min_samples=5 -> warm-up autorisé
        assert meta_label_filter("X", {}, counts={"X": 2}, min_samples=5) is True
        # après 5 échantillons -> strict
        assert meta_label_filter("X", {"X": 0.40}, counts={"X": 6}, min_samples=5) is False
        # compatibilité : appel 2 args inchangé
        assert meta_label_filter("A", {"A": 0.60}) is True
        assert meta_label_filter("A", {"A": 0.45}) is False


# --------------------------------------------------------------------------- #
# 7. INTÉGRATION main.py
# --------------------------------------------------------------------------- #
class TestMainIntegration:
    def test_state_has_risk_fields(self):
        import main
        assert "risk_state" in main.STATE
        assert "strategy_trade_counts" in main.STATE
        assert "position_strategies" in main.STATE

    def test_record_open_close_updates_winrate(self):
        import main
        main.record_open_position("TESTBTC", "Momentum", 60000.0)
        assert main.STATE["position_strategies"]["TESTBTC"]["strategy"] == "Momentum"
        main.record_closed_trade("TESTBTC", 62000.0, "SELL")  # +3.3% -> gagnant
        assert main.win_tracker.samples("Momentum") == 1
        assert main.STATE["position_strategies"].get("TESTBTC") is None

    def test_risk_state_in_telemetry(self):
        import main
        tel = main.compile_telemetry_data()
        assert "risk_state" in tel
        assert tel["risk_state"]["state"] in ("NORMAL", "CAUTION", "HALT")
