"""
LOT 6 — Tests « Portefeuille top-down + cycle de vie des positions »
(PROMPT MAÎTRE : Piliers L & M).

Vérifie :
 PILIER L : budget top-down (réserve cash + vol + CVaR), diversification réelle
            entre stratégies (pénalité redondance), capacité (1% volume 24h),
            rebalancing périodique.
 PILIER M : time stop, breakeven stop, scaling out (partial TP), pyramiding
            contrôlé (jamais de moyenne à la baisse), netting.
"""
import time

import numpy as np
import pytest

from core.portfolio_allocator import CASH_RESERVE_PCT, REDUNDANT_CORR, PortfolioAllocator
from core.position_manager import (
    PositionProtection,
    PositionProtectionStore,
    apply_breakeven_stop,
    can_pyramid,
    evaluate_protection,
    evaluate_time_stop,
    partial_take_profit,
)


# --------------------------------------------------------------------------- #
# PILIER L — PORTEFEUILLE TOP-DOWN
# --------------------------------------------------------------------------- #
class TestPortfolioAllocator:
    def test_cash_reserve_obligatoire(self):
        """Jamais 100% investi : réserve de cash 15% par défaut."""
        pa = PortfolioAllocator()
        b = pa.total_risk_budget(100000.0)
        assert b["investable"] == pytest.approx(85000.0)
        assert b["cash_reserve_usd"] == pytest.approx(15000.0)
        assert b["cash_reserve_pct"] == CASH_RESERVE_PCT

    def test_vol_targeting_reduces_budget(self):
        pa = PortfolioAllocator()
        b = pa.total_risk_budget(100000.0, realized_vol_annual=0.20)  # vol 2x cible
        assert b["vol_scale"] == pytest.approx(0.5)
        assert b["budget"] < 85000.0

    def test_cvar_reduces_budget(self):
        pa = PortfolioAllocator()
        b = pa.total_risk_budget(100000.0, portfolio_cvar_pct=0.08)  # CVaR élevé
        assert b["cvar_scale"] < 1.0
        assert b["budget"] < 85000.0

    def test_capacity_cap(self):
        """Taille max = 1% du volume 24h réel (mentalité n°11)."""
        pa = PortfolioAllocator()
        # volume 24h = 20 000 BTC (unités d'actif), participation 1% -> 200 BTC
        cap = pa.capacity_cap_qty("BTCUSDT", 20000.0, 60000.0)
        assert cap == pytest.approx(200.0)
        # sans volume réel -> None (pas de plafond fabriqué)
        assert pa.capacity_cap_qty("XAUUSD", None, 4000.0) is None

    def test_exposure_factor_reserve(self):
        pa = PortfolioAllocator()
        # aucune position -> facteur 1.0
        state = {"cached_positions": [], "assets": {}, "last_known_prices": {},
                 "current_equity": 100000.0}
        assert pa.portfolio_exposure_factor(state, "balance_demo") == 1.0
        # 85% investi (= max) -> facteur 0.0 (plus de cash pour de nouveaux trades)
        state2 = {"cached_positions": [{"symbol": "BTCUSDT", "qty": 1.5}],
                  "assets": {"BTCUSDT": {"price": 56000.0}},
                  "last_known_prices": {}, "current_equity": 100000.0}
        # 1.5 * 56000 = 84000 / 100000 = 84% < 85% -> petit facteur
        f = pa.portfolio_exposure_factor(state2, "balance_demo")
        assert 0.0 <= f < 1.0

    def test_strategy_diversification_penalty(self):
        pa = PortfolioAllocator()
        # 2 stratégies identiques -> corrélation 1.0 -> pénalité forte
        div = pa.strategy_diversification({"Alpha": [0.01] * 30, "Beta": [0.01] * 30},
                                          min_samples=20)
        assert div["Alpha"]["max_corr"] > REDUNDANT_CORR
        assert div["Alpha"]["weight_penalty"] < 1.0
        # stratégies décorrélées -> pas de pénalité
        rng = np.random.RandomState(0)
        a = list(rng.normal(0, 0.01, 30))
        b = list(rng.normal(0, 0.01, 30))
        div2 = pa.strategy_diversification({"Alpha": a, "Beta": b}, min_samples=20)
        assert div2["Alpha"]["weight_penalty"] == 1.0

    def test_rebalance_periodic(self):
        pa = PortfolioAllocator(rebalance_hours=24.0)
        assert pa.should_rebalance() is True  # jamais rebalancé -> oui
        state = {}
        pa.rebalance(state, 100000.0, portfolio_cvar_pct=0.03)
        assert pa.should_rebalance() is False  # vient de rebalancer
        assert "portfolio_allocation" in state
        assert "total_risk_budget" in state["portfolio_allocation"]
        assert state["portfolio_allocation"]["cash_reserve_pct"] == CASH_RESERVE_PCT


# --------------------------------------------------------------------------- #
# PILIER M — CYCLE DE VIE DES POSITIONS
# --------------------------------------------------------------------------- #
class TestPositionLifecycle:
    def _prot(self, entry=100.0, qty=1.0, atr=None):
        return PositionProtection("TEST", entry, qty, stop_loss_pct=0.03,
                                  take_profit_pct=0.054, atr=atr)

    def test_time_stop(self):
        prot = self._prot()
        prot.entry_ts = time.time() - 30 * 3600  # 30h -> trop vieux
        # sans profit suffisant (0.05% < min 0.1%) -> TIME_STOP
        assert evaluate_time_stop(prot, 100.05, 1.0, max_age_hours=24.0) == "TIME_STOP"
        # avec profit >= min -> HOLD
        assert evaluate_time_stop(prot, 102.0, 1.0, max_age_hours=24.0) == "HOLD"
        # position jeune -> HOLD
        prot.entry_ts = time.time()
        assert evaluate_time_stop(prot, 99.0, 1.0, max_age_hours=24.0) == "HOLD"

    def test_breakeven_stop(self):
        prot = self._prot()
        # pas encore en gain -> pas de breakeven
        assert apply_breakeven_stop(prot, 100.5, 1.0, trigger_pct=0.02) is False
        # gain >= 2% -> stop remonté à l'entrée
        assert apply_breakeven_stop(prot, 102.5, 1.0, trigger_pct=0.02) is True
        assert prot.stop_price == pytest.approx(100.0)  # prix d'entrée
        assert prot.breakeven_done is True
        # une seule fois
        assert apply_breakeven_stop(prot, 105.0, 1.0, trigger_pct=0.02) is False

    def test_partial_take_profit(self):
        prot = self._prot(entry=100.0)  # TP = 100 * 1.054 = 105.4
        # TP1 = 100 + (105.4-100)*0.5 = 102.7
        res = partial_take_profit(prot, 102.0, 1.0)  # pas encore
        assert res["action"] == "HOLD"
        res = partial_take_profit(prot, 103.0, 1.0)  # >= 102.7
        assert res["action"] == "PARTIAL_TP"
        assert res["exit_qty"] == pytest.approx(0.5)
        assert res["remain_qty"] == pytest.approx(0.5)
        # une seule fois
        res2 = partial_take_profit(prot, 105.0, 1.0)
        assert res2["action"] == "HOLD"

    def test_pyramiding_only_winners(self):
        prot = self._prot(entry=100.0)
        # position PERDANTE -> interdiction (moyenne à la baisse, mentalité n°12)
        ok, reason = can_pyramid(prot, 99.0, 1.0, reward_risk=1.8)
        assert ok is False
        assert "perdante" in reason
        # position gagnante mais RR dégradé -> refus
        ok2, reason2 = can_pyramid(prot, 101.0, 1.0, reward_risk=1.8, min_rr=3.0)
        assert ok2 is False
        # position gagnante, RR OK (1.33 > 1.2) -> autorisé
        ok3, reason3 = can_pyramid(prot, 100.6, 1.0, reward_risk=1.8, min_rr=1.2)
        assert ok3 is True
        # max ajouts
        ok4, _ = can_pyramid(prot, 100.6, 1.0, reward_risk=1.8, min_rr=1.2,
                             max_additions=2, additions=2)
        assert ok4 is False

    def test_store_persists_lifecycle(self):
        state = {}
        store = PositionProtectionStore(state)
        prot = self._prot()
        prot.breakeven_done = True
        prot.tp1_hit = True
        store.upsert(prot)
        loaded = store.get("TEST")
        assert loaded.breakeven_done is True
        assert loaded.tp1_hit is True
        assert loaded.entry_ts > 0

    def test_evaluate_protection_still_works(self):
        """Compatibilité : l'ancienne API SL/TP intacte."""
        prot = self._prot(entry=100.0, qty=1.0)
        assert evaluate_protection(prot, 96.0, 1.0) == "STOP_LOSS"   # -3% + marge
        prot2 = self._prot(entry=100.0, qty=1.0)
        assert evaluate_protection(prot2, 106.0, 1.0) == "TAKE_PROFIT"


# --------------------------------------------------------------------------- #
# INTÉGRATION MAIN
# --------------------------------------------------------------------------- #
class TestMainIntegration:
    def test_state_has_portfolio_fields(self):
        import main
        assert "portfolio_allocation" in main.STATE
        assert "strategy_diversification" in main.STATE
        assert "position_pyramids" in main.STATE

    def test_telemetry_has_portfolio(self):
        import main
        tel = main.compile_telemetry_data()
        assert "portfolio_allocation" in tel
        assert "strategy_diversification" in tel
        assert "position_pyramids" in tel

    def test_pipeline_has_capacity_and_cash(self):
        from core.risk_pipeline import RISK_PIPELINE_ORDER, apply_risk_pipeline
        assert "capacity" in RISK_PIPELINE_ORDER
        assert "cash_reserve" in RISK_PIPELINE_ORDER
        assert RISK_PIPELINE_ORDER.index("capacity") < RISK_PIPELINE_ORDER.index("order_flow")
        # cash_reserve à 0 -> pas de trade
        res = apply_risk_pipeline(
            base_qty=100.0, cvar_qty=1000.0, max_asset_qty=1000.0,
            conviction=1.0, risk_state_scale=1.0, cash_reserve_scale=0.0)
        assert res["qty"] == 0.0
        # capacity à 0.5 -> moitié
        res2 = apply_risk_pipeline(
            base_qty=100.0, cvar_qty=1000.0, max_asset_qty=1000.0,
            conviction=1.0, risk_state_scale=1.0, capacity_scale=0.5)
        assert res2["qty"] == pytest.approx(50.0)

    def test_rebalance_on_startup(self):
        """Au premier tick, le rebalance portfolio s'exécute (last_rebalance=0)."""
        import main
        assert main.portfolio_allocator.should_rebalance() is True
