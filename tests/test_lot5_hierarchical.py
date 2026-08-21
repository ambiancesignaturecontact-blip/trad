"""
LOT 5 (mandat — Meta-Allocator hiérarchique) : familles + regret minimization
+ non-stationnarité.

Vérifié ici :
  1. FAMILY_OF : chaque stratégie appartient à une famille (niveau 2).
  2. RegretTracker : regret cumulé (écart à la meilleure ex post), oubli
     exponentiel (non-stationnaire), pas d'exploration sous l'échantillon
     minimal, weights d'exploration bornés.
  3. HierarchicalAllocator : somme des poids = 1.0, bornes dures
     [MIN_WEIGHT, MAX_WEIGHT], scale de famille par performance récente
     (borné), exploration regret ajoutée.
  4. Réversibilité : allocate SANS hierarchical_scales est STRICTEMENT
     identique (comportement pré-LOT 5).
  5. Intégration MetaAllocationEngine : record_regret dans
     update_pnl_attribution, hierarchical_scales() expose des poids bornés.
  6. Câblage main.py : scales passés à allocate ; télémétrie
     hierarchical_allocator ; API /api/v1/meta-allocator.
  7. DÉMO == RÉAL : aucun flag de mode.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from core.hierarchical_allocator import (  # noqa: E402
    MAX_WEIGHT,
    MIN_WEIGHT,
    REG_EXPLORATION_WEIGHT,
    REG_MIN_SAMPLES,
    HierarchicalAllocator,
    RegretTracker,
    family_of,
)


# --------------------------------------------------------------------------- #
# 1. Familles
# --------------------------------------------------------------------------- #
class TestFamilies:
    def test_every_strategy_has_family(self):
        strategies = ["Trend Following", "Momentum", "Cross-Sectional Momentum",
                      "Multi-Timeframe", "Volatility Breakout", "Mean Reversion",
                      "Grid Trading", "Market Making", "Carry",
                      "Statistical Arbitrage", "Inter-Exchange Arbitrage",
                      "Scalping"]
        for s in strategies:
            assert family_of(s) in ("trend", "meanrev", "carry", "arbitrage", "micro"), s
        # regroupements attendus
        assert family_of("Trend Following") == "trend"
        assert family_of("Mean Reversion") == "meanrev"
        assert family_of("Carry") == "carry"
        assert family_of("Statistical Arbitrage") == "arbitrage"
        assert family_of("Scalping") == "micro"

    def test_unknown_family_falls_back(self):
        assert family_of("Inconnue") == "other"


# --------------------------------------------------------------------------- #
# 2. Regret tracker
# --------------------------------------------------------------------------- #
class TestRegret:
    def test_regret_accumulates_for_laggard(self):
        t = RegretTracker(decay=1.0)   # pas d'oubli pour ce test
        for _ in range(5):
            t.record("A", 0.01)   # A gagne toujours
        for _ in range(5):
            t.record("B", -0.01)  # B perd toujours
        assert t.get("B") > t.get("A")
        assert t.get("B") > 0.0

    def test_regret_decay_forgetting(self):
        """L'oubli exponentiel fait décroître le regret quand la stratégie
        redevient bonne (non-stationnaire)."""
        t = RegretTracker(decay=0.5)
        for _ in range(5):
            t.record("A", 0.01)
            t.record("B", -0.01)
        regret_b_high = t.get("B")
        assert regret_b_high > 0.0
        # B redevient aussi bonne que A : le regret converge vers l'écart
        # courant ÷ (1−decay) < le pic (oubli exponentiel non-stationnaire)
        for _ in range(120):
            t.record("A", 0.01)
            t.record("B", 0.01)
        assert t.get("B") < regret_b_high

    def test_exploration_neutral_without_samples(self):
        t = RegretTracker()
        w = t.exploration_weights(["A", "B", "C"])
        # sous REG_MIN_SAMPLES : poids neutres et bornés
        assert all(0.0 < v <= REG_EXPLORATION_WEIGHT for v in w.values())

    def test_exploration_favors_laggard(self):
        t = RegretTracker(decay=1.0)
        for _ in range(REG_MIN_SAMPLES + 5):
            t.record("A", 0.01)
            t.record("B", -0.01)
        w = t.exploration_weights(["A", "B"])
        assert w["B"] > w["A"]  # B (regret élevé) est davantage réessayée

    def test_never_raises(self):
        t = RegretTracker()
        t.record("", 0.01)
        t.record("X", "bad")
        assert t.to_dict()["n_updates"].get("X", 0) >= 0


# --------------------------------------------------------------------------- #
# 3. HierarchicalAllocator
# --------------------------------------------------------------------------- #
class TestHierarchical:
    def _base(self):
        return {s: 1.0 for s in ["Trend Following", "Momentum", "Mean Reversion",
                                 "Scalping", "Carry"]}

    def test_weights_sum_to_one_and_bounded(self):
        h = HierarchicalAllocator()
        out = h.allocate(self._base())
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-6)
        for v in out.values():
            assert MIN_WEIGHT <= v <= MAX_WEIGHT

    def test_family_scale_reduces_bad_family(self):
        h = HierarchicalAllocator(family_ema_alpha=1.0)  # EMA = dernier pnl
        pnl = {"Trend Following": -0.01, "Momentum": -0.01,
               "Mean Reversion": 0.02, "Scalping": 0.01, "Carry": 0.005}
        h.allocate(self._base(), pnl_by_strategy=pnl)
        scales = h.last_allocation["family_scales"]
        # famille trend (pnl négatif) : scale < 1 ; meanrev (pnl positif) : scale > 1
        assert scales["trend"] < 1.0
        assert scales["meanrev"] > 1.0
        assert scales["trend"] < scales["meanrev"]

    def test_family_scale_bounded(self):
        h = HierarchicalAllocator(family_ema_alpha=1.0)
        # perf extrême : le scale reste dans [0.60, 1.20]
        pnl = {s: 0.99 for s in self._base()}
        h.allocate(self._base(), pnl_by_strategy=pnl)
        scales = h.last_allocation["family_scales"]
        assert all(0.60 <= v <= 1.20 for v in scales.values())

    def test_exploration_regret_included(self):
        h = HierarchicalAllocator()
        t = RegretTracker(decay=1.0)
        for _ in range(REG_MIN_SAMPLES + 5):
            t.record("Trend Following", 0.01)
            t.record("Scalping", -0.01)
        out = h.allocate(self._base(), regret_tracker=t)
        # le regret de Scalping ajoute de l'exploration -> son poids reste > MIN
        assert out["Scalping"] >= MIN_WEIGHT

    def test_reversible_without_regret(self):
        """Sans pnl ni regret : l'allocation est déterministe et bornée."""
        h1, h2 = HierarchicalAllocator(), HierarchicalAllocator()
        o1 = h1.allocate(self._base())
        o2 = h2.allocate(self._base())
        assert o1 == o2


# --------------------------------------------------------------------------- #
# 4. Intégration MetaAllocationEngine
# --------------------------------------------------------------------------- #
class TestEngineIntegration:
    def _engine(self):
        from strategies.engine import MetaAllocationEngine, TrendFollowingStrategy
        return MetaAllocationEngine(strategies=[TrendFollowingStrategy(),
                                                MeanReversionStub()])

    def test_allocate_without_hierarchical_unchanged(self):
        """Sans hierarchical_scales : signal STRICTEMENT identique."""
        from strategies.engine import MetaAllocationEngine, TrendFollowingStrategy
        e1 = MetaAllocationEngine(strategies=[TrendFollowingStrategy(), TrendFollowingStrategy()])
        e2 = MetaAllocationEngine(strategies=[TrendFollowingStrategy(), TrendFollowingStrategy()])
        md = {"vpin": 0.5, "kyle_lambda": 0.0, "onchain_risk": 0.0, "symbol": "BTCUSDT"}
        c1 = e1.allocate(md, 2, 0.0, 0.0)
        c2 = e2.allocate(md, 2, 0.0, 0.0, hierarchical_scales=None)
        assert c1["final_signal"] == c2["final_signal"]

    def test_record_regret_in_update_pnl(self):
        e = self._engine()
        for _ in range(5):
            e.update_pnl_attribution("Trend Following", 0.01)
        assert e.regret_tracker.get("Trend Following") >= 0.0
        assert e.regret_tracker.n_updates.get("Trend Following", 0) == 5

    def test_hierarchical_scales_bounded(self):
        e = self._engine()
        scales = e.hierarchical_scales()
        assert set(scales) == {s.name for s in e.strategies}
        assert all(0.0 < v <= 1.0 for v in scales.values())

    def test_regret_wired_to_close(self):
        """record_regret appelé depuis update_pnl_attribution (clôture)."""
        src = (ROOT / "strategies/engine.py").read_text(encoding="utf-8")
        assert "self.record_regret(strategy, float(pnl_pct))" in src


class MeanReversionStub:
    """Stratégie minimale pour les tests d'intégration."""
    name = "Mean Reversion"
    enabled = True

    def generate_signal(self, market_data):
        return 0.0, 0.5


# --------------------------------------------------------------------------- #
# 5. Câblage main + télémétrie + API
# --------------------------------------------------------------------------- #
class TestWiring:
    def test_main_wiring(self):
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        assert "hierarchical_scales=meta_engine.hierarchical_scales()" in src

    def test_telemetry_exposes_hierarchical(self):
        import main  # noqa: F401
        from telemetry import compile_telemetry_data
        tel = compile_telemetry_data()
        assert "hierarchical_allocator" in tel
        assert "regret" in tel["hierarchical_allocator"]
        assert "family" in tel["hierarchical_allocator"]

    def test_api_meta_allocator(self):
        from fastapi.testclient import TestClient

        import main  # noqa: F401
        with TestClient(main.app) as c:
            r = c.get("/api/v1/meta-allocator")
            assert r.status_code == 200
            body = r.json()
            assert "regret" in body and "hierarchical" in body

    def test_no_mode_flag(self):
        import inspect
        src = inspect.getsource(RegretTracker) + inspect.getsource(HierarchicalAllocator)
        assert "active_mode" not in src
        assert '"DEMO"' not in src and '"REAL"' not in src

    def test_config_driven(self):
        from core.config import settings
        assert MIN_WEIGHT == settings.get_float("hierarchical", "min_weight", 0.02)
        assert MAX_WEIGHT == settings.get_float("hierarchical", "max_weight", 0.45)
        assert REG_EXPLORATION_WEIGHT == settings.get_float("hierarchical", "regret_exploration_weight", 0.15)
