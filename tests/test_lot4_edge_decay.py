"""
LOT 4 (mandat — Adaptativité) : Edge Decay Engine.

Vérifié ici :
  1. Cycle d'états : HEALTHY -> DEGRADED -> WARNING -> DISABLED -> RECOVERY,
     sans jugement sur échantillon trop faible (n < min_samples -> HEALTHY).
  2. Scales de pondération BORNÉS [0.30, 1.0] : JAMAIS 0 (pas de suppression
     dure — une stratégie dégradée est sous-pondérée, pas arrêtée).
  3. RÉVERSIBILITÉ : une stratégie DISABLED dont l'expectancy redevient
     positive passe en RECOVERY et retrouve un poids progressif.
  4. Métriques : expectancy EMA, hit rate EMA, profit factor, perf par régime.
  5. Intégration MetaAllocationEngine : edge_decay_scales multiplie les poids
     (absent -> comportement pré-LOT 4 strictement identique).
  6. Câblage main.py : record_outcome à la clôture, scales passés à allocate,
     télémétrie edge_decay exposée, API /api/v1/edge-decay.
  7. DÉMO == RÉAL : aucun flag de mode.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from core.edge_decay import (  # noqa: E402
    DEGRADED,
    DISABLED,
    EDGE_DISABLE_EXPECTANCY,
    EDGE_RECOVER_EXPECTANCY,
    EDGE_SCALE_DISABLED,
    EDGE_SCALE_RECOVERY,
    EDGE_SCALE_WARNING,
    EDGE_WARN_EXPECTANCY,
    HEALTHY,
    RECOVERY,
    WARNING,
    EdgeDecayEngine,
    StrategyEdge,
)


# --------------------------------------------------------------------------- #
# 1. Cycle d'états
# --------------------------------------------------------------------------- #
class TestStates:
    def test_healthy_with_insufficient_samples(self):
        e = StrategyEdge("S")
        for _ in range(5):   # < min_samples (10)
            e.record(0.01)
        assert e.state == HEALTHY

    def test_healthy_with_good_performance(self):
        e = StrategyEdge("S")
        for _ in range(12):
            e.record(0.005)
        assert e.state in (HEALTHY, RECOVERY)

    def test_warning_after_losses(self):
        e = StrategyEdge("S")
        for _ in range(12):
            e.record(-0.002)   # expectancy EMA -> ~-0.002 < warn (-0.001)
        assert e.state == WARNING

    def test_disabled_after_big_losses(self):
        e = StrategyEdge("S")
        for _ in range(12):
            e.record(-0.004)   # < disable (-0.003)
        assert e.state == DISABLED

    def test_degraded_between(self):
        e = StrategyEdge("S")
        for _ in range(12):
            e.record(-0.0004)   # entre recover (+0.0005) et warn (-0.001)
        assert e.state == DEGRADED

    def test_recovery_after_disabled(self):
        e = StrategyEdge("S")
        for _ in range(12):
            e.record(-0.004)
        assert e.state == DISABLED
        for _ in range(12):
            e.record(0.004)     # expectancy remonte au-dessus de recover
        assert e.state == RECOVERY


# --------------------------------------------------------------------------- #
# 2. Scales bornés + réversibilité
# --------------------------------------------------------------------------- #
class TestScales:
    def test_scales_bounded_never_zero(self):
        e = StrategyEdge("S")
        for _ in range(12):
            e.record(-0.005)
        assert e.state == DISABLED
        assert e.weight_scale() == pytest.approx(EDGE_SCALE_DISABLED)
        assert 0.0 < e.weight_scale() <= 1.0

    def test_healthy_scale_one(self):
        e = StrategyEdge("S")
        assert e.weight_scale() == 1.0
        for _ in range(12):
            e.record(0.005)
        assert e.weight_scale() == 1.0

    def test_warning_scale(self):
        e = StrategyEdge("S")
        for _ in range(12):
            e.record(-0.002)
        assert e.state == WARNING
        assert e.weight_scale() == pytest.approx(EDGE_SCALE_WARNING)

    def test_recovery_scale_progressive(self):
        e = StrategyEdge("S")
        for _ in range(12):
            e.record(-0.004)
        for _ in range(12):
            e.record(0.004)
        assert e.state == RECOVERY
        assert e.weight_scale() == pytest.approx(EDGE_SCALE_RECOVERY)
        # le scale de recovery est > scale disabled (poids progressif)
        assert EDGE_SCALE_RECOVERY > EDGE_SCALE_DISABLED


# --------------------------------------------------------------------------- #
# 3. Métriques
# --------------------------------------------------------------------------- #
class TestMetrics:
    def test_expectancy_and_hit_rate(self):
        e = StrategyEdge("S")
        for _ in range(6):
            e.record(0.01)
        for _ in range(6):
            e.record(-0.01)
        assert e.expectancy is not None
        assert e.hit_rate is not None
        assert 0.0 < e.hit_rate < 1.0

    def test_profit_factor(self):
        e = StrategyEdge("S")
        for _ in range(5):
            e.record(0.01)
        for _ in range(5):
            e.record(-0.005)
        pf = e.profit_factor()
        assert pf is not None and pf > 1.0

    def test_regime_summary(self):
        e = StrategyEdge("S")
        e.record(0.01, regime_id=0)
        e.record(-0.005, regime_id=0)
        e.record(0.02, regime_id=2)
        rs = e.regime_summary()
        assert rs["0"]["n"] == 2
        assert rs["2"]["n"] == 1


# --------------------------------------------------------------------------- #
# 4. Moteur global
# --------------------------------------------------------------------------- #
class TestEngine:
    def test_scales_and_report(self):
        eng = EdgeDecayEngine(strategies=["A", "B"])
        for _ in range(12):
            eng.record_outcome("A", -0.004, regime_id=1)   # -> DISABLED
        for _ in range(12):
            eng.record_outcome("B", 0.005, regime_id=2)    # -> HEALTHY/RECOVERY
        rep = eng.report()
        assert rep["per_strategy"]["A"]["state"] == DISABLED
        assert rep["per_strategy"]["A"]["weight_scale"] == pytest.approx(EDGE_SCALE_DISABLED)
        assert rep["counts"]["disabled"] == 1
        scales = eng.scales()
        assert set(scales) == {"A", "B"}
        assert all(0.0 < v <= 1.0 for v in scales.values())

    def test_never_raises(self):
        eng = EdgeDecayEngine()
        eng.record_outcome("", 0.01)          # stratégie vide ignorée
        eng.record_outcome("X", "not-a-number")  # ne doit pas lever
        assert eng.report()["counts"]["total"] >= 0


# --------------------------------------------------------------------------- #
# 5. Intégration meta-allocation
# --------------------------------------------------------------------------- #
class TestAllocationIntegration:
    def _engine(self):
        from strategies.engine import MetaAllocationEngine, TrendFollowingStrategy
        return MetaAllocationEngine(strategies=[TrendFollowingStrategy(),
                                                TrendFollowingStrategy()])

    def test_allocate_without_scales_unchanged(self):
        """Sans edge_decay_scales, les signaux sont STRICTEMENT identiques
        (comportement pré-LOT 4 conservé)."""
        e1, e2 = self._engine(), self._engine()
        md = {"vpin": 0.5, "kyle_lambda": 0.0, "onchain_risk": 0.0, "symbol": "BTCUSDT"}
        c1 = e1.allocate(md, 2, 0.0, 0.0)
        c2 = e2.allocate(md, 2, 0.0, 0.0, edge_decay_scales=None)
        assert c1["final_signal"] == c2["final_signal"]

    def test_allocate_with_scales_changes_weights(self):
        """Avec des scales inégaux, le poids de la stratégie est réduit
        (visible dans les contributions — le scale multiplie le poids)."""
        e = self._engine()
        md = {"vpin": 0.5, "kyle_lambda": 0.0, "onchain_risk": 0.0, "symbol": "BTCUSDT"}
        c0 = e.allocate(md, 2, 0.0, 0.0)
        name = e.strategies[0].name
        c1 = e.allocate(md, 2, 0.0, 0.0,
                        edge_decay_scales={name: EDGE_SCALE_DISABLED})
        contrib0 = c0["contributions"].get(name, {})
        contrib1 = c1["contributions"].get(name, {})
        w0 = float(contrib0.get("weight", 0.0))
        w1 = float(contrib1.get("weight", 0.0))
        assert w1 < w0  # la stratégie sous-pondérée voit son poids réduit


# --------------------------------------------------------------------------- #
# 6. Câblage main + télémétrie + API
# --------------------------------------------------------------------------- #
class TestWiring:
    def test_main_wiring(self):
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        assert "edge_decay = EdgeDecayEngine(" in src
        assert "edge_decay.record_outcome(strategy, pnl_pct" in src
        assert "edge_decay_scales=edge_decay.scales()" in src

    def test_telemetry_exposes_edge_decay(self):
        import main  # noqa: F401
        from telemetry import compile_telemetry_data
        tel = compile_telemetry_data()
        assert "edge_decay" in tel
        assert "per_strategy" in tel["edge_decay"]

    def test_api_edge_decay(self):
        from fastapi.testclient import TestClient

        import main  # noqa: F401
        with TestClient(main.app) as c:
            r = c.get("/api/v1/edge-decay")
            assert r.status_code == 200
            assert "per_strategy" in r.json()

    def test_no_mode_flag(self):
        import inspect
        src = inspect.getsource(StrategyEdge) + inspect.getsource(EdgeDecayEngine)
        assert "active_mode" not in src
        assert '"DEMO"' not in src and '"REAL"' not in src

    def test_config_driven(self):
        from core.config import settings
        assert EDGE_SCALE_DISABLED == settings.get_float("edge_decay", "scale_disabled", 0.30)
        assert EDGE_WARN_EXPECTANCY == settings.get_float("edge_decay", "warn_expectancy", -0.001)
        assert EDGE_DISABLE_EXPECTANCY == settings.get_float("edge_decay", "disable_expectancy", -0.003)
        assert EDGE_RECOVER_EXPECTANCY == settings.get_float("edge_decay", "recover_expectancy", 0.0005)
