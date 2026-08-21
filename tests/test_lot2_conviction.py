"""
LOT 2 (mandat — Conviction Engine + Trade Opportunity / No-Trade Engine).

Vérifié ici :
  1. expected_edge_net : math exacte (E = p·R − (1−p) − coûts/SL), point mort,
     None sans historique (on n'invente pas d'edge).
  2. conviction_level : HIGH/NORMAL/LOW/NO_TRADE selon les seuils.
  3. ConvictionEngine.calibrate : base = calibrated_conviction (pas de
     ré-implémentation — valeurs identiques sans modificateur), modificateurs
     DÉFENSIFS bornés (régime incertain x0.95, slippage > 50bps x0.90),
     jamais de boost artificiel, UNCERTAIN quand non calibré.
  4. TrackOutcome + calibration_report : buckets, win rate par bucket,
     expectancy, calibration error — rien de fabriqué sans échantillon.
  5. TradeOpportunityEngine : TRADE quand edge net > 0 ; WAIT avec raisons
     stables (conviction / EDGE_INSUFFICIENT / EXECUTION_RISK /
     UNCALIBRATED / halt) ; l'ordre des vérifications est fixe (sécurité
     d'abord).
  6. Câblage main.py : conviction_engine + trade_opportunity instanciés,
     appelés dans la boucle, outcome tracké à la clôture.
  7. DÉMO == RÉAL : aucun flag de mode dans le module.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from core.conviction_engine import (  # noqa: E402
    HIGH_CONVICTION,
    LOW_CONVICTION,
    MIN_ENTRY_THRESHOLD,
    NO_TRADE,
    NORMAL,
    UNCERTAIN,
    WAIT_CONVICTION,
    WAIT_EDGE_INSUFFICIENT,
    WAIT_EXECUTION_RISK,
    WAIT_UNCALIBRATED,
    ConvictionEngine,
    TradeOpportunityEngine,
    conviction_level,
    expected_edge_net,
)
from core.risk_pipeline import calibrated_conviction  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Edge net (math exacte)
# --------------------------------------------------------------------------- #
class TestExpectedEdgeNet:
    def test_math_exact(self):
        # E = p·R − (1−p) − coûts/SL avec p=0.55, R=1.8, coûts=0.002, SL=0.03
        e = expected_edge_net(0.55, 1.8, 0.002, 0.03)
        assert e == pytest.approx(0.55 * 1.8 - 0.45 - 0.002 / 0.03, abs=1e-9)
        assert e > 0

    def test_none_without_history(self):
        """Sans win rate (pas d'historique) -> None : on n'invente pas un edge."""
        assert expected_edge_net(None) is None

    def test_point_mort(self):
        """p = (1 + coûts/SL)/(R+1) -> edge = 0 (limite)."""
        p_star = (1 + 0.002 / 0.03) / (1.8 + 1)
        e = expected_edge_net(p_star, 1.8, 0.002, 0.03)
        assert abs(e) < 1e-9

    def test_negative_edge_possible(self):
        """Avec un win rate très bas (non borné ici), l'edge est négatif."""
        e = expected_edge_net(0.30, 1.8, 0.002, 0.03)
        assert e < 0


# --------------------------------------------------------------------------- #
# 2. Niveaux de conviction
# --------------------------------------------------------------------------- #
class TestConvictionLevel:
    def test_levels(self):
        assert conviction_level(0.05) == NO_TRADE
        assert conviction_level(MIN_ENTRY_THRESHOLD) == LOW_CONVICTION
        assert conviction_level(0.12) == LOW_CONVICTION
        assert conviction_level(0.15) == NORMAL
        assert conviction_level(0.20) == NORMAL
        assert conviction_level(0.25) == HIGH_CONVICTION
        assert conviction_level(0.60) == HIGH_CONVICTION


# --------------------------------------------------------------------------- #
# 3. ConvictionEngine.calibrate
# --------------------------------------------------------------------------- #
class TestCalibrate:
    def test_base_equals_calibrated_conviction(self):
        """Sans modificateur (confiance régime OK, pas de slippage), la base
        est STRICTEMENT calibrated_conviction — aucun ré-implémentation."""
        eng = ConvictionEngine()
        r = eng.calibrate(0.20, win_rate=0.55, has_history=True,
                          regime_confidence=0.9, slippage_bps=5.0)
        assert r["conviction"] == pytest.approx(
            calibrated_conviction(0.20, 0.55), abs=1e-6)
        assert r["base"] == r["conviction"]

    def test_regime_uncertain_reduces(self):
        eng = ConvictionEngine()
        r = eng.calibrate(0.20, win_rate=0.55, has_history=True,
                          regime_confidence=0.10)
        base = calibrated_conviction(0.20, 0.55)
        assert r["conviction"] == pytest.approx(base * 0.95, abs=0.001)
        assert r["conviction"] < base
        assert any("régime incertain" in x for x in r["reasons"])

    def test_slippage_high_reduces(self):
        eng = ConvictionEngine()
        r = eng.calibrate(0.20, win_rate=0.55, has_history=True,
                          regime_confidence=0.9, slippage_bps=80.0)
        base = calibrated_conviction(0.20, 0.55)
        assert r["conviction"] == pytest.approx(base * 0.90, abs=0.001)

    def test_no_artificial_boost(self):
        """Aucun input ne peut AUGMENTER la conviction au-delà de la base
        calibrée (sauf le win rate, déjà dans calibrated_conviction)."""
        eng = ConvictionEngine()
        base = eng.calibrate(0.30, win_rate=0.60, has_history=True,
                             regime_confidence=0.9)
        # tous les modificateurs sont <= 1.0 : conviction <= base
        assert base["conviction"] <= base["base"] + 1e-9

    def test_uncalibrated_when_no_history_and_weak_signal(self):
        eng = ConvictionEngine()
        r = eng.calibrate(0.10, win_rate=0.5, has_history=False,
                          regime_confidence=0.9)
        assert r["uncalibrated"] is True
        assert r["level"] == UNCERTAIN

    def test_not_uncalibrated_with_history(self):
        eng = ConvictionEngine()
        r = eng.calibrate(0.10, win_rate=0.55, has_history=True,
                          regime_confidence=0.9)
        assert r["uncalibrated"] is False
        assert r["level"] in (LOW_CONVICTION, NORMAL, NO_TRADE)

    def test_edge_net_exposed(self):
        eng = ConvictionEngine()
        r = eng.calibrate(0.20, win_rate=0.55, has_history=True,
                          regime_confidence=0.9)
        assert r["edge_net"] == pytest.approx(expected_edge_net(0.55), abs=1e-5)
        r2 = eng.calibrate(0.20, win_rate=None, has_history=False,
                           regime_confidence=0.9)
        assert r2["edge_net"] is None

    def test_bounded(self):
        eng = ConvictionEngine()
        r = eng.calibrate(5.0, win_rate=0.65, has_history=True,
                          regime_confidence=1.0)
        # borne = CONVICTION_CALIB_MAX (1.25) — comportement de calibrated_conviction
        assert 0.0 <= r["conviction"] <= 1.25


# --------------------------------------------------------------------------- #
# 4. Calibration mesurée (buckets)
# --------------------------------------------------------------------------- #
class TestCalibration:
    def test_track_outcome_and_report(self):
        eng = ConvictionEngine()
        # 3 trades à conviction ~0.20 (bucket [0.15,0.25)) : 2 gagnants
        for i in range(3):
            eng.track_outcome(0.20, success=(i < 2), pnl_pct=0.01 if i < 2 else -0.02)
        rep = eng.calibration_report()
        assert rep["n"] == 3
        assert rep["calibration_error"] == pytest.approx(abs(2 / 3 - 0.20), abs=0.01)
        bucket = [b for b in rep["buckets"] if b["bucket"] == "[0.15,0.25)"][0]
        assert bucket["n"] == 3
        assert bucket["win_rate"] == pytest.approx(2 / 3, abs=0.001)

    def test_empty_report_honest(self):
        eng = ConvictionEngine()
        rep = eng.calibration_report()
        assert rep["n"] == 0
        assert "attente de données" in rep["note"]

    def test_never_raises_on_bad_input(self):
        eng = ConvictionEngine()
        eng.track_outcome("not-a-number", True)  # ne doit pas lever
        eng.track_outcome(0.2, True, pnl_pct="x")  # ne doit pas lever
        rep = eng.calibration_report()
        # entrées invalides ignorées SANS lever ; les valides sont comptées
        assert rep["n"] >= 0


# --------------------------------------------------------------------------- #
# 5. Trade Opportunity Engine
# --------------------------------------------------------------------------- #
class TestTradeOpportunity:
    def _eng(self):
        return TradeOpportunityEngine()

    def test_trade_when_edge_positive(self):
        e = self._eng()
        r = e.evaluate(signal=0.20, conviction=0.18, threshold=0.08,
                       edge_net=0.25, win_rate=0.55, risk_state="NORMAL")
        assert r["decision"] == "TRADE"

    def test_wait_below_threshold(self):
        e = self._eng()
        r = e.evaluate(signal=0.05, conviction=0.05, threshold=0.08,
                       edge_net=0.25, risk_state="NORMAL")
        assert r["decision"] == "WAIT"
        assert r["reason"] == WAIT_CONVICTION

    def test_wait_edge_insufficient(self):
        e = self._eng()
        r = e.evaluate(signal=0.20, conviction=0.18, threshold=0.08,
                       edge_net=-0.01, win_rate=0.30, risk_state="NORMAL")
        assert r["decision"] == "WAIT"
        assert r["reason"] == WAIT_EDGE_INSUFFICIENT
        assert "frais + slippage" in r["detail"]

    def test_wait_execution_risk(self):
        e = self._eng()
        r = e.evaluate(signal=0.20, conviction=0.18, threshold=0.08,
                       edge_net=0.25, slippage_bps=80.0, risk_state="NORMAL")
        assert r["decision"] == "WAIT"
        assert r["reason"] == WAIT_EXECUTION_RISK

    def test_wait_uncalibrated(self):
        e = self._eng()
        r = e.evaluate(signal=0.10, conviction=0.10, threshold=0.08,
                       edge_net=None, uncalibrated=True, risk_state="NORMAL")
        assert r["decision"] == "WAIT"
        assert r["reason"] == WAIT_UNCALIBRATED
        assert "non calibrée" in r["detail"]

    def test_wait_halt_first(self):
        """L'ordre des vérifications est FIXE : HALT prime sur tout."""
        e = self._eng()
        r = e.evaluate(signal=0.30, conviction=0.28, threshold=0.08,
                       edge_net=0.5, risk_state="HALT")
        assert r["decision"] == "WAIT"
        assert r["reason"] == "halt"

    def test_never_raises(self):
        e = self._eng()
        r = e.evaluate(signal=None, conviction=None, threshold=0.08)  # ne doit pas lever
        assert r["decision"] in ("TRADE", "WAIT")


# --------------------------------------------------------------------------- #
# 6. Câblage main.py
# --------------------------------------------------------------------------- #
class TestMainWiring:
    def test_engines_instantiated_and_wired(self):
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        assert "conviction_engine = ConvictionEngine(STATE)" in src
        assert "trade_opportunity = TradeOpportunityEngine()" in src
        # calibrate appelé dans la boucle (remplace calibrated_conviction brut)
        assert "conviction_engine.calibrate(" in src
        assert "trade_opportunity.evaluate(" in src
        # outcome tracké à la clôture
        assert "conviction_engine.track_outcome(" in src
        # repli pré-LOT 2 conservé (réversibilité)
        assert "OPP fallback" in src

    def test_no_mode_flag_in_engine(self):
        import inspect
        src = (inspect.getsource(ConvictionEngine)
               + inspect.getsource(TradeOpportunityEngine))
        assert "active_mode" not in src
        assert '"DEMO"' not in src and '"REAL"' not in src

    def test_telemetry_exposes_engine(self):
        import main  # noqa: F401
        from telemetry import compile_telemetry_data
        tel = compile_telemetry_data()
        assert "conviction_engine" in tel
        assert "last_opportunity" in tel
        assert "conviction_calibration" in tel
        assert isinstance(tel["conviction_calibration"], dict)


# --------------------------------------------------------------------------- #
# 7. No-Trade buckets (raisons du moteur catégorisées)
# --------------------------------------------------------------------------- #
class TestNoTradeBuckets:
    def test_edge_and_uncalibrated_buckets(self):
        from core.meta_cognition import _no_trade_bucket
        assert _no_trade_bucket("|signal| 0.2 < seuil 0.3 | OPP:EDGE_INSUFFICIENT | edge net -0.01 <= 0") == "edge_insufficient"
        assert _no_trade_bucket("OPP:UNCALIBRATED | setup intéressant mais conviction non calibrée") == "uncalibrated"
        assert _no_trade_bucket("OPP:EXECUTION_RISK | slippage attendu 80 bps") == "execution_risk"
