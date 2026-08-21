"""
LOT 1 (mandat — diagnostic réel + observabilité décisionnelle).

Vérifié ici :
  1. `decide_no_trade` : la raison contient TOUJOURS |signal| vs seuil
     (observabilité : « pourquoi le bot n'a pas tradé ? ») + breakdown par
     catégorie agrégé dans event_log["reasons"].
  2. Catégorisation : chaque type de raison tombe dans le bon bucket.
  3. Télémétrie : no_trade_reasons + signal_stats exposés dans le payload.
  4. Dashboard : section « POURQUOI PAS DE TRADE ? » présente et alimentée.
  5. Aucune régression du comportement (le seuil n'est pas modifié, la
     décision d'abstention reste identique — seule la TRACE s'enrichit).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from core.meta_cognition import _no_trade_bucket, decide_no_trade  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Raison enrichie + comportement préservé
# --------------------------------------------------------------------------- #
class TestDecideNoTrade:
    def test_abstains_below_threshold(self):
        assert decide_no_trade("BTCUSDT", 0.05, 0.15, [], None, None) is True

    def test_trades_above_threshold(self):
        assert decide_no_trade("BTCUSDT", 0.30, 0.15, [], None, None) is False

    def test_reason_contains_signal_vs_threshold(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        decide_no_trade("BTCUSDT", 0.05, 0.15, [], None, None)
        assert "|signal| 0.050 < seuil 0.150" in caplog.text

    def test_reason_appends_contextual_reasons(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        decide_no_trade("BTCUSDT", 0.05, 0.15,
                        ["regime=Mean-Reverting Range", "moe={}"], None, None)
        assert "|signal| 0.050 < seuil 0.150" in caplog.text
        assert "regime=Mean-Reverting Range" in caplog.text

    def test_event_log_breakdown_accumulates(self):
        log = {"count": 0}
        decide_no_trade("A", 0.05, 0.15, [], log, None)
        decide_no_trade("B", 0.04, 0.15, ["RR filter: RR 1.5 < requis 2.0"], log, None)
        decide_no_trade("C", 0.06, 0.15, [f"HALT: {1}"], log, None)
        assert log["count"] == 3
        assert log["reasons"]["conviction"] == 1
        assert log["reasons"]["rr_filter"] == 1
        assert log["reasons"]["halt"] == 1

    def test_db_event_recorded_with_rich_reason(self, monkeypatch):
        events = []

        class FakeDB:
            def add_event(self, ts, etype, payload):
                events.append((etype, json.loads(payload)))

        decide_no_trade("BTCUSDT", 0.05, 0.15, [], None, FakeDB())
        assert events[0][0] == "no_trade"
        assert "|signal| 0.050 < seuil 0.150" in events[0][1]["reason"]

    def test_never_raises(self):
        # event_log malformé (None, pas un dict) ne doit jamais lever
        assert decide_no_trade("A", 0.05, 0.15, [], None, None) is True
        assert decide_no_trade("A", 0.05, 0.15, [], 42, None) is True


# --------------------------------------------------------------------------- #
# 2. Catégorisation
# --------------------------------------------------------------------------- #
class TestBucket:
    def test_conviction_bucket(self):
        assert _no_trade_bucket("|signal| 0.050 < seuil 0.150") == "conviction"
        assert _no_trade_bucket("|signal| 0.050 < seuil 0.150 | regime=X | moe={}") == "conviction"

    def test_rr_bucket(self):
        assert _no_trade_bucket("|signal| 0.2 < seuil 0.3 | RR filter: RR 1.5 < requis 2.0") == "rr_filter"

    def test_halt_bucket(self):
        assert _no_trade_bucket("|signal| 0.2 < seuil 0.3 | HALT: NEWS_SHOCK") == "halt"

    def test_cascade_and_orderflow_buckets(self):
        assert _no_trade_bucket("|signal| 0.2 < seuil 0.3 | cascade de liquidations") == "cascade"
        assert _no_trade_bucket("flux agressif dominant") == "order_flow"

    def test_meta_label_bucket(self):
        assert _no_trade_bucket("meta-label REAL: wr 0.40 (n=5)") == "meta_label"

    def test_regime_context_bucket(self):
        assert _no_trade_bucket("regime=Bear Trend (High Vol) | moe={'x': 1}") == "regime_context"

    def test_other_bucket(self):
        assert _no_trade_bucket("raison inconnue quelconque") == "other"


# --------------------------------------------------------------------------- #
# 3. Télémétrie + dashboard
# --------------------------------------------------------------------------- #
class TestObservability:
    def test_telemetry_exposes_no_trade_reasons(self):
        import main  # noqa: F401  (charge main COMPLET avant telemetry — évite le cycle d'import)
        from telemetry import compile_telemetry_data
        tel = compile_telemetry_data()
        assert "no_trade_reasons" in tel
        assert isinstance(tel["no_trade_reasons"], dict)
        assert "signal_stats" in tel
        assert "conviction_threshold" in tel

    def test_dashboard_has_decision_section(self):
        dash = (ROOT / "templates/dashboard.html").read_text(encoding="utf-8")
        assert "POURQUOI PAS DE TRADE" in dash
        assert "val-dec-reasons" in dash
        assert "val-dec-threshold" in dash

    def test_state_initialization_compatible(self):
        """L'état initial {'count': 0} reste valide (reasons via setdefault)."""
        import main
        log = main.STATE["no_trade_stats"]
        assert isinstance(log, dict)
        assert "count" in log
