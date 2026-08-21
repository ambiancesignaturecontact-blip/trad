"""
LOT 10 (mandat — Paper-Trading) : rapport objectif de validation + décision
sur la suite.

Vérifié ici :
  1. build_paper_validation_report : 10 critères PASS/WARN/FAIL, chacun avec
     preuve ; statut global READY / IN_PROGRESS / NOT_READY.
  2. HONNÊTETÉ : sans données suffisantes, les critères sont WARN/FAIL (jamais
     PASS sans mesure) — un système qui vient de démarrer ne peut PAS être READY.
  3. Protection : drawdown/kill switch -> FAIL ; explicabilité -> PASS si les
     décisions ont des raisons ; limites -> PASS si taux de rejets bas.
  4. API /api/v1/paper-validation-report + télémétrie paper_validation_report.
  5. Dashboard : section validation présente.
  6. La décision de passage en REAL reste MANUELLE (note dans le rapport).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from core.paper_validation import (  # noqa: E402
    _paper_days,
    _streak,
    build_paper_validation_report,
)


class FakeDB:
    """Mini DB : jours de paper + journal + orders + events."""

    def __init__(self, days=None, decisions=0, closed=0, orders=None,
                 fills=0):
        self.days = days or []
        self.decisions = decisions
        self.closed = closed
        self.orders = orders or []
        self.fills = fills

    def get_setting(self, key, decrypt=False):
        import json as j
        if key == "paper_validation_days":
            return j.dumps(self.days)
        return ""

    def decision_journal_summary(self):
        return {"total": self.decisions, "closed_n": self.closed,
                "by_decision": {}, "by_reason": {}}

    def get_decision_journal(self, limit=200):
        return [{"reason": "conviction", "symbol": "X"} for _ in range(min(limit, self.decisions))]

    def get_all_orders(self):
        return self.orders

    def list_events(self, event_type="", limit=5000):
        return [{"event_type": "paper_fill"} for _ in range(min(limit, self.fills))]


def _state(**overrides):
    s = {
        "kill_switch_active": False,
        "risk_state": {"state": "NORMAL"},
        "equity_history_demo": [100.0, 101.0, 100.5],
        "current_equity": 100.5,
        "initial_capital_demo": 100.0,
        "last_tick_ts": 0.0,
        "no_trade_stats": {"reasons": {"conviction": 10, "halt": 2}},
    }
    s.update(overrides)
    return s


# --------------------------------------------------------------------------- #
# 1. Structure du rapport
# --------------------------------------------------------------------------- #
class TestReportStructure:
    def test_ten_criteria(self):
        db = FakeDB(decisions=300, closed=40,
                    orders=[{"status": "FILLED"} for _ in range(90)]
                           + [{"status": "REJECTED"} for _ in range(5)],
                    fills=40)
        st = _state(last_tick_ts=1_000_000.0)
        r = build_paper_validation_report(db, st)
        assert len(r["criteria"]) == 10
        for k, v in r["criteria"].items():
            assert v["status"] in ("PASS", "WARN", "FAIL"), k
            assert v["evidence"], f"{k}: preuve manquante"

    def test_status_global_one_of_three(self):
        db = FakeDB(decisions=300, closed=40, fills=40,
                    orders=[{"status": "FILLED"} for _ in range(90)])
        r = build_paper_validation_report(db, _state(last_tick_ts=1_000_000.0))
        assert r["status"] in ("READY", "IN_PROGRESS", "NOT_READY")

    def test_note_manual_decision(self):
        db = FakeDB(decisions=300, closed=40, fills=40,
                    orders=[{"status": "FILLED"} for _ in range(90)])
        r = build_paper_validation_report(db, _state(last_tick_ts=1_000_000.0))
        assert "MANUELLE" in r["note"]


# --------------------------------------------------------------------------- #
# 2. Honnêteté : pas de PASS sans mesure
# --------------------------------------------------------------------------- #
class TestHonesty:
    def test_empty_system_not_ready(self):
        """Système vide (aucune donnée) -> NOT_READY, jamais READY."""
        db = FakeDB()
        r = build_paper_validation_report(db, _state())
        assert r["status"] == "NOT_READY"
        assert r["criteria"]["C1_CONTINUITE"]["status"] == "FAIL"
        assert r["criteria"]["C2_FONCTIONNEMENT"]["status"] == "FAIL"

    def test_insufficient_samples_warn_not_pass(self):
        """Échantillon faible -> WARN (calibration/exécution/PnL), pas PASS."""
        db = FakeDB(decisions=300, closed=5, fills=5,
                    orders=[{"status": "FILLED"} for _ in range(10)])
        r = build_paper_validation_report(db, _state(last_tick_ts=1_000_000.0))
        assert r["criteria"]["C8_EXECUTION"]["status"] == "WARN"
        assert r["criteria"]["C9_CALIBRATION"]["status"] == "WARN"
        assert r["criteria"]["C10_PNL_NET"]["status"] == "WARN"

    def test_kill_switch_fails_protection(self):
        db = FakeDB(decisions=300, closed=40, fills=40,
                    orders=[{"status": "FILLED"} for _ in range(90)])
        r = build_paper_validation_report(db, _state(kill_switch_active=True,
                                                     last_tick_ts=1_000_000.0))
        assert r["criteria"]["C4_PROTECTION"]["status"] == "FAIL"

    def test_drawdown_above_limit_fails(self):
        db = FakeDB(decisions=300, closed=40, fills=40,
                    orders=[{"status": "FILLED"} for _ in range(90)])
        st = _state(equity_history_demo=[100.0, 80.0, 85.0], current_equity=85.0,
                    initial_capital_demo=100.0, last_tick_ts=1_000_000.0)
        r = build_paper_validation_report(db, st)
        assert r["criteria"]["C4_PROTECTION"]["status"] in ("WARN", "FAIL")


# --------------------------------------------------------------------------- #
# 3. Critères spécifiques
# --------------------------------------------------------------------------- #
class TestCriteria:
    def test_continuity_from_days(self):
        db = FakeDB(decisions=300, closed=40, fills=40,
                    orders=[{"status": "FILLED"} for _ in range(90)],
                    days=["2026-08-01", "2026-08-02", "2026-08-03"])
        r = build_paper_validation_report(db, _state(last_tick_ts=1_000_000.0))
        assert r["criteria"]["C1_CONTINUITE"]["status"] == "FAIL"  # 3/28 jours

    def test_explicability_pass_with_reasons(self):
        db = FakeDB(decisions=300, closed=40, fills=40,
                    orders=[{"status": "FILLED"} for _ in range(90)])
        r = build_paper_validation_report(db, _state(last_tick_ts=1_000_000.0))
        assert r["criteria"]["C5_EXPLICABILITE"]["status"] == "PASS"

    def test_limits_pass_low_reject_rate(self):
        db = FakeDB(decisions=300, closed=40, fills=40,
                    orders=[{"status": "FILLED"} for _ in range(90)]
                           + [{"status": "REJECTED"} for _ in range(3)])
        r = build_paper_validation_report(db, _state(last_tick_ts=1_000_000.0))
        assert r["criteria"]["C6_LIMITES"]["status"] == "PASS"

    def test_limits_fail_high_reject_rate(self):
        db = FakeDB(decisions=300, closed=40, fills=40,
                    orders=[{"status": "FILLED"} for _ in range(50)]
                           + [{"status": "REJECTED"} for _ in range(50)])
        r = build_paper_validation_report(db, _state(last_tick_ts=1_000_000.0))
        assert r["criteria"]["C6_LIMITES"]["status"] == "FAIL"

    def test_drift_warn_when_severe(self):
        db = FakeDB(decisions=300, closed=40, fills=40,
                    orders=[{"status": "FILLED"} for _ in range(90)])
        r = build_paper_validation_report(db, _state(last_tick_ts=1_000_000.0),
                                          drift_psi={"status": "SEVERE"})
        assert r["criteria"]["C7_NON_DERIVE"]["status"] == "WARN"


# --------------------------------------------------------------------------- #
# 4. Helpers
# --------------------------------------------------------------------------- #
class TestHelpers:
    def test_paper_days_parsing(self):
        class D:
            def get_setting(self, k, decrypt=False):
                import json as j
                return j.dumps(["2026-08-01", "2026-08-02"])
        assert _paper_days(D()) == ["2026-08-01", "2026-08-02"]

    def test_streak(self):
        assert _streak(["2026-08-01"]) == 1
        assert _streak(["2026-08-01", "2026-08-02", "2026-08-03"]) == 3
        assert _streak(["2026-08-01", "2026-08-02", "2026-08-05"]) == 1
        assert _streak([]) == 0


# --------------------------------------------------------------------------- #
# 5. API + télémétrie + dashboard
# --------------------------------------------------------------------------- #
class TestExposure:
    def test_api_report(self):
        from fastapi.testclient import TestClient

        import main  # noqa: F401
        with TestClient(main.app) as c:
            r = c.get("/api/v1/paper-validation-report")
            assert r.status_code == 200
            body = r.json()
            assert "criteria" in body and len(body["criteria"]) == 10
            assert body["status"] in ("READY", "IN_PROGRESS", "NOT_READY")

    def test_telemetry_exposes_report(self):
        import main  # noqa: F401
        from telemetry import compile_telemetry_data
        tel = compile_telemetry_data()
        assert "paper_validation_report" in tel
        assert len(tel["paper_validation_report"]["criteria"]) == 10

    def test_dashboard_has_section(self):
        dash = (ROOT / "templates/dashboard.html").read_text(encoding="utf-8")
        assert "VALIDATION PAPER-TRADING" in dash
        assert "val-pt-status" in dash

    def test_no_mode_flag(self):
        import inspect
        src = inspect.getsource(build_paper_validation_report)
        assert "active_mode" not in src


# --------------------------------------------------------------------------- #
# 6. PHASE 2 — régression : le journal reflète la décision FINALE
# --------------------------------------------------------------------------- #
class TestJournalFinalDecision:
    def test_journal_recorded_after_all_gates(self):
        """PHASE 2 (audit) : l'appel journal_decision est APRÈS les gates
        (RR filter, HALT, order flow, cascade, pyramiding, netting) — preuve
        du bug : 112 TRADE journalisés sans fill en 19h."""
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        i_gate_init = src.find("_gate_block = None")
        i_journal = src.find("_dj_id = journal_decision(")
        i_desired = src.find("desired_qty = target_direction * target_qty")
        assert i_gate_init != -1 and i_journal != -1 and i_desired != -1
        # le journal vient APRÈS l'init de la capture de gate et APRÈS le bloc gates
        assert i_gate_init < i_journal < i_desired

    def test_gate_block_captured_in_each_gate(self):
        """Chaque gate qui annule capture sa raison (fidélité du journal)."""
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        for marker in ("RR filter: {_rr_reason}", "HALT: {risk_state.reason}",
                       "_gate_block = _avoid_reason", "_gate_block = _casc_reason",
                       "_gate_block = _pyr_reason",
                       "netting: retournement sans signal fort"):
            assert marker in src, f"gate reason non capturée : {marker}"

    def test_journal_uses_final_decision(self):
        """La décision journalisée est TRADE seulement si target_direction != 0."""
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        assert '_dj_decision = "TRADE" if target_direction != 0.0 else "WAIT"' in src
        # la raison d'un WAIT par gate est le _gate_block
        assert "_gate_block or _dj_opp.get(\"reason\", \"conviction\")" in src

    def test_no_journal_in_evaluate_block(self):
        """Aucun appel journal_decision avant le bloc des gates."""
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        i_eval = src.find("_opp = trade_opportunity.evaluate(")
        i_journal = src.find("_dj_id = journal_decision(")
        assert 0 < i_eval < i_journal
        # entre evaluate et le journal, les gates sont présentes
        between = src[i_eval:i_journal]
        assert "GATES ORDER FLOW" in between or "FILTRE D'ENTRÉE" in between \
            or "PYRAMIDING" in between
