"""
PHASE 3 — plateforme quantitative vivante (Cycle 1).

Vérifié ici :
  1. ResearchMemory : hypothèse enregistrée, expérience complétée (résultats
     OOS/stress/conclusion), KILL LIST (hypothèse tuée jamais re-proposée).
  2. CapitalAllocationEngine : recommandations (INCREASE exige une preuve ;
     REDUCE possible même si PnL récent positif ; FREEZE sur kill switch).
     Aucun changement de capital automatique (avis seulement).
  3. Benchmarking : comparaison bot vs buy & hold sur données réelles ;
     sans clôtures -> « insuffisant » honnête (aucun chiffre inventé).
  4. Daily Quant Report : sections complètes + persistance DB (index 30 jours).
  5. API : /api/v1/research-memory, /api/v1/capital-allocation,
     /api/v1/benchmark, /api/v1/daily-quant-report répondent.
"""
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from core.capital_allocation import (  # noqa: E402
    FREEZE,
    INCREASE,
    MAINTAIN,
    REDUCE,
    CapitalAllocationEngine,
)
from core.research_memory import ResearchMemory  # noqa: E402


class FakeDB:
    """Mini DB : experiments + journal + candles (pour les tests unitaires)."""

    def __init__(self):
        self.experiments = []
        self._next = 1
        self._killed = set()

    # experiments
    def ensure_experiments_table(self):
        pass

    def add_experiment(self, hypothesis, status="PENDING", result=""):
        eid = self._next
        self._next += 1
        self.experiments.append({"id": eid, "hypothesis": hypothesis,
                                 "status": status, "result": result,
                                 "killed": 0})
        return eid

    def update_experiment(self, eid, fields):
        for e in self.experiments:
            if e["id"] == eid:
                e.update(fields)
                return True
        return False

    def get_kill_list(self, limit=100):
        return [e for e in self.experiments if e.get("killed")][:limit]

    def is_hypothesis_killed(self, hypothesis):
        return any(e["hypothesis"] == hypothesis and e.get("killed")
                   for e in self.experiments)

    def list_experiments(self, limit=100):
        return self.experiments[-limit:][::-1]

    # journal
    def decision_journal_summary(self):
        return {"total": 100, "closed_n": 0, "by_decision": {"TRADE": 10, "WAIT": 90},
                "by_reason": {}, "closed_avg_pnl_pct": None}

    # candles (benchmark)
    def load_candles(self, symbol, limit=200):
        import numpy as np
        import pandas as pd
        n = min(limit, 100)
        close = np.linspace(100.0, 110.0, n)   # +10% buy & hold
        idx = pd.date_range("2026-01-01", periods=n, freq="h")
        return pd.DataFrame({"open": close, "high": close + 1,
                             "low": close - 1, "close": close,
                             "volume": np.full(n, 1000.0)}, index=idx)

    def list_events(self, event_type="", since=0.0, limit=5000):
        return []

    def save_setting(self, k, v):
        pass

    def get_setting(self, k, decrypt=False):
        return ""


# --------------------------------------------------------------------------- #
# 1. Research Memory + Kill List
# --------------------------------------------------------------------------- #
class TestResearchMemory:
    def test_hypothesis_registered(self):
        rm = ResearchMemory(FakeDB())
        eid = rm.record_hypothesis("Momentum trop sensible en HIGH_VOL", "obs")
        assert eid > 0

    def test_killed_hypothesis_never_reproposed(self):
        db = FakeDB()
        rm = ResearchMemory(db)
        eid = rm.record_hypothesis("Hypothèse A", "obs")
        assert rm.kill(eid, "invalidée par OOS") is True
        # re-proposition -> refusée (id 0)
        assert rm.record_hypothesis("Hypothèse A", "obs") == 0
        assert rm.is_killed("Hypothèse A") is True

    def test_experiment_result_recorded(self):
        db = FakeDB()
        rm = ResearchMemory(db)
        eid = rm.record_hypothesis("Hypothèse B", "obs")
        ok = rm.record_experiment_result(
            eid, {"oos": {"sharpe": -0.5}, "stress": {"worst": -3.0}},
            "edge négatif en OOS", "REJECT")
        assert ok is True
        exp = [e for e in db.experiments if e["id"] == eid][0]
        assert exp["status"] == "REJECTED"
        assert "sharpe" in exp["oos_results"]

    def test_promote_sets_candidate(self):
        db = FakeDB()
        rm = ResearchMemory(db)
        eid = rm.record_hypothesis("Hypothèse C", "obs")
        rm.record_experiment_result(eid, {"oos": {"sharpe": 1.2}}, "edge positif", "PROMOTE")
        exp = [e for e in db.experiments if e["id"] == eid][0]
        assert exp["status"] == "CANDIDATE"

    def test_never_raises(self):
        rm = ResearchMemory(None)
        assert rm.record_hypothesis("X", "obs") == 0
        assert rm.record_experiment_result(0, {}, "c", "REJECT") is False
        assert rm.kill(0, "r") is False
        assert rm.is_killed("X") is False


# --------------------------------------------------------------------------- #
# 2. Capital Allocation Engine
# --------------------------------------------------------------------------- #
class TestCapitalAllocation:
    def _eng(self):
        return CapitalAllocationEngine()

    def test_freeze_on_kill_switch(self):
        r = self._eng().recommend(kill_switch=True)
        assert r["recommendation"] == FREEZE

    def test_no_increase_without_proof(self):
        """0 clôture + validation NOT_READY -> jamais INCREASE."""
        r = self._eng().recommend(validation_status="NOT_READY", closed_trades=0)
        assert r["recommendation"] != INCREASE

    def test_increase_requires_full_proof(self):
        r = self._eng().recommend(
            validation_status="READY", closed_trades=40, expectancy_pct=0.15,
            drift_status="STABLE", calibration_n=40, calibration_error=0.05)
        assert r["recommendation"] == INCREASE

    def test_reduce_even_if_recent_pnl_positive(self):
        """Edge négatif sur échantillon suffisant -> REDUCE (mandat §3)."""
        r = self._eng().recommend(
            validation_status="IN_PROGRESS", closed_trades=40, expectancy_pct=-0.05)
        assert r["recommendation"] == REDUCE
        assert "même si le PnL récent est positif" in r["note"]

    def test_reduce_on_drift_severe(self):
        r = self._eng().recommend(drift_status="SEVERE")
        assert r["recommendation"] == REDUCE
        assert r["degradation_level"] in ("WARNING", "CRITICAL")

    def test_maintain_when_insufficient_evidence(self):
        r = self._eng().recommend()
        assert r["recommendation"] == MAINTAIN

    def test_never_changes_capital(self):
        r = self._eng().recommend(closed_trades=40, expectancy_pct=0.2,
                                  validation_status="READY")
        assert "AUCUN changement automatique" in r["capital_change"]


# --------------------------------------------------------------------------- #
# 3. Benchmarking
# --------------------------------------------------------------------------- #
class TestBenchmark:
    def test_no_closed_trades_honest(self):
        from core.benchmark import benchmark_report
        db = FakeDB()   # list_events -> [] : aucun trade clôturé
        r = benchmark_report(db)
        assert r["n_closed"] == 0
        assert "aucun chiffre inventé" in r["note"]

    def test_buy_and_hold_computed(self):
        from core.benchmark import buy_and_hold_return
        db = FakeDB()
        r = buy_and_hold_return(db, "BTCUSDT", window_bars=100)
        assert r == pytest.approx(10.0, abs=1e-6)  # linspace 100 -> 110


# --------------------------------------------------------------------------- #
# 4. Daily Quant Report
# --------------------------------------------------------------------------- #
class TestDailyReport:
    def test_sections_present(self):
        from core.daily_quant_report import build_daily_quant_report
        r = build_daily_quant_report(FakeDB(), {"regime_name": "Range",
                                                "regime_id": 2})
        for section in ("market", "trading", "calibration_tracking",
                        "intelligence", "risk", "execution", "research",
                        "recommendation"):
            assert section in r, f"section {section} manquante"
        assert r["recommendation"]["recommendation"] in (
            "INCREASE", "MAINTAIN", "REDUCE", "FREEZE")


# --------------------------------------------------------------------------- #
# 5. API
# --------------------------------------------------------------------------- #
class TestAPI:
    def test_endpoints_respond(self):
        from fastapi.testclient import TestClient

        import main  # noqa: F401
        with TestClient(main.app) as c:
            for path in ("/api/v1/research-memory", "/api/v1/capital-allocation",
                         "/api/v1/benchmark", "/api/v1/daily-quant-report"):
                r = c.get(path)
                assert r.status_code == 200, path
                assert isinstance(r.json(), dict), path

    def test_research_hypothesis_post(self):
        from fastapi.testclient import TestClient

        import main  # noqa: F401
        with TestClient(main.app) as c:
            r = c.post("/api/v1/research/hypothesis",
                       json={"hypothesis": "Test PHASE3 : scalping inefficace en Range",
                             "observation": "test unitaire"})
            body = r.json()
            assert body.get("id", 0) > 0 or "déjà tuée" in str(body.get("error", ""))


# --------------------------------------------------------------------------- #
# PHASE 3 Cycle 2 — suivi des clôtures du calibrage actuel (item 1)
# --------------------------------------------------------------------------- #
class TestCalibrationTracking:
    """calibration_close_tracking : les clôtures du calibrage ACTUEL (par
    system_version) deviennent mesurables — condition du CONDITIONAL GO."""

    def _make_db(self, tmp_path, rows):
        import sqlite3

        path = tmp_path / "dj.db"

        class MiniDB:
            is_postgres = False

            def __init__(self, p):
                self._conn = sqlite3.connect(str(p))
                self._conn.execute(
                    "CREATE TABLE decision_journal (id INTEGER PRIMARY KEY, "
                    "pnl_pct REAL, system_version TEXT)"
                )
                self._conn.executemany(
                    "INSERT INTO decision_journal (pnl_pct, system_version) "
                    "VALUES (?, ?)", rows)
                self._conn.commit()

            def get_connection(self):
                return self._conn

        return MiniDB(path)

    def test_zero_closes_honest(self, tmp_path):
        from core.paper_validation import calibration_close_tracking
        db = self._make_db(tmp_path, [])
        r = calibration_close_tracking(db, version="qp-test-00000000")
        assert r["n_closes"] == 0
        assert r["expectancy_pct"] is None      # jamais 0.0 inventé
        assert r["conditions_met"] is False
        assert r["progress_pct"] == 0.0

    def test_only_current_version_counted(self, tmp_path):
        from core.paper_validation import calibration_close_tracking
        db = self._make_db(tmp_path, [
            (0.01, "qp-old-00000000"),   # autre calibrage : ignoré
            (-0.005, "qp-old-00000000"),
            (0.02, "qp-current-11111111"),
            (-0.01, "qp-current-11111111"),
            (0.03, "qp-current-11111111"),
        ])
        r = calibration_close_tracking(db, version="qp-current-11111111")
        assert r["n_closes"] == 3
        assert r["win_rate"] == round(2 / 3, 4)
        assert r["expectancy_pct"] == round(((0.02 - 0.01 + 0.03) / 3) * 100, 4)
        assert r["cumulative_pnl_pct"] == round((0.02 - 0.01 + 0.03) * 100, 4)

    def test_conditions_met_at_30_positive(self, tmp_path):
        from core.paper_validation import calibration_close_tracking
        rows = [(0.01, "qp-current-22222222")] * 30
        db = self._make_db(tmp_path, rows)
        r = calibration_close_tracking(db, version="qp-current-22222222")
        assert r["n_closes"] == 30
        assert r["conditions_met"] is True
        assert r["progress_pct"] == 100.0

    def test_30_closes_negative_expectancy_not_met(self, tmp_path):
        from core.paper_validation import calibration_close_tracking
        rows = [(-0.01, "qp-current-33333333")] * 30
        db = self._make_db(tmp_path, rows)
        r = calibration_close_tracking(db, version="qp-current-33333333")
        assert r["n_closes"] == 30
        assert r["conditions_met"] is False  # expectancy <= 0 : pas de GO

    def test_no_version_counts_all(self, tmp_path):
        from core.paper_validation import calibration_close_tracking
        db = self._make_db(tmp_path, [(0.01, "v1"), (0.02, "v2")])
        r = calibration_close_tracking(db, version="")
        assert r["n_closes"] == 2


# --------------------------------------------------------------------------- #
# PHASE 3 Cycle 3 — persistance de l'équité + bootstrap Sharpe
# --------------------------------------------------------------------------- #
class TestEquityHistory:
    """save/load équité (table equity_history) + bootstrap Sharpe honnête."""

    def _make_db(self, tmp_path):
        import sqlite3

        class MiniDB:
            is_postgres = False

            def __init__(self, p):
                self._conn = sqlite3.connect(str(p))
                self._conn.execute(
                    "CREATE TABLE equity_history (id INTEGER PRIMARY KEY "
                    "AUTOINCREMENT, ts REAL, mode TEXT, equity REAL)")

            def get_connection(self):
                return self._conn

            def save_equity_point(self, mode, equity):
                self._conn.execute(
                    "INSERT INTO equity_history (ts, mode, equity) "
                    "VALUES (?, ?, ?)", (time.time(), mode, float(equity)))
                self._conn.commit()
                return True

            def load_equity_series(self, mode="DEMO", since=0.0, limit=100000):
                cur = self._conn.execute(
                    "SELECT ts, equity FROM equity_history "
                    "WHERE mode = ? AND ts >= ? ORDER BY ts ASC LIMIT ?",
                    (mode, since, limit))
                return [(float(r[0]), float(r[1])) for r in cur.fetchall()]

        return MiniDB(tmp_path / "eq.db")

    def test_save_and_load_chronological(self, tmp_path):
        from database.db_manager import DBManager
        db = DBManager()
        db.ensure_equity_history_table()
        db.save_equity_point("DEMO", 100.0)
        db.save_equity_point("DEMO", 101.0)
        db.save_equity_point("REAL", 500.0)
        s = db.load_equity_series("DEMO")
        assert len(s) == 2
        assert [e for _, e in s] == [100.0, 101.0]      # chronologique
        assert len(db.load_equity_series("REAL")) == 1
        # mode inconnu : vide (pas de fuite entre modes)
        assert db.load_equity_series("AUTRE") == []

    def test_bootstrap_insufficient_honest(self, tmp_path):
        from core.benchmark import bootstrap_sharpe
        db = self._make_db(tmp_path)
        r = bootstrap_sharpe(db, mode="DEMO")
        assert r["sharpe_obs"] is None        # jamais de chiffre inventé
        assert "insuffisant" in (r["note"] or "")

    def test_bootstrap_with_20_days(self, tmp_path):
        import numpy as np

        from core.benchmark import bootstrap_sharpe
        db = self._make_db(tmp_path)
        rng = np.random.default_rng(3)
        eq = 100.0
        t0 = 1_700_000_000
        for i in range(25):
            db._conn.execute(
                "INSERT INTO equity_history (ts, mode, equity) VALUES (?, ?, ?)",
                (t0 + i * 86400, "DEMO", eq))
            eq *= 1.0 + float(rng.normal(0.001, 0.01))   # dérive positive
        db._conn.commit()
        r = bootstrap_sharpe(db, mode="DEMO", n_sims=500)
        assert r["n_days"] == 24             # 25 points -> 24 rendements
        assert r["sharpe_obs"] is not None
        assert r["sharpe_obs"] > 0.0         # dérive positive mesurable
        assert r["sharpe_p5"] <= r["sharpe_obs"] <= r["sharpe_p95"]
        assert r["n_points"] == 25

    def test_daily_report_exposes_equity_bootstrap(self):
        from core.daily_quant_report import build_daily_quant_report
        r = build_daily_quant_report(FakeDB(), {"mode": "DEMO",
                                                "regime_name": "Range",
                                                "regime_id": 2})
        assert "equity_bootstrap" in r["trading"]
        assert r["trading"]["equity_bootstrap"] is not None


# --------------------------------------------------------------------------- #
# PHASE 3 Cycle 3 — wrapper SQLite : la connexion est FERMÉE après le `with`
# (cause racine du flaky test_routes_health : fuite de connexions sous charge)
# --------------------------------------------------------------------------- #
class TestSQLiteConnClosed:
    def test_conn_closed_after_with(self):
        import sqlite3

        from database.db_manager import DBManager
        db = DBManager()
        conn = db.get_connection()
        with conn:
            conn.execute("SELECT 1")
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")   # fermée : usage post-bloc impossible

    def test_conn_still_usable_without_with(self):
        from database.db_manager import DBManager
        db = DBManager()
        conn = db.get_connection()
        try:
            assert conn.execute("SELECT 1").fetchone() is not None
        finally:
            conn.close()

    def test_explicit_close_still_works(self):
        from database.db_manager import DBManager
        db = DBManager()
        conn = db.get_connection()
        conn.close()   # délégation vers sqlite3 : pas d'erreur
