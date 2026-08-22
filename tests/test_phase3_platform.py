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
import json
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


# --------------------------------------------------------------------------- #
# PHASE 3 Cycle 4 — rapport de friction (chantier slippage p95)
# --------------------------------------------------------------------------- #
class TestFrictionReport:
    def _db_with_fills(self, tmp_path, n=100):
        import sqlite3

        class MiniDB:
            is_postgres = False

            def __init__(self, p, rows):
                self._conn = sqlite3.connect(str(p))
                self._conn.execute(
                    "CREATE TABLE events (id INTEGER PRIMARY KEY "
                    "AUTOINCREMENT, ts REAL, event_type TEXT, payload TEXT)")
                self._conn.executemany(
                    "INSERT INTO events (ts, event_type, payload) "
                    "VALUES (?, 'paper_fill', ?)",
                    [(i, json.dumps(r)) for i, r in enumerate(rows)])
                self._conn.commit()

            def get_connection(self):
                return self._conn

            def list_events(self, event_type="", since=0.0, limit=500):
                cur = self._conn.execute(
                    "SELECT ts, event_type, payload FROM events "
                    "WHERE event_type = ? ORDER BY ts DESC LIMIT ?",
                    (event_type, limit))
                return [{"ts": r[0], "event_type": r[1], "payload": r[2]}
                        for r in cur.fetchall()]

        rows = [{"symbol": "BTCUSDT", "side": "BUY", "qty": 0.001,
                 "arrival": 100.0, "fill": 100.0 + i * 0.01,
                 "slippage_bps": float(i % 10), "latency_ms": 70.0 + i % 5,
                 "fee": 0.0001}
                for i in range(n)]
        return MiniDB(tmp_path / "f.db", rows)

    def test_insufficient_honest(self, tmp_path):
        from core.execution_intel import friction_report
        db = self._db_with_fills(tmp_path, n=3)
        r = friction_report(db)
        assert r["n_fills"] == 3
        assert "insuffisant" in (r["note"] or "")
        assert r["slippage_bps"] == {}

    def test_distribution_and_fee(self, tmp_path):
        from core.execution_intel import friction_report
        db = self._db_with_fills(tmp_path, n=100)
        r = friction_report(db)
        assert r["n_fills"] == 100
        assert r["slippage_bps"]["p95"] == 9.0        # i%10 -> p95 = 9
        assert r["slippage_bps"]["max"] == 9.0
        # notional = 0.001 qty × 100 = 0.1 $ ; fee 0.0001 $ -> 0.1 %
        assert r["fee_pct"] == pytest.approx(0.1, abs=1e-6)
        assert "BTCUSDT" in r["by_symbol"]

    def test_daily_report_exposes_friction(self):
        from core.daily_quant_report import build_daily_quant_report
        r = build_daily_quant_report(FakeDB(), {"mode": "DEMO",
                                                "regime_name": "Range",
                                                "regime_id": 2})
        assert "friction" in r["execution"]


# --------------------------------------------------------------------------- #
# PHASE 3 Cycle 5 — GATE DE FRICTION (décision opérateur EURUSD micro-taille)
# --------------------------------------------------------------------------- #
class TestFrictionGate:
    def test_expected_cost_formula(self):
        from core.execution_intel import expected_roundtrip_cost_pct
        # fee 0,1 %/side ×2 + slippage p95 ×2 : AAPL 7 bps -> 0,34 %
        assert expected_roundtrip_cost_pct(0.1, 7.01) == \
            pytest.approx(0.3402, abs=1e-6)
        # EURUSD 157 bps -> 3,35 %
        assert expected_roundtrip_cost_pct(0.1, 157.47) == \
            pytest.approx(3.3494, abs=1e-6)
        # pas de mesure -> None (jamais 0 inventé)
        assert expected_roundtrip_cost_pct(0.1, None) is None

    def test_gate_blocks_heavy_tail(self):
        from core.execution_intel import friction_gate_blocks
        cache = {"EURUSD": 157.47, "SOLUSDT": 75.26, "AAPL": 7.01}
        blk, reason, cost = friction_gate_blocks("EURUSD", cache, 1.0)
        assert blk is True and "friction" in reason and cost > 1.0
        blk, _, _ = friction_gate_blocks("SOLUSDT", cache, 1.0)
        assert blk is True
        blk, _, cost = friction_gate_blocks("AAPL", cache, 1.0)
        assert blk is False and cost < 1.0

    def test_no_measure_no_block(self):
        from core.execution_intel import friction_gate_blocks
        blk, reason, cost = friction_gate_blocks("BTCUSDT", {}, 1.0)
        assert blk is False and reason is None and cost is None

    def test_gate_wired_in_main_after_netting_before_journal(self):
        """Le gate friction est APRÈS le netting (dernière gate) et AVANT le
        journal de décision (fidélité du journal préservée)."""
        src = (Path(__file__).parent.parent / "main.py").read_text(encoding="utf-8")
        i_fric = src.find("GATE DE FRICTION")
        i_net = src.find("netting: retournement sans signal fort")
        i_journal = src.find("_dj_id = journal_decision(")
        assert i_fric != -1 and i_net != -1 and i_journal != -1
        assert i_net < i_fric < i_journal
        assert "friction_gate_blocks(symbol" in src

    def test_no_trade_bucket_friction(self):
        from core.meta_cognition import _no_trade_bucket
        assert _no_trade_bucket("friction: coût AR attendu 3.35% > seuil") == \
            "friction"

    def test_config_default_present(self):
        import yaml
        cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml",
                                  encoding="utf-8"))
        assert cfg["execution"]["max_expected_roundtrip_cost_pct"] == 1.0
        from core.config import settings
        assert settings.get_float("execution",
                                  "max_expected_roundtrip_cost_pct", 1.0) == 1.0


# --------------------------------------------------------------------------- #
# PHASE 4 P4-F (consolidation) — persistance du prix tick dans le journal
# --------------------------------------------------------------------------- #
class TestJournalPricePersistence:
    """Le prix tick et la taille cible sont persistés à la décision (le champ
    existait mais n'était jamais rempli) — le replay du Digital Twin devient
    exact pour les nouvelles décisions."""

    def test_journal_decision_persists_price_qty(self):
        import sqlite3

        from core.decision_journal import journal_decision

        class MiniDB:
            is_postgres = False

            def __init__(self):
                self._conn = sqlite3.connect(":memory:")
                self._conn.execute(
                    "CREATE TABLE decision_journal (id INTEGER PRIMARY KEY "
                    "AUTOINCREMENT, ts REAL, decision TEXT, symbol TEXT, "
                    "regime TEXT, signal REAL, conviction REAL, level TEXT, "
                    "edge_net REAL, win_rate REAL, reason TEXT, detail TEXT, "
                    "threshold REAL, risk_state TEXT, strategy TEXT, "
                    "qty REAL, price REAL, slippage_bps_expected REAL, "
                    "slippage_bps_real REAL, pnl_pct REAL, mfe_pct REAL, "
                    "mae_pct REAL, duration_sec REAL, exit_reason TEXT, "
                    "payload TEXT, system_version TEXT, config_hash TEXT)")

            def get_connection(self):
                return self._conn

            def log_decision_entry(self, entry):
                cols = list(entry.keys())
                ph = ",".join("?" for _ in cols)
                self._conn.execute(
                    f"INSERT INTO decision_journal ({','.join(cols)}) "
                    f"VALUES ({ph})", list(entry.values()))
                return self._conn.execute(
                    "SELECT last_insert_rowid()").fetchone()[0]

        db = MiniDB()
        eid = journal_decision(
            db, "WAIT", "BTCUSDT", "Range", 0.05, 0.03, "LOW", 0.001, 0.5,
            "conviction", "détail", 0.08, "NORMAL", price=64240.25)
        assert eid > 0
        row = db._conn.execute(
            "SELECT price, qty FROM decision_journal WHERE id = ?",
            (eid,)).fetchone()
        assert row[0] == 64240.25
        # WAIT : pas de qty
        assert row[1] is None
        # TRADE avec qty
        eid2 = journal_decision(
            db, "TRADE", "BTCUSDT", "Range", 0.3, 0.25, "HIGH", 0.01, 0.5,
            "conviction", "détail", 0.08, "NORMAL", qty=0.001,
            price=64000.0)
        row2 = db._conn.execute(
            "SELECT price, qty FROM decision_journal WHERE id = ?",
            (eid2,)).fetchone()
        assert row2[0] == 64000.0 and row2[1] == 0.001

    def test_main_passes_price_to_journal(self):
        """L'appel de production passe price=current_price (vérifié dans le
        source — pas de régression silencieuse)."""
        src = (Path(__file__).parent.parent / "main.py").read_text(
            encoding="utf-8")
        i_call = src.find("_dj_id = journal_decision(")
        i_price = src.find("price=current_price,", i_call)
        assert i_call != -1 and i_price != -1
        assert i_call < i_price
        # le qty n'est passé que pour les TRADE
        assert "qty=(abs(target_qty) if target_direction != 0.0" in src
