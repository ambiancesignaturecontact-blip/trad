"""
PHASE 4 — P4-E : tests du Version Benchmark (core/version_benchmark.py).

Couvre : métriques par version (expectancy, win rate, drawdown, médiane),
gate PURE (PROMOTED / NOT_PROMOTED / INSUFFICIENT), tri chronologique,
listing des versions sans clôture (transparence), câblage Daily Report.
"""
import sqlite3

import pytest


class MiniDB:
    is_postgres = False

    def __init__(self, rows):
        """rows : list of (system_version, pnl_pct, ts, duration_sec)."""
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute(
            "CREATE TABLE decision_journal (id INTEGER PRIMARY KEY, ts REAL, "
            "system_version TEXT, pnl_pct REAL, duration_sec REAL, symbol TEXT)")
        for i, r in enumerate(rows):
            self._conn.execute(
                "INSERT INTO decision_journal (id, ts, system_version, "
                "pnl_pct, duration_sec, symbol) VALUES (?,?,?,?,?,?)",
                (i + 1, r[2], r[0], r[1], r[3] if len(r) > 3 else None, "X"))

    def get_connection(self):
        return self._conn


def _v30(version, pnl=0.001):
    """30 clôtures de la version, ts croissants, pnl constant."""
    return [(version, pnl, 1_700_000_000 + i * 100) for i in range(30)]


class TestVersionPerformance:
    def test_metrics(self):
        from core.version_benchmark import version_performance
        closes = [{"ts": i, "pnl_pct": 0.01, "duration_sec": 60.0}
                  for i in range(30)]
        p = version_performance(closes)
        assert p["n_closes"] == 30
        assert p["status"] == "OK"
        assert p["win_rate"] == 1.0
        assert p["expectancy_pct"] == pytest.approx(1.0, abs=1e-6)
        assert p["cumulative_pnl_pct"] == pytest.approx(30.0, abs=1e-6)
        assert p["max_drawdown_pct"] == 0.0      # jamais négatif
        assert p["avg_duration_sec"] == pytest.approx(60.0)

    def test_drawdown_computed(self):
        from core.version_benchmark import version_performance
        # pertes d'abord puis gains : le drawdown est négatif
        closes = [{"ts": i, "pnl_pct": -0.01} for i in range(20)]
        p = version_performance(closes)
        assert p["max_drawdown_pct"] < 0

    def test_insufficient_below_30(self):
        from core.version_benchmark import version_performance
        p = version_performance([{"ts": 1, "pnl_pct": 0.01}])
        assert p["status"] == "INSUFFICIENT"

    def test_zero_closes(self):
        from core.version_benchmark import version_performance
        p = version_performance([])
        assert p["n_closes"] == 0
        assert p["expectancy_pct"] is None


class TestVersionGate:
    def _perf(self, n, exp, dd):
        return {"n_closes": n, "expectancy_pct": exp,
                "max_drawdown_pct": dd}

    def test_promoted_when_better(self):
        from core.version_benchmark import version_gate
        g = version_gate(self._perf(40, 0.5, -5.0),
                         self._perf(40, 0.2, -8.0))
        assert g["decision"] == "PROMOTED"

    def test_not_promoted_when_edge_worse(self):
        from core.version_benchmark import version_gate
        g = version_gate(self._perf(40, 0.1, -5.0),
                         self._perf(40, 0.2, -5.0))
        assert g["decision"] == "NOT_PROMOTED"

    def test_not_promoted_when_drawdown_worse(self):
        from core.version_benchmark import version_gate
        g = version_gate(self._perf(40, 0.5, -20.0),
                         self._perf(40, 0.2, -8.0))
        assert g["decision"] == "NOT_PROMOTED"

    def test_insufficient_any_side(self):
        from core.version_benchmark import version_gate
        g = version_gate(self._perf(29, 0.5, -5.0),
                         self._perf(40, 0.2, -8.0))
        assert g["decision"] == "INSUFFICIENT"
        g = version_gate(self._perf(40, 0.5, -5.0),
                         self._perf(29, 0.2, -8.0))
        assert g["decision"] == "INSUFFICIENT"

    def test_negative_expectancy_comparison(self):
        """Amélioration d'une expectancy négative (-0.05 -> -0.02) :
        la nouvelle version est MEILLEURE (moins négative)."""
        from core.version_benchmark import version_gate
        g = version_gate(self._perf(40, -0.02, -5.0),
                         self._perf(40, -0.05, -5.0))
        assert g["decision"] == "PROMOTED"


class TestReport:
    def test_versions_listed_even_without_closes(self):
        from core.version_benchmark import version_benchmark_report
        db = MiniDB([("v1", None, 100.0), ("v2", None, 200.0)])
        r = version_benchmark_report(db)
        assert r["n_versions"] == 2
        assert all(v["n_closes"] == 0 for v in r["versions"])

    def test_chronological_and_gate(self):
        from core.version_benchmark import version_benchmark_report
        rows = _v30("v1", 0.001) + _v30("v2", 0.002)
        r = version_benchmark_report(db=MiniDB(rows), current_version="v2")
        assert r["n_versions"] == 2
        v1, v2 = r["versions"]
        assert v1["version"] == "v1" and v2["version"] == "v2"
        assert v2["is_current"] is True
        assert v1["gate_vs_previous"] is None
        assert v2["gate_vs_previous"]["decision"] == "PROMOTED"

    def test_legacy_grouped(self):
        from core.version_benchmark import version_benchmark_report
        db = MiniDB([(None, 0.01, 100.0)])
        r = version_benchmark_report(db)
        assert r["versions"][0]["version"] == "legacy_sans_version"


class TestWiring:
    def test_daily_report_exposes_version_benchmark(self):
        from core.daily_quant_report import build_daily_quant_report

        class FakeDB:
            def decision_journal_summary(self):
                return {"total": 0, "by_decision": {}, "by_reason": {},
                        "closed_n": 0}
            def list_events(self, event_type="", since=0.0, limit=500):
                return []
            def save_setting(self, k, v):
                pass
            def get_setting(self, k, decrypt=False):
                return ""
            def get_connection(self):
                return None

        r = build_daily_quant_report(FakeDB(), {"mode": "DEMO",
                                                "regime_name": "Range",
                                                "regime_id": 2})
        assert "version_benchmark" in r["intelligence"]
