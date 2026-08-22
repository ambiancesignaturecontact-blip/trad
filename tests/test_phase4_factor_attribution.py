"""
PHASE 4 — P4-B : tests du module Factor Attribution (core/factor_attribution.py).

Couvre : insuffisance honnête (0 clôture actuelle), statistiques robustes
(médiane vs moyenne), décomposition alpha/bêta, nettoyage des payloads
incomplets, source historique avec réserve, exposition Daily Report.
"""
import json

import pytest


class MiniDB:
    """Mini DB : decision_journal + events closed_trade_alpha."""

    is_postgres = False

    def __init__(self, journal_closes=None, alpha_events=None):
        import sqlite3
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute(
            "CREATE TABLE decision_journal (id INTEGER PRIMARY KEY, "
            "symbol TEXT, strategy TEXT, pnl_pct REAL, exit_reason TEXT, "
            "system_version TEXT)")
        for i, c in enumerate(journal_closes or []):
            self._conn.execute(
                "INSERT INTO decision_journal (symbol, strategy, pnl_pct, "
                "exit_reason, system_version) VALUES (?, ?, ?, ?, ?)",
                (c.get("symbol", "X"), c.get("strategy", "S"),
                 c["pnl_pct"], c.get("exit_reason", "SL"),
                 c.get("system_version", "v")))
        self._events = list(alpha_events or [])

    def get_connection(self):
        return self._conn

    def list_events(self, event_type="", since=0.0, limit=500):
        return [{"payload": json.dumps(e)} for e in self._events
                if e.get("_type", "closed_trade_alpha") == event_type
                or event_type == ""][:limit]


def _mk_events(n=20, pnl=0.01, alpha=0.005):
    """n trades : pnl constant + alpha constant -> bench = pnl - alpha."""
    return [{"_type": "closed_trade_alpha", "symbol": "BTCUSDT",
             "strategy": "M", "pnl_pct": pnl, "marginal_alpha": alpha}
            for _ in range(n)]


class TestAttribution:
    def test_insufficient_honest(self):
        from core.factor_attribution import alpha_beta_report
        db = MiniDB(journal_closes=[])
        r = alpha_beta_report(db)
        assert r["current"]["n_closes"] == 0
        assert r["current"]["attribution"]["status"] == "INSUFFICIENT"
        assert r["historical"]["n_closes"] == 0

    def test_current_insufficient_but_historical_ok(self):
        from core.factor_attribution import alpha_beta_report
        db = MiniDB(journal_closes=[], alpha_events=_mk_events(n=20))
        r = alpha_beta_report(db)
        assert r["current"]["attribution"]["status"] == "INSUFFICIENT"
        assert r["historical"]["attribution"]["status"] == "OK"

    def test_alpha_beta_decomposition(self):
        """pnl 1 % + alpha 0,5 % sur 20 trades : bench = 0,5 % (par diff.),
        part alpha = 50 %, bêta = 50 % (sommes égales)."""
        from core.factor_attribution import alpha_beta_report
        db = MiniDB(alpha_events=_mk_events(n=20, pnl=0.01, alpha=0.005))
        a = alpha_beta_report(db)["historical"]["attribution"]
        assert a["n_trades"] == 20
        assert a["pnl"]["mean_pct"] == pytest.approx(1.0, abs=1e-6)
        assert a["alpha"]["mean_pct"] == pytest.approx(0.5, abs=1e-6)
        assert a["beta"]["mean_pct"] == pytest.approx(0.5, abs=1e-6)
        assert a["share_alpha_pct"] == pytest.approx(50.0, abs=1e-3)
        assert a["share_beta_pct"] == pytest.approx(50.0, abs=1e-3)

    def test_pct_alpha_positive(self):
        from core.factor_attribution import alpha_beta_report
        evs = [{"_type": "closed_trade_alpha", "symbol": "X",
                "strategy": "S", "pnl_pct": -0.01, "marginal_alpha": 0.001}
               for _ in range(10)]
        evs += [{"_type": "closed_trade_alpha", "symbol": "X",
                 "strategy": "S", "pnl_pct": -0.01, "marginal_alpha": -0.001}
                for _ in range(10)]
        db = MiniDB(alpha_events=evs)
        a = alpha_beta_report(db)["historical"]["attribution"]
        assert a["pct_trades_alpha_positive"] == 50.0

    def test_dirty_payloads_ignored(self):
        """Payloads incomplets (pnl manquant) ou NaN ignorés — pas de crash."""
        from core.factor_attribution import alpha_beta_report
        evs = _mk_events(n=10)
        evs += [{"_type": "closed_trade_alpha", "symbol": "X"}]  # incomplet
        evs += [{"_type": "closed_trade_alpha", "symbol": "X",
                 "strategy": "S", "pnl_pct": float("nan"),
                 "marginal_alpha": 0.0}]
        db = MiniDB(alpha_events=evs)
        r = alpha_beta_report(db)
        assert r["historical"]["n_closes"] == 10

    def test_historical_has_reserve(self):
        from core.factor_attribution import alpha_beta_report
        db = MiniDB(alpha_events=_mk_events(n=10))
        h = alpha_beta_report(db)["historical"]
        assert "reserve" in h
        assert "ancien calibrage" in h["reserve"].lower()
        assert "reconstruit" in h["reserve"].lower()


class TestWiring:
    def test_daily_report_exposes_factor_attribution(self):
        from core.daily_quant_report import build_daily_quant_report

        class FakeStateDB:
            def decision_journal_summary(self):
                return {"total": 0, "by_decision": {}, "by_reason": {},
                        "closed_n": 0}
            def list_events(self, event_type="", since=0.0, limit=500):
                return []
            def save_setting(self, k, v):
                pass
            def get_setting(self, k, decrypt=False):
                return ""
            def load_candles(self, symbol, limit=200):
                import numpy as np
                import pandas as pd
                n = 100
                close = np.linspace(100.0, 110.0, n)
                idx = pd.date_range("2026-01-01", periods=n, freq="h")
                return pd.DataFrame({"open": close, "high": close + 1,
                                     "low": close - 1, "close": close,
                                     "volume": np.full(n, 1000.0)}, index=idx)
            def get_connection(self):
                return self._conn if hasattr(self, "_conn") else None

        r = build_daily_quant_report(FakeStateDB(), {"mode": "DEMO",
                                                     "regime_name": "Range",
                                                     "regime_id": 2})
        assert "factor_attribution" in r["intelligence"]
