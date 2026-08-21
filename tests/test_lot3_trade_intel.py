"""
LOT 3 (mandat — Trade Intelligence) : Decision Journal, Trade Intelligence
Database, P&L attribution, non-trade analysis.

Vérifié ici :
  1. DB : table decision_journal créée ; log_decision_entry retourne un id ;
     update_decision_outcome complète (pnl, durée, MFE/MAE) ; get_decision_journal
     filtre ; decision_journal_summary agrège (nb par décision/raison, win rate
     des clôturés).
  2. core/decision_journal : journal_decision/journal_fill/journal_close
     construisent et complètent les entrées ; mfe_mae_from_candles (None si
     données insuffisantes — jamais inventé) ; close_journal_entry finalise
     et retire la référence ; non_trade_analysis.
  3. API : /api/v1/decision-journal et /api/v1/non-trade répondent.
  4. Télémétrie : decision_journal_summary exposé.
  5. Câblage main.py : journal_decision à la décision, journal_fill à
     l'exécution, close_journal_entry à la clôture.
  6. DÉMO == RÉAL : aucun flag de mode.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from core.decision_journal import (  # noqa: E402
    close_journal_entry,
    journal_close,
    journal_decision,
    journal_fill,
    mfe_mae_from_candles,
    non_trade_analysis,
)


class FakeDB:
    """Mini DB en mémoire imitant les méthodes du journal (pas de SQLite)."""

    def __init__(self):
        self.rows = []
        self._next = 1

    def log_decision_entry(self, entry: dict) -> int:
        row = dict(entry)
        row["id"] = self._next
        self._next += 1
        self.rows.append(row)
        return row["id"]

    def update_decision_outcome(self, entry_id, outcome: dict) -> bool:
        for r in self.rows:
            if r["id"] == entry_id:
                r.update({k: v for k, v in outcome.items() if v is not None})
                return True
        return False

    def load_candles(self, symbol, limit=200):
        import numpy as np
        import pandas as pd
        idx = pd.date_range("2026-01-01", periods=48, freq="h")
        close = np.linspace(100, 110, 48)
        return pd.DataFrame({"open": close, "high": close + 1,
                             "low": close - 1, "close": close,
                             "volume": np.full(48, 1000.0)}, index=idx)


# --------------------------------------------------------------------------- #
# 1. Journal : construction + complétion
# --------------------------------------------------------------------------- #
class TestJournal:
    def test_decision_creates_entry_and_returns_id(self):
        db = FakeDB()
        eid = journal_decision(db, "WAIT", "BTCUSDT", "Range", 0.05, 0.04,
                                "NO_TRADE", None, None, "conviction",
                                "|signal| 0.050 < seuil 0.080", 0.08, "NORMAL")
        assert eid == 1
        assert db.rows[0]["decision"] == "WAIT"
        assert db.rows[0]["symbol"] == "BTCUSDT"
        assert db.rows[0]["conviction"] == 0.04

    def test_fill_completes_entry(self):
        db = FakeDB()
        eid = journal_decision(db, "TRADE", "BTCUSDT", "Range", 0.2, 0.18,
                                "NORMAL", 0.25, 0.55, "opportunity_ok", "d", 0.08, "NORMAL")
        journal_fill(db, eid, 0.001, 60000.0, slippage_bps_real=4.5)
        row = db.rows[0]
        assert row["qty"] == 0.001
        assert row["price"] == 60000.0
        assert row["slippage_bps_real"] == 4.5

    def test_close_finalizes(self):
        db = FakeDB()
        eid = journal_decision(db, "TRADE", "BTCUSDT", "Range", 0.2, 0.18,
                                "NORMAL", 0.25, 0.55, "opportunity_ok", "d", 0.08, "NORMAL")
        journal_close(db, eid, pnl_pct=0.012, duration_sec=3600.0,
                      exit_reason="SELL", mfe_pct=2.0, mae_pct=-0.5)
        row = db.rows[0]
        assert row["pnl_pct"] == 0.012
        assert row["duration_sec"] == 3600.0
        assert row["exit_reason"] == "SELL"

    def test_close_journal_entry_finalizes_and_pops(self):
        db = FakeDB()
        eid = journal_decision(db, "TRADE", "BTCUSDT", "Range", 0.2, 0.18,
                                "NORMAL", 0.25, 0.55, "opportunity_ok", "d", 0.08, "NORMAL")
        state = {"decision_journal_per_symbol": {"BTCUSDT": {"id": eid, "ts": 1_000_000.0}}}
        close_journal_entry(db, state, "BTCUSDT", 100.0, 102.0, "SELL", 0.02)
        assert "BTCUSDT" not in state["decision_journal_per_symbol"]
        row = db.rows[0]
        assert row["pnl_pct"] == 0.02
        assert row["exit_reason"] == "SELL"

    def test_never_raises(self):
        db = FakeDB()
        close_journal_entry(db, {"decision_journal_per_symbol": {}}, "X", 1, 2, "SELL", 0.0)
        journal_decision(None, "TRADE", "X", "R", 0.2, 0.18, "NORMAL", 0.25, 0.55,
                         "r", "d", 0.08, "NORMAL")  # db None -> 0, pas d'exception


# --------------------------------------------------------------------------- #
# 2. MFE/MAE
# --------------------------------------------------------------------------- #
class TestMfeMae:
    def test_long_position_mfe_mae(self):
        import pandas as pd
        df = pd.DataFrame({"high": [100.0, 105.0, 103.0], "low": [99.0, 101.0, 100.5]})
        mfe, mae = mfe_mae_from_candles(df, 100.0, 102.0, "SELL")  # SELL clôt un long
        assert mfe == pytest.approx(5.0, abs=1e-6)   # (105-100)/100*100
        assert mae == pytest.approx(-1.0, abs=1e-6)  # (99-100)/100*100

    def test_insufficient_data_none(self):
        mfe, mae = mfe_mae_from_candles(None, 100.0, 102.0, "SELL")
        assert mfe is None and mae is None
        mfe, mae = mfe_mae_from_candles("garbage", 100.0, 102.0, "SELL")
        assert mfe is None and mae is None


# --------------------------------------------------------------------------- #
# 3. Non-trade analysis
# --------------------------------------------------------------------------- #
class TestNonTrade:
    def test_analysis_from_state(self):
        state = {"no_trade_stats": {"count": 42, "reasons": {"conviction": 30, "halt": 12}},
                 "last_no_trade_reasons": ["r1", "r2"]}
        a = non_trade_analysis(state)
        assert a["count"] == 42
        assert a["by_reason"]["conviction"] == 30
        assert a["last_reasons"] == ["r1", "r2"]

    def test_analysis_empty_honest(self):
        a = non_trade_analysis({})
        assert a["count"] == 0
        assert a["by_reason"] == {}


# --------------------------------------------------------------------------- #
# 4. API + télémétrie + câblage main
# --------------------------------------------------------------------------- #
class TestIntegration:
    def test_api_endpoints_respond(self):
        from fastapi.testclient import TestClient

        import main  # noqa: F401
        with TestClient(main.app) as c:
            r = c.get("/api/v1/decision-journal")
            assert r.status_code == 200
            body = r.json()
            assert "decisions" in body and "summary" in body
            assert "by_decision" in body["summary"]
            r2 = c.get("/api/v1/non-trade")
            assert r2.status_code == 200
            assert "count" in r2.json()

    def test_telemetry_exposes_summary(self):
        import main  # noqa: F401
        from telemetry import compile_telemetry_data
        tel = compile_telemetry_data()
        assert "decision_journal_summary" in tel
        assert isinstance(tel["decision_journal_summary"], dict)

    def test_main_wiring(self):
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        assert "journal_decision(" in src
        assert "journal_fill(db" in src
        assert "close_journal_entry(db, STATE" in src
        assert "decision_journal_per_symbol" in src

    def test_db_table_real(self):
        """La VRAIE DB de test crée la table et accepte une ligne (puis la
        retire — pas de pollution)."""
        import main  # noqa: F401
        db = main.db
        db.ensure_decision_journal_table()
        eid = db.log_decision_entry({
            "ts": 1.0, "decision": "WAIT", "symbol": "TESTLOT3", "regime": "Range",
            "signal": 0.05, "conviction": 0.04, "level": "NO_TRADE", "edge_net": None,
            "win_rate": None, "reason": "conviction", "detail": "test",
            "threshold": 0.08, "risk_state": "NORMAL", "strategy": "", "qty": None,
            "price": None, "slippage_bps_expected": None, "payload": "{}"})
        assert eid > 0
        rows = db.get_decision_journal(decision="WAIT", limit=10)
        assert any(r["symbol"] == "TESTLOT3" for r in rows)
        assert db.update_decision_outcome(eid, {"pnl_pct": 0.01, "exit_reason": "TEST"}) is True
        # nettoyage (la base de test est par ailleurs isolée par conftest)
        with db.get_connection() as conn:
            cur = conn.cursor()
            if db.is_postgres:
                cur.execute("DELETE FROM decision_journal WHERE symbol = %s", ("TESTLOT3",))
            else:
                cur.execute("DELETE FROM decision_journal WHERE symbol = ?", ("TESTLOT3",))
            conn.commit()

    def test_no_mode_flag(self):
        import inspect
        src = inspect.getsource(close_journal_entry) + inspect.getsource(journal_decision)
        assert "active_mode" not in src
        assert '"DEMO"' not in src and '"REAL"' not in src
