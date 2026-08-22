"""
PHASE 4 — P4-F : tests du Digital Twin (core/digital_twin.py).

Couvre : décision (seuil × scale), sizing borné, gates répliqués (friction +
exposition), exécution simulée (frais + slippage), SL/TP, replay sur
décisions réelles (journal MiniDB), câblage main.py, config.
"""
import sqlite3
from pathlib import Path

import pytest


class MiniDB:
    """decision_journal + candles + tables twin (mémoire)."""

    is_postgres = False

    def __init__(self, rows=None, candles=None):
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute(
            "CREATE TABLE decision_journal (id INTEGER PRIMARY KEY, ts REAL, "
            "symbol TEXT, signal REAL, conviction REAL, threshold REAL, "
            "risk_state TEXT)")
        for i, r in enumerate(rows or []):
            self._conn.execute(
                "INSERT INTO decision_journal (id, ts, symbol, signal, "
                "conviction, threshold, risk_state) VALUES (?,?,?,?,?,?,?)",
                (i + 1, r["ts"], r["symbol"], r["signal"], r["conviction"],
                 r["threshold"], r.get("risk_state", "NORMAL")))
        self._conn.execute(
            "CREATE TABLE twin_decisions (id INTEGER PRIMARY KEY "
            "AUTOINCREMENT, twin_id TEXT, ts REAL, symbol TEXT, "
            "signal REAL, conviction REAL, decision TEXT, price REAL)")
        self._conn.execute(
            "CREATE TABLE twin_trades (id INTEGER PRIMARY KEY "
            "AUTOINCREMENT, twin_id TEXT, entry_ts REAL, exit_ts REAL, "
            "symbol TEXT, side TEXT, entry_price REAL, exit_price REAL, "
            "pnl_pct REAL, duration_sec REAL, exit_reason TEXT)")
        self._candles = candles or {}

    def get_connection(self):
        return self._conn

    def load_candles(self, symbol, limit=200):
        if symbol in self._candles:
            return self._candles[symbol]
        import numpy as np
        import pandas as pd
        idx = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
        close = np.full(100, 100.0)
        return pd.DataFrame({"open": close, "high": close + 1,
                             "low": close - 1, "close": close,
                             "volume": np.full(100, 1000.0)}, index=idx)

    def save_twin_decision(self, twin_id, ts, symbol, signal, conviction,
                           decision, price):
        self._conn.execute(
            "INSERT INTO twin_decisions (twin_id, ts, symbol, signal, "
            "conviction, decision, price) VALUES (?,?,?,?,?,?,?)",
            (twin_id, ts, symbol, signal, conviction, decision, price))
        return True

    def save_twin_trade(self, twin_id, entry_ts, exit_ts, symbol, side,
                        entry_price, exit_price, pnl_pct, duration_sec,
                        exit_reason):
        self._conn.execute(
            "INSERT INTO twin_trades (twin_id, entry_ts, exit_ts, symbol, "
            "side, entry_price, exit_price, pnl_pct, duration_sec, "
            "exit_reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (twin_id, entry_ts, exit_ts, symbol, side, entry_price,
             exit_price, pnl_pct, duration_sec, exit_reason))
        return True

    def load_twin_trades(self, twin_id, limit=10000):
        cur = self._conn.execute(
            "SELECT entry_ts, exit_ts, symbol, side, entry_price, "
            "exit_price, pnl_pct, duration_sec, exit_reason FROM twin_trades "
            "WHERE twin_id = ? ORDER BY exit_ts ASC", (twin_id,))
        return [{"entry_ts": r[0], "exit_ts": r[1], "symbol": r[2],
                 "side": r[3], "entry_price": r[4], "exit_price": r[5],
                 "pnl_pct": r[6], "duration_sec": r[7],
                 "exit_reason": r[8]} for r in cur.fetchall()]


class TestTwinLogic:
    def test_decide_threshold_scale(self):
        from core.digital_twin import TwinEngine
        t = TwinEngine(params={"threshold_scale": 0.85})
        # seuil twin = 0.15 × 0.85 = 0.1275 : 0.13 accepté, 0.10 refusé
        d, r = t.decide(0.5, 0.13, 0.15)
        assert d == 1 and "twin_trade" in r
        d, r = t.decide(0.5, 0.10, 0.15)
        assert d == 0 and "twin_wait" in r

    def test_size_bounded(self):
        from core.digital_twin import TwinEngine
        t = TwinEngine(params={"threshold_scale": 0.85,
                               "max_exposure_pct": 0.05})
        # conviction = seuil -> intensité 1 -> 5 % équité
        s = t.size(100000.0, 0.1275, 0.15)
        assert s == pytest.approx(5000.0, abs=1.0)
        # conviction double du seuil -> plafonné à 5 %
        s = t.size(100000.0, 0.5, 0.15)
        assert s == pytest.approx(5000.0, abs=1.0)
        # conviction nulle -> 0
        assert t.size(100000.0, 0.0, 0.15) == 0.0

    def test_gates_friction_block(self):
        from core.digital_twin import TwinEngine
        t = TwinEngine()
        blk, reason = t._gates_block("EURUSD", "BUY", 1.0, 1.1, 100000.0,
                                     {}, {"EURUSD": 157.0}, [], {})
        assert blk is True and "friction" in reason
        blk, _ = t._gates_block("BTCUSDT", "BUY", 1.0, 60000.0, 100000.0,
                                {}, {}, [], {})
        assert blk is False

    def test_gates_exposure_block(self):
        from core.digital_twin import TwinEngine
        t = TwinEngine()
        positions = [{"symbol": "BTCUSDT", "qty": 0.5, "price": 60000.0}]
        betas = {"BTCUSDT": 1.0, "ETHUSDT": 1.5}
        blk, reason = t._gates_block("ETHUSDT", "BUY", 10.0, 4000.0,
                                     100000.0, betas, {}, positions, {})
        assert blk is True and "exposure" in reason

    def test_gates_disabled(self):
        from core.digital_twin import TwinEngine
        t = TwinEngine(params={"use_friction_gate": False,
                               "use_exposure_gate": False})
        blk, _ = t._gates_block("EURUSD", "BUY", 1.0, 1.1, 100000.0,
                                {}, {"EURUSD": 157.0},
                                [{"symbol": "BTCUSDT", "qty": 0.5,
                                  "price": 60000.0}], {})
        assert blk is False

    def test_execution_price_adverse(self):
        from core.digital_twin import TwinEngine
        t = TwinEngine()
        assert t._execution_price(100.0, "BUY", 10.0) == pytest.approx(
            100.1, abs=1e-6)
        assert t._execution_price(100.0, "SELL", 10.0) == pytest.approx(
            99.9, abs=1e-6)

    def test_feed_open_and_sl(self):
        from core.digital_twin import TwinEngine
        db = MiniDB()
        t = TwinEngine(params={"threshold_scale": 0.85})
        out = t.feed_bar(db, "X", 100.0, 0.5, 0.2, 0.15, 100000.0)
        assert out["decision"] == "TRADE"
        assert "X" in t.positions
        out = t.feed_bar(db, "X", 95.0, 0.5, 0.2, 0.15, 100000.0)
        assert out["trade_closed"] is not None
        assert out["trade_closed"]["exit_reason"] == "SL"
        assert out["trade_closed"]["pnl_pct"] == pytest.approx(-0.034,
                                                               abs=0.005)
        trades = db.load_twin_trades("twin_vnext")
        assert len(trades) == 1 and trades[0]["exit_reason"] == "SL"


class TestReplay:
    def _rows(self, n=40):
        rows = []
        for i in range(n):
            if i % 2 == 0:
                rows.append({"ts": 1_700_000_000 + i * 100, "symbol": "X",
                             "signal": 0.5, "conviction": 0.2,
                             "threshold": 0.15})
            else:
                rows.append({"ts": 1_700_000_000 + i * 100, "symbol": "X",
                             "signal": 0.0, "conviction": 0.0,
                             "threshold": 0.15})
        return rows

    def test_replay_closes_trades(self):
        from core.digital_twin import TwinEngine, replay_journal
        db = MiniDB(rows=self._rows(40))
        t = TwinEngine(twin_id="twin_replay")
        r = replay_journal(db, t, limit=100)
        assert r["n_decisions"] == 40
        assert r["n_trades_closed"] >= 10
        assert r["status"] == "OK"

    def test_replay_insufficient(self):
        from core.digital_twin import TwinEngine, replay_journal
        db = MiniDB(rows=self._rows(6))
        t = TwinEngine(twin_id="twin_replay2")
        r = replay_journal(db, t, limit=100)
        assert r["n_trades_closed"] < 10
        assert r["status"] == "INSUFFICIENT"

    def test_replay_no_rows(self):
        from core.digital_twin import TwinEngine, replay_journal
        db = MiniDB(rows=[])
        r = replay_journal(db, TwinEngine(), limit=100)
        assert "aucune décision" in (r["note"] or "")


class TestWiring:
    def test_feed_wired_and_instance(self):
        src = (Path(__file__).parent.parent / "main.py").read_text(
            encoding="utf-8")
        assert "twin_vnext = TwinEngine(" in src
        assert "twin_vnext.feed_bar(" in src
        assert "from core.digital_twin import TwinEngine" in src

    def test_config_default(self):
        import yaml
        cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml",
                                  encoding="utf-8"))
        assert cfg["digital_twin"]["enabled"] is True
        assert cfg["digital_twin"]["max_exposure_pct"] == 0.05
        from core.config import settings
        assert settings.get_float("digital_twin", "max_exposure_pct",
                                  -1.0) == 0.05

    def test_never_raises_without_db(self):
        from core.digital_twin import TwinEngine, replay_journal
        t = TwinEngine()
        out = t.feed_bar(None, "X", 100.0, 0.5, 0.2, 0.15, 1000.0)
        assert out["decision"] in ("TRADE", "WAIT")
        r = replay_journal(None, t)
        assert r["note"] is not None
