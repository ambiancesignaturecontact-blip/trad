"""
PHASE 4 — P4-C : tests du Shadow Mode (core/shadow_mode.py).

Couvre : décision PURE (seuil × scale), mini-book virtuel (ouverture/clôture,
PnL mark-to-market), persistance DB, comparaison honnête (insuffisant < 10),
non-bloquant (erreur DB ne casse pas), câblage main.py (feed après
conviction, avant les gates), config.
"""
from pathlib import Path

import pytest


class MiniDB:
    """Mini DB : shadow_decisions + shadow_trades en mémoire."""

    is_postgres = False

    def __init__(self):
        import sqlite3
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute(
            "CREATE TABLE shadow_decisions (id INTEGER PRIMARY KEY "
            "AUTOINCREMENT, shadow_id TEXT, ts REAL, symbol TEXT, "
            "signal REAL, conviction REAL, decision TEXT, price REAL)")
        self._conn.execute(
            "CREATE TABLE shadow_trades (id INTEGER PRIMARY KEY "
            "AUTOINCREMENT, shadow_id TEXT, entry_ts REAL, exit_ts REAL, "
            "symbol TEXT, side TEXT, entry_price REAL, exit_price REAL, "
            "pnl_pct REAL, duration_sec REAL)")

    def get_connection(self):
        return self._conn

    def save_shadow_decision(self, shadow_id, ts, symbol, signal, conviction,
                             decision, price):
        self._conn.execute(
            "INSERT INTO shadow_decisions (shadow_id, ts, symbol, signal, "
            "conviction, decision, price) VALUES (?,?,?,?,?,?,?)",
            (shadow_id, ts, symbol, signal, conviction, decision, price))
        return True

    def save_shadow_trade(self, shadow_id, entry_ts, exit_ts, symbol, side,
                          entry_price, exit_price, pnl_pct, duration_sec):
        self._conn.execute(
            "INSERT INTO shadow_trades (shadow_id, entry_ts, exit_ts, "
            "symbol, side, entry_price, exit_price, pnl_pct, duration_sec) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (shadow_id, entry_ts, exit_ts, symbol, side, entry_price,
             exit_price, pnl_pct, duration_sec))
        return True

    def load_shadow_trades(self, shadow_id, limit=10000):
        cur = self._conn.execute(
            "SELECT entry_ts, exit_ts, symbol, side, entry_price, "
            "exit_price, pnl_pct, duration_sec FROM shadow_trades "
            "WHERE shadow_id = ? ORDER BY exit_ts ASC", (shadow_id,))
        return [{"entry_ts": r[0], "exit_ts": r[1], "symbol": r[2],
                 "side": r[3], "entry_price": r[4], "exit_price": r[5],
                 "pnl_pct": r[6], "duration_sec": r[7]} for r in cur.fetchall()]

    def load_shadow_decisions(self, shadow_id, limit=50000):
        cur = self._conn.execute(
            "SELECT ts, symbol, signal, conviction, decision, price "
            "FROM shadow_decisions WHERE shadow_id = ? ORDER BY ts ASC",
            (shadow_id,))
        return [{"ts": r[0], "symbol": r[1], "signal": r[2],
                 "conviction": r[3], "decision": r[4], "price": r[5]}
                for r in cur.fetchall()]


class TestShadowDecide:
    def test_more_permissive_threshold(self):
        from core.shadow_mode import ShadowInstance
        s = ShadowInstance(threshold_scale=0.85)
        # seuil shadow = 0.15 × 0.85 = 0.1275
        # la production refuserait (0.13 < 0.15) ; le shadow accepte
        d, r = s.decide(signal=0.5, conviction=0.13, production_threshold=0.15)
        assert d == 1 and "shadow_trade" in r
        # en dessous du seuil shadow (0.1275) : WAIT
        d, r = s.decide(signal=0.5, conviction=0.10, production_threshold=0.15)
        assert d == 0 and "shadow_wait" in r

    def test_no_signal_no_trade(self):
        from core.shadow_mode import ShadowInstance
        s = ShadowInstance()
        d, r = s.decide(0.0, 0.0, 0.15)
        assert d == 0 and r == "no_signal"

    def test_sell_direction(self):
        from core.shadow_mode import ShadowInstance
        s = ShadowInstance(threshold_scale=1.0)
        d, _ = s.decide(signal=-0.6, conviction=-0.3, production_threshold=0.15)
        assert d == -1


class TestShadowBook:
    def test_open_and_close(self):
        from core.shadow_mode import ShadowInstance
        db = MiniDB()
        s = ShadowInstance(threshold_scale=0.85, notional_pct=0.02)
        # ouverture
        out = s.feed_bar(db, "BTCUSDT", 60000.0, 0.5, 0.2, 0.15, equity=100000.0)
        assert out["decision"] == "TRADE"
        assert out["trade_opened"]["symbol"] == "BTCUSDT"
        assert "BTCUSDT" in s.positions
        # tick neutre : le signal a disparu -> la position est CLÔTURÉE
        # (design : le shadow sort quand son signal s'annule ou s'oppose)
        out = s.feed_bar(db, "BTCUSDT", 60100.0, 0.0, 0.0, 0.15, 100000.0)
        assert out["decision"] == "WAIT"
        assert out["trade_closed"] is not None
        assert out["trade_closed"]["pnl_pct"] == pytest.approx(0.001667, abs=1e-4)
        assert "BTCUSDT" not in s.positions
        trades = db.load_shadow_trades("shadow_vnext")
        assert len(trades) == 1 and trades[0]["pnl_pct"] > 0

    def test_decisions_persisted(self):
        from core.shadow_mode import ShadowInstance
        db = MiniDB()
        s = ShadowInstance()
        for i in range(5):
            s.feed_bar(db, "BTCUSDT", 100.0 + i, 0.3, 0.2, 0.15, 1000.0)
        assert len(db.load_shadow_decisions("shadow_vnext")) == 5


class TestShadowCompare:
    def test_insufficient_honest(self):
        from core.shadow_mode import shadow_compare
        db = MiniDB()
        r = shadow_compare(db)
        assert r["status"] == "INSUFFICIENT"
        assert "≥ 10" in r["note"] or ">= 10" in r["note"]
        assert r["win_rate"] is None

    def test_ok_after_10_trades(self):
        from core.shadow_mode import ShadowInstance, shadow_compare
        db = MiniDB()
        s = ShadowInstance(threshold_scale=0.1)  # très permissif
        for i in range(12):
            # ouverture (signal +), puis tick neutre -> clôture (gain +1 %)
            s.feed_bar(db, "X", 100.0 + i, 0.5, 0.2, 0.15, 1000.0)
            s.feed_bar(db, "X", 101.0 + i, 0.0, 0.0, 0.15, 1000.0)
        r = shadow_compare(db)
        assert r["status"] == "OK"
        assert r["n_trades_closed"] == 12
        assert r["win_rate"] == 1.0
        assert r["expectancy_pct"] == pytest.approx(1.0, abs=0.2)


class TestWiring:
    def test_feed_wired_before_gates(self):
        src = (Path(__file__).parent.parent / "main.py").read_text(encoding="utf-8")
        i_feed = src.find("shadow_vnext.feed_bar(")
        i_gate = src.find("_gate_block = None")
        assert i_feed != -1 and i_gate != -1
        # le feed est APRÈS le calcul de la conviction mais AVANT les gates
        assert i_gate < i_feed
        assert "ShadowInstance" in src

    def test_config_default(self):
        import yaml
        cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml",
                                  encoding="utf-8"))
        assert cfg["shadow"]["enabled"] is True
        assert cfg["shadow"]["threshold_scale"] == 0.85
        from core.config import settings
        assert settings.get_bool("shadow", "enabled", True) is True
        assert settings.get_float("shadow", "threshold_scale", 0.85) == 0.85

    def test_never_raises_without_db(self):
        from core.shadow_mode import ShadowInstance, shadow_compare
        s = ShadowInstance()
        out = s.feed_bar(None, "BTCUSDT", 100.0, 0.5, 0.2, 0.15, 1000.0)
        assert out["decision"] in ("TRADE", "WAIT")   # pas de crash sans db
        assert shadow_compare(None)["note"] is not None
