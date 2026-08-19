"""
LOT 2 — P0 audit indépendant (items 4 et 5).

P0-4 : instrumentation de la distribution réelle de `final_scale` (p10/p50/p90)
       sur 48h glissantes + alerte si p50 < 20% (la chaîne de 17 facteurs
       s'auto-amplifie : 0.8^15 ≈ 3.5% — diagnostic audit §2.1).
P0-5 : tous les scripts/backtests utilisent la MÊME archi LSTM que le live
       (hidden_dim=24) et le même garde-fou anti-biais (audit_backtest avec
       rejet) que l'endpoint /api/run-backtest (audit §4.9).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main


# ---------------------------------------------------------------- P0-5 ----

CLI_SCRIPTS = [
    "run_walk_forward.py",
    "run_simulation_test.py",
    "run_trend_proof.py",
    "run_extended_cycles_backtest.py",
    "run_micro_budget_test.py",
    "optimize_for_profit.py",
]


def test_lstm_hidden_dim_aligned_with_live():
    """Aucune instanciation LSTMLikePredictor en production ne doit diverger
    du live (hidden_dim=24). Le défaut du constructeur est 24."""
    root = Path(main.__file__).parent
    problems = []
    for py in root.rglob("*.py"):
        if "venv" in py.parts or ".git" in py.parts or "tests" in py.parts:
            continue
        src = py.read_text(encoding="utf-8", errors="ignore")
        for line in src.splitlines():
            if "LSTMLikePredictor(" not in line or "def __init__" in line:
                continue
            if line.strip().startswith("#"):   # commentaires ignorés
                continue
            # tolère uniquement hidden_dim=24 ou positionnel (5, 24)
            if "hidden_dim=24" in line or "(5, 24)" in line:
                continue
            problems.append(f"{py.name}:{line.strip()}")
    assert problems == [], f"instances LSTM non alignées sur le live : {problems}"

    from models.price_predictor import LSTMLikePredictor
    m = LSTMLikePredictor()
    assert m.hidden_dim == 24, "le défaut du constructeur doit être 24 (garde-fou §4.9)"


def test_cli_scripts_have_bias_audit_guard():
    """Chaque script CLI importe audit_backtest ET rejette le backtest si
    l'audit des biais échoue (même garde-fou que le live)."""
    root = Path(main.__file__).parent
    for script in CLI_SCRIPTS:
        src = (root / script).read_text(encoding="utf-8")
        assert "from backtester.bias_audit import audit_backtest" in src, \
            f"{script} : import audit_backtest manquant"
        assert "audit_backtest(" in src, f"{script} : appel audit_backtest manquant"
        assert '"REJECTED"' in src, f"{script} : rejet REJECTED manquant"


def test_cli_cost_assumptions_pass_bias_audit():
    """Les coûts déclarés par les scripts (slippage 1-2 bps, frais 2-5 bps)
    passent l'audit — sinon les scripts seraient rejetés en boucle."""
    from backtester.bias_audit import audit_backtest
    idx = pd.date_range("2026-01-01", periods=200, freq="h")
    close = np.cumsum(np.random.RandomState(7).normal(0, 0.01, 200)) + 100
    df = pd.DataFrame({"close": close, "high": close + 1, "low": close - 1,
                       "volume": np.full(200, 10.0), "open": close - 0.5},
                      index=idx)
    for slip_bps, comm in [(1.0, 0.0002), (1.0, 0.0004), (1.0, 0.0005),
                           (2.0, 0.0005)]:
        res = audit_backtest(df, ["BTCUSDT"], ["BTCUSDT"],
                             slippage_bps=slip_bps, commission_pct=comm)
        assert res["status"] == "PASSED", \
            f"coûts ({slip_bps} bps, {comm}) rejetés : {res['issues']}"


# ---------------------------------------------------------------- P0-4 ----

@pytest.fixture(autouse=True)
def _clean_final_scale_state():
    main.STATE["final_scale_samples"] = []
    main.STATE["final_scale_last_ts"] = {}
    main.STATE.pop("final_scale_stats", None)
    yield
    main.STATE["final_scale_samples"] = []
    main.STATE["final_scale_last_ts"] = {}
    main.STATE.pop("final_scale_stats", None)


def test_record_final_scale_downsample_per_symbol(monkeypatch):
    """Au maximum 1 échantillon / 60 s / symbole (borne mémoire)."""
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(main.time, "time", lambda: clock["t"])
    for _ in range(10):
        main._record_final_scale("BTCUSDT", 0.5, 17)
    assert len(main.STATE["final_scale_samples"]) == 1
    # deux symboles -> deux échantillons
    main._record_final_scale("ETHUSDT", 0.4, 17)
    assert len(main.STATE["final_scale_samples"]) == 2
    # après 60 s, un nouveau point pour le même symbole est accepté
    clock["t"] += 61.0
    main._record_final_scale("BTCUSDT", 0.6, 17)
    assert len(main.STATE["final_scale_samples"]) == 3


def test_record_final_scale_upper_bound(monkeypatch):
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(main.time, "time", lambda: clock["t"])
    for i in range(main.FINAL_SCALE_MAX_SAMPLES + 1000):
        main.STATE["final_scale_last_ts"] = {}
        clock["t"] += 61.0  # 1 échantillon / min / symbole
        main._record_final_scale("BTCUSDT", 0.5, 17)
    assert len(main.STATE["final_scale_samples"]) <= main.FINAL_SCALE_MAX_SAMPLES


def test_final_scale_stats_percentiles():
    vals = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    t0 = 1_000_000.0
    for i, v in enumerate(vals):
        main.STATE["final_scale_samples"].append(
            {"ts": t0 + i * 3600, "symbol": "BTCUSDT",
             "final_scale": v, "n_steps": 17})
    stats = main._final_scale_stats()
    assert stats is not None
    assert stats["n"] == 10
    assert stats["p50"] == pytest.approx(0.275, abs=1e-3)   # médiane 10 valeurs
    assert stats["p10"] == pytest.approx(0.095, abs=1e-3)
    assert stats["p90"] == pytest.approx(0.455, abs=1e-3)
    assert stats["min"] == 0.05 and stats["max"] == 0.50
    assert stats["span_hours"] == pytest.approx(9.0, abs=0.01)


def test_final_scale_report_logs_distribution(caplog):
    import logging
    t0 = main.time.time()
    for i in range(60):
        # 60 échantillons étalés sur ~46h (dans la fenêtre 48h) : tous conservés
        main.STATE["final_scale_samples"].append(
            {"ts": t0 - (59 - i) * 2800, "symbol": "BTCUSDT",
             "final_scale": 0.5 + 0.001 * i, "n_steps": 17})
    caplog.set_level(logging.INFO, logger="InstitutionalTradingBot")
    stats = main._final_scale_report()
    assert stats is not None and stats["n"] == 60
    assert "FINAL_SCALE distribution" in caplog.text
    assert "p50=" in caplog.text
    assert main.STATE["final_scale_stats"] == stats


def test_final_scale_p50_below_threshold_warns(caplog):
    """p50 < 20% -> warning explicite (diagnostic audit §2.1)."""
    import logging
    t0 = main.time.time()
    for i in range(30):
        main.STATE["final_scale_samples"].append(
            {"ts": t0 - (29 - i) * 3600, "symbol": "BTCUSDT",
             "final_scale": 0.10 + 0.001 * i, "n_steps": 17})
    caplog.set_level(logging.WARNING, logger="InstitutionalTradingBot")
    main._final_scale_report()
    assert "FINAL_SCALE p50 < 20%" in caplog.text


def test_final_scale_purge_old_samples(monkeypatch):
    """Les échantillons de plus de 48h sont purgés."""
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(main.time, "time", lambda: clock["t"])
    main.STATE["final_scale_samples"] = [
        {"ts": clock["t"] - 50 * 3600, "symbol": "BTCUSDT", "final_scale": 0.1, "n_steps": 17},
        {"ts": clock["t"] - 10 * 3600, "symbol": "BTCUSDT", "final_scale": 0.5, "n_steps": 17},
        {"ts": clock["t"] - 1 * 3600, "symbol": "BTCUSDT", "final_scale": 0.6, "n_steps": 17},
    ]
    main._purge_final_scale_samples(main.FINAL_SCALE_WINDOW_HOURS * 3600.0)
    assert len(main.STATE["final_scale_samples"]) == 2


def test_telemetry_exposes_final_scale_stats():
    client = TestClient(main.app)
    resp = client.get("/api/telemetry")
    assert resp.status_code == 200
    body = resp.json()
    assert "final_scale_stats" in body
    assert "final_scale_samples_count" in body


# ---------------------- durcissement : persistance + accès ------------------

class _StoreDB:
    def __init__(self):
        self.s = {}

    def save_setting(self, k, v, user_id=1, encrypt=False):
        self.s[k] = v

    def get_setting(self, k, user_id=1, decrypt=False):
        return self.s.get(k, "")


def test_final_scale_persistence_survives_restart(monkeypatch):
    """L'observation 24-48h ne doit PAS repartir de zéro à chaque redémarrage."""
    sdb = _StoreDB()
    monkeypatch.setattr(main, "db", sdb)
    t0 = main.time.time()
    main.STATE["final_scale_samples"] = [
        {"ts": t0 - 3600, "symbol": "BTCUSDT", "final_scale": 0.30, "n_steps": 17},
        {"ts": t0, "symbol": "ETHUSDT", "final_scale": 0.55, "n_steps": 17},
    ]
    main._persist_final_scale_samples()
    assert "final_scale_samples_json" in sdb.s
    # redémarrage simulé : état vide, puis rechargement depuis la DB
    main.STATE["final_scale_samples"] = []
    main._load_final_scale_samples()
    assert len(main.STATE["final_scale_samples"]) == 2
    assert main.STATE["final_scale_samples"][1]["symbol"] == "ETHUSDT"
    assert main.STATE["final_scale_samples"][1]["final_scale"] == 0.55


def test_final_scale_load_ignores_corrupted_data(monkeypatch):
    sdb = _StoreDB()
    sdb.s["final_scale_samples_json"] = "not json at all {{{"
    monkeypatch.setattr(main, "db", sdb)
    main.STATE["final_scale_samples"] = []
    main._load_final_scale_samples()  # ne doit pas lever
    assert main.STATE["final_scale_samples"] == []


def test_api_v1_final_scale_endpoint():
    client = TestClient(main.app)
    resp = client.get("/api/v1/final-scale")
    assert resp.status_code == 200
    body = resp.json()
    assert "stats" in body and "samples_count" in body
    assert "window_hours" in body and body["window_hours"] == main.FINAL_SCALE_WINDOW_HOURS
    assert "alert_p50_below_20pct" in body


# ---------------------- données réelles (plus de synthétique) ----------------

def test_cli_scripts_no_synthetic_data():
    """Aucun script CLI ne génère plus de données synthétiques (np.random)."""
    root = Path(main.__file__).parent
    for script in CLI_SCRIPTS:
        src = (root / script).read_text(encoding="utf-8")
        assert "np.random" not in src, f"{script} : génération synthétique restante"
        assert "fetch_real_candles" in src, f"{script} : module données réelles manquant"


def test_live_candles_module_has_no_synthetic_fallback():
    import backtester.live_candles as lc
    src = Path(lc.__file__).read_text(encoding="utf-8")
    assert "np.random" not in src, "live_candles ne doit jamais générer de données"


def test_live_candles_returns_none_when_all_sources_fail(monkeypatch):
    """Si aucune source réelle ne répond -> (None, "") : jamais de données
    fabriquées, les scripts CLI s'arrêtent proprement."""
    import backtester.live_candles as lc

    def boom(symbol, limit):
        raise RuntimeError("offline")

    for src in ("okx", "coinbase", "kraken", "binance"):
        monkeypatch.setitem(lc._FETCHERS, src, boom)
    df, source = lc.fetch_real_candles("BTCUSDT", verbose=False)
    assert df is None
    assert source == ""
