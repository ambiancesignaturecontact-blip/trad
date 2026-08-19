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
def _clean_final_scale_state(monkeypatch):
    # LEÇON APPRISE (P0-4) : AUCUN test de ce fichier ne doit écrire dans la
    # vraie DB — sinon les données de test polluent la collecte réelle
    # (c'est ce qui a faussé le p50=11,45% au boot).
    monkeypatch.setattr(main, "db", _StoreDB())
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
    # borne réduite pour un test rapide (la sérialisation JSON complète à
    # chaque persist est O(n) — 25 000 échantillons réels restent OK en prod)
    monkeypatch.setattr(main, "FINAL_SCALE_MAX_SAMPLES", 100)
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(main.time, "time", lambda: clock["t"])
    for _ in range(1100):
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


def test_final_scale_report_logs_distribution(caplog, monkeypatch):
    import logging
    # NE JAMAIS écrire dans la vraie DB (les données de test pollueraient la
    # collecte réelle — leçon apprise : c'est ce qui a faussé le p50 au boot)
    monkeypatch.setattr(main, "db", _StoreDB())
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


def test_final_scale_p50_below_threshold_warns(caplog, monkeypatch):
    """p50 < 20% -> warning explicite (diagnostic audit §2.1)."""
    import logging
    monkeypatch.setattr(main, "db", _StoreDB())
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


# ---------------------- pagination : > 300 barres réelles -------------------

class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
    def json(self):
        return self._payload


class _PagedHTTPX:
    """Simule l'API OKX paginée (2 pages de 300 barres, ordre DESC)."""
    def __init__(self):
        self.calls = 0

    def get(self, url, timeout=10.0):
        self.calls += 1
        page = self.calls
        rows = []
        base = 1_780_000_000_000
        for i in range(300):
            ts = base - (page - 1) * 300 * 3600_000 - i * 3600_000
            rows.append([str(ts), "100", "101", "99", "100.5", "10", "0", "0", "0"])
        return _FakeResp({"data": rows})


def test_okx_pagination_exceeds_300_bars(monkeypatch):
    """La pagination OKX doit fournir > 300 barres (fenêtres walk-forward)."""
    import backtester.live_candles as lc
    fake = _PagedHTTPX()
    monkeypatch.setattr(lc.httpx, "get", fake.get)
    df, src = lc.fetch_real_candles("BTCUSDT", limit=600, verbose=False)
    assert src == "okx"
    assert df is not None and len(df) == 600
    assert fake.calls >= 2, "la pagination doit faire plusieurs requêtes"
    assert df.index.is_monotonic_increasing
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


# ---------------------------- verdicts honnêtes -----------------------------

def test_honest_verdict_categories(capsys):
    from backtester.honest_verdict import print_honest_result
    assert print_honest_result(100000, 101500) == "profit"
    assert print_honest_result(100000, 100020) == "marginal"     # +0.02% : PAS « massif »
    assert print_honest_result(100000, 99000) == "loss"
    assert print_honest_result(100000, 100000) == "breakeven"
    out = capsys.readouterr().out
    assert "PERTE NETTE" in out


def test_cli_scripts_no_boastful_success_messages():
    """Les messages trompeurs de fin de backtest ont disparu des scripts."""
    root = Path(main.__file__).parent
    forbidden = ["massive net profits", "excellent compound profits",
                 "survived and achieved profits", "ZERO OVERFITTING",
                 "Zero overfitting detected"]
    for script in CLI_SCRIPTS:
        src = (root / script).read_text(encoding="utf-8")
        for phrase in forbidden:
            assert phrase.lower() not in src.lower(), \
                f"{script} : message trompeur restant -> {phrase}"


# -------------------- P0-6 : paper-trading daté et continu ------------------

class _PaperDB:
    def __init__(self):
        self.s = {}

    def save_setting(self, k, v, user_id=1, encrypt=False):
        self.s[k] = v

    def get_setting(self, k, user_id=1, decrypt=False):
        return self.s.get(k, "")


def _make_days(days):
    from datetime import datetime, timedelta
    base = datetime(2026, 8, 10)
    return [(base + timedelta(days=d)).strftime("%Y-%m-%d") for d in days]


def test_paper_validation_marks_only_real_runtime(monkeypatch):
    """Un jour n'est compté que si le bot tourne réellement ce jour-là."""
    pdb = _PaperDB()
    monkeypatch.setattr(main, "db", pdb)
    main._mark_paper_validation_day()
    stats = main._paper_validation_stats()
    assert stats["active_days"] == 1
    # le jour marqué est bien le jour UTC courant
    from datetime import datetime, timezone
    assert stats["days"][0] == datetime.now(timezone.utc).strftime("%Y-%m-%d")


def test_paper_validation_streak_and_validation(monkeypatch):
    """Série consécutive calculée ; validated uniquement si streak >= requis."""
    pdb = _PaperDB()
    monkeypatch.setattr(main, "db", pdb)
    pdb.s["paper_validation_days"] = __import__("json").dumps(_make_days([0, 1, 2, 3]))
    pdb.s["paper_validation_start_ts"] = "1000.0"
    stats = main._paper_validation_stats()
    assert stats["active_days"] == 4
    assert stats["latest_streak_days"] == 4
    # P0-6 : l'exigence par défaut est passée de 7 à 28 jours (audit : 4-8 semaines)
    assert stats["required_days"] == 28
    assert stats["validated"] is False


def test_paper_validation_interruption_breaks_streak(monkeypatch):
    """Un jour manquant (bot arrêté) casse la série continue."""
    pdb = _PaperDB()
    monkeypatch.setattr(main, "db", pdb)
    # jours 0,1,2 puis trou (3), puis 4,5,6,7
    pdb.s["paper_validation_days"] = __import__("json").dumps(_make_days([0, 1, 2, 4, 5, 6, 7]))
    stats = main._paper_validation_stats()
    assert stats["active_days"] == 7
    assert stats["latest_streak_days"] == 4   # seule la série la plus récente compte
    assert stats["validated"] is False        # pas de 7 jours CONTINUS


def test_api_v1_paper_validation_endpoint():
    client = TestClient(main.app)
    resp = client.get("/api/v1/paper-validation")
    assert resp.status_code == 200
    body = resp.json()
    assert "active_days" in body and "latest_streak_days" in body
    assert "required_days" in body and "validated" in body


# ----------------- facteur limitant (idée n°1 audit) + plancher ------------

def test_record_final_scale_tracks_limiting_factor(monkeypatch):
    """L'échantillon mémorise le facteur le PLUS réducteur (contrainte
    dominante) à partir des steps du pipeline."""
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(main.time, "time", lambda: clock["t"])
    steps = [
        {"step": "cvar_cap", "op": "min", "value": 100.0, "qty_after": 50.0},
        {"step": "max_asset_cap", "op": "min", "value": 100.0, "qty_after": 50.0},
        {"step": "conviction", "op": "mul", "value": 0.9, "qty_after": 45.0},
        {"step": "cash_reserve", "op": "mul", "value": 0.4, "qty_after": 18.0},
        {"step": "capacity", "op": "mul", "value": 0.7, "qty_after": 12.6},
        {"step": "tradability", "op": "mul", "value": 0.9, "qty_after": 11.34},
    ]
    main._record_final_scale("BTCUSDT", 0.2268, 6, steps=steps)
    s = main.STATE["final_scale_samples"][-1]
    assert s["limit_factor"] == "cash_reserve"   # 0.4 = le plus réducteur
    assert s["limit_value"] == pytest.approx(0.4)


def test_limiting_factor_stats_aggregation(monkeypatch):
    """Agrégation : le facteur le plus souvent limitant + sa valeur médiane."""
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(main.time, "time", lambda: clock["t"])
    for i in range(6):
        main.STATE.setdefault("final_scale_last_ts", {})
        main._record_final_scale(
            "BTCUSDT", 0.3, 17,
            steps=[{"step": "conviction", "op": "mul", "value": 0.9, "qty_after": 1.0},
                   {"step": "cash_reserve", "op": "mul", "value": 0.4, "qty_after": 1.0},
                   {"step": "capacity", "op": "mul", "value": 0.8, "qty_after": 1.0}])
        clock["t"] += 61.0
    for _ in range(4):
        main.STATE.setdefault("final_scale_last_ts", {})
        main._record_final_scale(
            "BTCUSDT", 0.5, 17,
            steps=[{"step": "conviction", "op": "mul", "value": 0.9, "qty_after": 1.0},
                   {"step": "cash_reserve", "op": "mul", "value": 0.7, "qty_after": 1.0},
                   {"step": "capacity", "op": "mul", "value": 0.5, "qty_after": 1.0}])
        clock["t"] += 61.0
    lim = main._limiting_factor_stats()
    assert lim["n"] == 10
    assert lim["top"][0]["factor"] == "cash_reserve"
    assert lim["top"][0]["count"] == 6
    assert lim["top"][0]["pct_of_samples"] == 60.0
    assert lim["top"][0]["median_value"] == pytest.approx(0.4)
    assert lim["top"][1]["factor"] == "capacity"


def test_vpin_bounded_01_on_real_bars():
    """FIX VPIN : sur des barres réelles (volumes 100-1000), le VPIN doit
    rester une probabilité bornée [0,1] — plus jamais 6 988 465."""
    import numpy as np
    import pandas as pd
    from models.microstructure_edge import MicrostructureEdgeEngine

    n = 120
    idx = pd.date_range("2026-08-01", periods=n, freq="h")
    close = np.cumsum(np.random.RandomState(3).normal(0, 50, n)) + 65000
    volume = np.random.RandomState(4).uniform(100, 1000, n)
    df = pd.DataFrame({"close": close, "volume": volume}, index=idx)

    eng = MicrostructureEdgeEngine()
    for nb in (10, 50, 100):
        vpin = eng.calculate_vpin(df, num_buckets=nb)
        assert 0.0 <= vpin <= 1.0, f"VPIN hors bornes avec {nb} buckets : {vpin}"


def test_vpin_neutral_on_empty_or_zero_volume():
    from models.microstructure_edge import MicrostructureEdgeEngine
    import pandas as pd
    eng = MicrostructureEdgeEngine()
    assert eng.calculate_vpin(pd.DataFrame()) == 0.5
    df = pd.DataFrame({"close": [1.0, 1.0], "volume": [0.0, 0.0]})
    assert eng.calculate_vpin(df) == 0.5


def test_allocate_modulate_ignores_out_of_bounds_vpin(monkeypatch):
    """FIX : un VPIN aberrant (hors [0,1]) ne doit PLUS réduire la conviction
    dans MetaAllocationEngine.allocate() (max(0.50, 1-(6.9M-0.90)) = 0.50
    appliqué à chaque tick = cause majeure des signaux faibles)."""
    from strategies.engine import MetaAllocationEngine, TrendFollowingStrategy

    eng = MetaAllocationEngine(strategies=[TrendFollowingStrategy()])
    md = {
        "df": None, "symbol": "BTCUSDT", "price_primary": 69000.0,
        "price_secondary": 69000.0, "bids": [], "asks": [],
        "inventory": 0.0, "max_inventory": 1.0,
        "vpin": 6_988_465.87,          # valeur aberrante observée en prod
        "kyle_lambda": 2.1e-07, "onchain_risk": 0.5, "sentiment": 0.0,
        "funding_rate_8h": 0.0, "market_avg_return": 0.0, "cross_asset_bias": 0.0,
    }
    # VPIN normal (0.95) DOIT réduire ; VPIN aberrant doit être NEUTRE (1.0)
    res_bad = eng.allocate(md, 1, 0.001, 0.0)
    assert res_bad["modulate_factor"] == 1.0, \
        f"VPIN aberrant modulé la conviction : {res_bad['modulate_factor']}"
    md2 = dict(md, vpin=0.95)
    res_high = eng.allocate(md2, 1, 0.001, 0.0)
    assert res_high["modulate_factor"] < 1.0, "VPIN 0.95 devrait réduire"


def test_final_scale_persists_every_5min(monkeypatch):
    """P0-4 : persistance toutes les ~5 min (perte max 5 min au lieu de 60)."""
    sdb = _StoreDB()
    monkeypatch.setattr(main, "db", sdb)
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(main.time, "time", lambda: clock["t"])
    main.STATE["final_scale_last_persist"] = 0.0
    main.STATE["final_scale_samples"] = []
    main._record_final_scale("BTCUSDT", 0.3, 17, steps=None)
    assert "final_scale_samples_json" in sdb.s, \
        "le premier échantillon doit persister (0 >= 300s)"
    # 4 minutes plus tard : pas encore de persist
    sdb.s.clear()
    clock["t"] += 240.0
    main._record_final_scale("BTCUSDT", 0.4, 17, steps=None)
    assert "final_scale_samples_json" not in sdb.s, \
        "persist trop fréquent (< 5 min)"
    # 6 minutes plus tard : persist
    clock["t"] += 120.0
    main._record_final_scale("BTCUSDT", 0.5, 17, steps=None)
    assert "final_scale_samples_json" in sdb.s, \
        "persist attendu après >= 5 min"


def test_floor_prevents_cumulative_self_strangulation():
    """P0-4 : le produit de 17 facteurs prudents est plafonné à 15 % — le
    correctif de la chaîne identifié par l'audit (0.8^15 ≈ 3,5 %)."""
    from core.risk_pipeline import apply_risk_pipeline, FINAL_SCALE_FLOOR
    res = apply_risk_pipeline(
        base_qty=1000.0, cvar_qty=1e9, max_asset_qty=1e9, conviction=0.8,
        risk_state_scale=0.8, news_scale=0.8, macro_scale=0.8, onchain_scale=0.8,
        corr_scale=0.8, confidence_scale=0.8, org_scale=0.8, rlhf_scale=0.8,
        vol_scale=0.8, tradability_scale=0.8, capacity_scale=0.8,
        cash_reserve_scale=0.8, order_flow_scale=0.8, regime_confidence_scale=0.8)
    # 16 facteurs à 0.8 => 0.8^16 ≈ 2.8% -> clampé au plancher 15%
    assert res["final_scale"] == pytest.approx(FINAL_SCALE_FLOOR)
    assert res["qty"] == pytest.approx(1000.0 * FINAL_SCALE_FLOOR)
    assert res["steps"][-1]["step"] == "cumulative_floor"
