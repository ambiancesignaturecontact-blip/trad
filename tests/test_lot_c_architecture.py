"""
LOT C (F3 — architecture) : vérifications structurelles du découpage.

  1. main.py < 4000 lignes (objectif LOT C) et aucun `from main import *`.
  2. Les gros blocs extraits vivent dans des modules dédiés et main les
     ré-exporte (l'espace de noms main.X reste stable pour les tests,
     TASK_FACTORIES, api/routes.py, schedulers.py, telemetry.py).
  3. Aucun nouveau `from main import *` nulle part dans le repo.
  4. Les fonctions extraites se comportent à l'identique (échantillons).
"""
import inspect
from pathlib import Path

import pytest

import main

ROOT = Path(main.__file__).parent


def test_main_under_4000_lines():
    """Objectif LOT C : main.py < 4000 lignes."""
    src = inspect.getsource(main)
    n_lines = src.count("\n") + 1
    assert n_lines < 4000, f"main.py = {n_lines} lignes (objectif < 4000)"


def _star_imports(py: Path) -> list:
    """Vrai `from main import *` (nœud AST ImportFrom), pas les commentaires."""
    import ast
    try:
        tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.ImportFrom) and node.module == "main"
                and any(a.name == "*" for a in node.names)):
            out.append(node.lineno)
    return out


def test_no_star_imports_from_main():
    """Aucun `from main import *` réel ne doit subsister (F3)."""
    targets = [
        "main.py",
        "api/routes.py",
        "schedulers.py",
        "telemetry.py",
        "core/observability.py",
        "core/ccxt_client.py",
        "core/autonomous_ai.py",
        "core/decision_explain.py",
        "market_data/historical_fetch.py",
    ]
    for path in targets:
        bad = _star_imports(ROOT / path)
        assert not bad, f"{path} contient encore un star import (lignes {bad})"


def test_no_star_imports_anywhere():
    """Le repo entier est exempt de `from main import *` (code réel)."""
    for py in ROOT.rglob("*.py"):
        if "__pycache__" in str(py) or "node_modules" in str(py):
            continue
        bad = _star_imports(py)
        if bad:
            pytest.fail(f"{py.relative_to(ROOT)} contient un 'from main import *' "
                        f"(lignes {bad})")


def test_extracted_modules_exist_and_reexports_present():
    """Les modules extraits existent et main ré-exporte les symboles."""
    for mod in ("core.observability", "core.ccxt_client",
                "core.autonomous_ai", "core.decision_explain",
                "market_data.historical_fetch"):
        __import__(mod)
    for name in ("_final_scale_report", "_record_final_scale",
                 "_paper_validation_stats", "_mark_paper_validation_day",
                 "fetch_historical_market_data", "fetch_yahoo_finance_candles",
                 "get_ccxt_client", "format_exchange_size",
                 "autonomous_ai_scheduler", "explain_last_decision",
                 "update_metrics_from_state"):
        assert hasattr(main, name), f"main.{name} manquant (ré-export cassé)"


def test_extracted_modules_import_explicitly_from_main():
    """Les modules extraits n'utilisent pas de star import et passent ruff."""
    import subprocess
    res = subprocess.run(
        ["ruff", "check",
         "core/observability.py", "core/ccxt_client.py",
         "core/autonomous_ai.py", "core/decision_explain.py",
         "market_data/historical_fetch.py", "api/routes.py",
         "schedulers.py", "telemetry.py"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert res.returncode == 0, res.stdout + res.stderr


def test_observability_functions_behavior_unchanged():
    """Comportement des fonctions extraites identique (échantillon réel)."""
    from core.observability import (
        _final_scale_stats,
        _purge_final_scale_samples,
        _record_final_scale,
    )
    t0 = 1_000_000.0
    main.STATE["final_scale_samples"] = []
    main.STATE["final_scale_last_ts"] = {}
    for i in range(10):
        main.STATE["final_scale_samples"].append(
            {"ts": t0 + i * 3600, "symbol": "BTCUSDT",
             "final_scale": 0.2 + 0.01 * i, "n_steps": 17})
    stats = _final_scale_stats()
    assert stats is not None and stats["n"] == 10
    assert stats["p50"] == pytest.approx(0.245, abs=1e-3)
    _purge_final_scale_samples(48 * 3600.0)
    _record_final_scale("BTCUSDT", 0.9, 17)  # ne doit pas lever

    # _signal_stats : distribution réelle des signaux
    from core.observability import _signal_stats
    main.STATE["recent_signals"] = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    ss = _signal_stats()
    assert ss["abs_p50"] == pytest.approx(0.25, abs=1e-3)


def test_klines_converter_unchanged():
    """_klines_to_df préserve les valeurs réelles (échantillon Binance)."""
    from market_data.historical_fetch import _klines_to_df
    bars = [[1700000000000, "100.0", "110.0", "90.0", "105.0", "1234.5"]]
    df = _klines_to_df(bars)
    assert df.iloc[0]["open"] == 100.0
    assert df.iloc[0]["high"] == 110.0
    assert df.iloc[0]["close"] == 105.0
    assert df.iloc[0]["volume"] == 1234.5


def test_scheduler_registered_in_task_factories():
    """Le scheduler IA autonome reste enregistré (watchdog)."""
    assert "autonomous_ai_scheduler" in main.TASK_FACTORIES
    assert callable(main.TASK_FACTORIES["autonomous_ai_scheduler"])


def test_telemetry_payload_unchanged():
    """Le payload de télémétrie contient toujours les sections extraites."""
    import core.observability  # noqa: F401  (patch appliqué par les tests)
    from telemetry import compile_telemetry_data
    tel = compile_telemetry_data()
    assert "regime_autonomy" in tel          # LOT B
    assert "active_factors_last" in tel      # LOT A
    assert "paper_validation" in tel         # P0-6
    assert "final_scale_stats" in tel        # P0-4
