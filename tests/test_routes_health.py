"""
Régression prod (logs Railway) : aucune route ne doit lever NameError
(le bug 'compute_health_score is not defined' rendait /api/v1/health 500)
et le supervisor ne doit pas crier quand le bot est volontairement en pause.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from core.robustness import Supervisor
from test_support import all_api_paths

ROOT = Path(__file__).resolve().parent.parent


def test_health_endpoint_responds():
    """GET /api/v1/health ne doit PAS lever NameError (bug logs Railway)."""
    with TestClient(main.app) as c:
        r = c.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert "health_score" in body
        assert 0 <= body["health_score"] <= 100


def test_all_get_routes_respond_without_nameerror():
    """Aucune route GET ne doit lever NameError (le 'from main import *' a
    perdu des symboles au nettoyage ruff F401)."""
    with TestClient(main.app) as c:
        for path in all_api_paths(main.app):
            if "{" in path:  # routes paramétrées (ex. /packs/{name})
                continue
            try:
                r = c.get(path)
            except Exception as e:
                pytest.fail(f"GET {path} a levé {type(e).__name__}: {e}")
            # les 4xx (auth/validation) sont acceptables, jamais une erreur 5xx
            assert r.status_code < 500, f"GET {path} -> {r.status_code}"


def test_supervisor_silent_when_bot_paused():
    """Bot en pause volontaire (is_running=False) : PAS de 'heartbeat stale'
    ni 'order flow silent' — c'était le spam des logs Railway."""
    state = {"is_running": False, "last_tick_ts": 0.0, "last_price": 100.0,
             "data_quality_status": "LIVE", "order_flow": {}}
    sup = Supervisor(state)
    issues = sup.check(now=1_000_000.0, force=True)
    assert "trading loop heartbeat stale" not in issues
    assert "order flow silent" not in issues


def test_supervisor_alerts_when_running_but_stale():
    """Bot censé tourner mais heartbeat vieux : l'alerte reste active."""
    state = {"is_running": True, "last_tick_ts": 0.0, "last_price": 100.0,
             "data_quality_status": "LIVE", "order_flow": {}}
    sup = Supervisor(state)
    issues = sup.check(now=1_000_000.0, force=True)
    assert "trading loop heartbeat stale" in issues


def test_extracted_modules_have_no_missing_imports():
    """Les modules extraits (routes/schedulers/telemetry) n'utilisent que des
    symboles résolus (imports directs) — vérification statique par AST."""
    import ast
    import builtins

    main_ns = set(dir(main))
    for path in ("api/routes.py", "schedulers.py", "telemetry.py"):
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        module_defs = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                module_defs.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    module_defs.add(a.asname or a.name.split(".")[0])
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                    if isinstance(t, ast.Name):
                        module_defs.add(t.id)
        missing = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                local = set()
                for sub in ast.walk(node):
                    if isinstance(sub, ast.arg):
                        local.add(sub.arg)
                    elif isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store,)):
                        local.add(sub.id)
                    elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        local.add(sub.name)
                    elif isinstance(sub, ast.ExceptHandler) and sub.name:
                        local.add(sub.name)
                    # imports locaux dans la fonction (ex: import bcrypt as _bc)
                    elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                        for a in sub.names:
                            local.add(a.asname or a.name.split(".")[0])
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                        n = sub.id
                        if n in local or n in module_defs or n in dir(builtins) or n in main_ns:
                            continue
                        missing.add(n)
        assert not missing, f"{path} : symboles non résolus -> {sorted(missing)}"
