"""
LOT 4 — P1-9 (audit §2.8/§4.10) + P1-15 (audit §5) : CI et couverture.

P1-9 : le CI doit installer requirements.lock (figé, = celui de la prod
Docker) + requirements-dev.txt, et pip-audit doit faire ÉCHOUER le build sur
une CVE haute/critique (fini le `|| true`).
P1-15 : seuil de couverture minimal PAR FICHIER sur le cœur qui touche
l'argent (main.py, strategies/engine.py, core/risk_pipeline.py).

Vérifications statiques (le CI réel s'exécute sur GitHub Actions) + preuve
pip-audit exécutée localement dans le rapport de lot.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _ci_yaml() -> str:
    return (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def _lock_lines() -> list:
    lines = []
    for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            lines.append(s)
    return lines


# ----------------------------------------------------------------- P1-9 ----

def test_ci_installs_locked_requirements():
    """Le CI installe requirements.lock (+ dev), PAS requirements.txt."""
    ci = _ci_yaml()
    assert "pip install -r requirements.lock" in ci
    assert "pip install -r requirements-dev.txt" in ci
    assert "pip install -r requirements.txt" not in ci, \
        "requirements.txt (>= non figé) ne doit plus être installé en CI"


def test_ci_python_version_matches_dockerfile():
    """CI et prod Docker sur le même Python (3.11) — sinon les tests ne
    prouvent rien pour la prod."""
    ci = _ci_yaml()
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'python-version: "3.11"' in ci
    assert "python:3.11-slim" in dockerfile


def test_ci_pip_audit_is_blocking_on_high_critical():
    """Le CI bloque le build sur CVE haute/critique (fini le || true)."""
    ci = _ci_yaml()
    assert "scripts/check_vulnerabilities.py" in ci, \
        "le CI doit appeler le script bloquant de vulnérabilités"
    for line in ci.splitlines():
        if "pip-audit" in line:
            assert "|| true" not in line, \
                f"pip-audit ne doit plus être non bloquant : {line}"
    # aucun step du CI ne doit être non bloquant (ruff est passé au crible aussi)
    assert "|| true" not in ci
    # ruff est BLOQUANT (nettoyage fait : ruff check . passe sur tout le repo)
    assert "ruff check ." in ci


def test_ci_has_coverage_step():
    """Le CI exécute le script de seuils de couverture (P1-15)."""
    ci = _ci_yaml()
    assert "scripts/check_coverage.py" in ci


def test_requirements_lock_is_fully_pinned():
    """Toutes les dépendances du lock sont épinglées (==), aucune >= ."""
    for line in _lock_lines():
        assert "==" in line, f"dépendance non épinglée dans le lock : {line}"
        assert ">=" not in line, f"contrainte >= interdite dans le lock : {line}"
    names = [re.split(r"[=<>]", line)[0] for line in _lock_lines()]
    assert "fastapi" in names and "uvicorn" in names and "ccxt" in names
    assert "numpy" in names and "pandas" in names and "web3" in names


def test_requirements_dev_has_ci_tools():
    dev = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    for tool in ("pytest==", "pytest-asyncio==", "pytest-cov==",
                 "pip-audit==", "ruff=="):
        assert tool in dev, f"{tool} manquant dans requirements-dev.txt"


def test_lock_has_no_duplicate_packages():
    names = [re.split(r"[=<>]", line)[0] for line in _lock_lines()]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"paquets dupliqués dans le lock : {dupes}"


# ---------------------------------------------------------------- P1-15 ----

def test_check_coverage_script_thresholds():
    src = (ROOT / "scripts" / "check_coverage.py").read_text(encoding="utf-8")
    assert '"main.py": 30.0' in src
    assert '"strategies/engine.py": 80.0' in src
    assert '"core/risk_pipeline.py": 95.0' in src
    # codes de sortie distincts : 0 OK / 1 sous seuil / 2 mesure impossible
    assert "return 0 if ok else 1" in src
    assert "return 2" in src


def test_check_coverage_script_help_runs():
    """Le script démarre (argparse) sans lancer la suite complète."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_coverage.py"), "--help"],
        capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0
    assert "usage:" in proc.stdout
    assert "base-dir" in proc.stdout


# ------------------------------------------- script vulnérabilités (P1-9) --

def test_vuln_script_severity_parsing():
    """Parsing des sévérités pip-audit (str ET dict cvssV3)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cv", ROOT / "scripts" / "check_vulnerabilities.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._severity_of({"severity": "high"}) == "high"
    assert mod._severity_of({"severity": "CRITICAL"}) == "critical"
    assert mod._severity_of({"severity": "medium"}) == "medium"
    assert mod._severity_of({"severity": ""}) == "unknown"
    assert mod._severity_of({"severity": {"source": "GHSA",
                                          "cvssV3": {"baseSeverity": "HIGH"}}}) == "high"
    assert mod._severity_of({}) == "unknown"


def test_vuln_script_blocking_logic(monkeypatch, capsys):
    """Une CVE haute/critique -> exit 1 ; sinon exit 0."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cv", ROOT / "scripts" / "check_vulnerabilities.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(sys, "argv",
                        ["check_vulnerabilities.py", "--lock", str(ROOT / "requirements.lock")])

    class FakeProc:
        returncode = 1
        stdout = '{"dependencies": [{"name": "fake", "version": "1.0", "vulns": [' \
                 '{"id": "CVE-2026-0001", "severity": "critical"}]}]}'
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: FakeProc())
    assert mod.main() == 1
    out = capsys.readouterr().out
    assert "BUILD BLOQUÉ" in out and "CVE-2026-0001" in out

    # version sans vulnérabilité -> 0
    class FakeProc2:
        returncode = 0
        stdout = '{"dependencies": [{"name": "fake", "version": "1.0", "vulns": []}]}'
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: FakeProc2())
    assert mod.main() == 0


def test_vuln_script_help_runs():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_vulnerabilities.py"), "--help"],
        capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0
    assert "usage:" in proc.stdout


def test_ruff_passes_on_whole_repo():
    """Garde-fou local : ruff check . doit passer (le CI est bloquant)."""
    proc = subprocess.run(["ruff", "check", "."], capture_output=True,
                          text=True, timeout=120, cwd=str(ROOT))
    assert proc.returncode == 0, f"ruff échoue:\n{proc.stdout[-1500:]}"


def test_ruff_config_has_no_deprecated_rule():
    """W503 (retiré de ruff) ne doit pas revenir dans la config."""
    cfg = (ROOT / ".ruff.toml").read_text(encoding="utf-8")
    assert "W503" not in cfg
