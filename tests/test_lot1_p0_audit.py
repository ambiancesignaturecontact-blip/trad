"""
LOT 1 — P0 audit indépendant (items 1, 2, 3).

P0-1 : aucune fausse allégation « MevShield On-Chain active » dans le code
       Python (le contrat Solidity n'est ni compilé, ni déployé, ni appelé).
P0-2 : AUTH forcée automatiquement sur tout déploiement non-local
       (PORT / RAILWAY_*), sans casser le bypass local DEMO.
P0-3 : le mot de passe admin auto-généré n'apparaît JAMAIS dans les logs ;
       il est livré par canal dédié (Telegram DM sinon fichier 0600).

Preuve par le comportement, pas par la déclaration.
"""
import logging
import os
import stat

import pytest
from fastapi import HTTPException
from pathlib import Path

import main

# ---------------------------------------------------------------- P0-1 ----

def test_no_fake_mevshield_claim_in_source():
    """Aucun fichier Python de PRODUCTION ne mentionne plus MevShield (l'ancienne
    fausse allégation active envoyée à l'utilisateur)."""
    root = Path(main.__file__).parent
    hits = []
    for py in root.rglob("*.py"):
        if "venv" in py.parts or ".git" in py.parts or "node_modules" in py.parts:
            continue
        if "tests" in py.parts:  # le présent test nomme le mot par nécessité
            continue
        if "MevShield" in py.read_text(encoding="utf-8", errors="ignore"):
            hits.append(str(py))
    assert hits == [], f"occurrence(s) MevShield restante(s) dans la prod : {hits}"


def test_no_fake_captured_claim_in_source():
    """Le message de notification signal-only ne prétend plus qu'un arbitrage
    a été « CAPTURÉ » alors qu'il n'est pas exécuté."""
    src = Path(main.__file__).read_text(encoding="utf-8")
    assert "ARBITRAGE DEX-CEX CAPTURÉ" not in src
    # le nouveau message doit être explicite sur la non-exécution
    assert "NON EXÉCUTÉ (signal-only)" in src


# ---------------------------------------------------------------- P0-2 ----

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Environnement neutre pour chaque test d'auth."""
    for k in list(os.environ):
        if k.startswith("RAILWAY_"):
            monkeypatch.delenv(k, raising=False)
    for k in ("AUTH_ENABLED", "ADMIN_PASSWORD", "JWT_SECRET_KEY", "PORT",
              "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(k, raising=False)
    yield


def test_remote_detection_port(monkeypatch):
    monkeypatch.setenv("PORT", "8080")
    assert main._is_remote_deployment() is True


def test_remote_detection_railway(monkeypatch):
    monkeypatch.setenv("RAILWAY_SERVICE_NAME", "quant-portal")
    assert main._is_remote_deployment() is True


def test_local_env_not_remote():
    assert main._is_remote_deployment() is False


def test_remote_deployment_forces_auth_in_require_auth(monkeypatch):
    """P0-2 : URL publique (PORT défini) -> un appel non authentifié est refusé
    même en mode DEMO."""
    monkeypatch.setenv("PORT", "8080")
    old_mode = main.STATE["mode"]
    main.STATE["mode"] = "DEMO"
    try:
        with pytest.raises(HTTPException) as exc:
            main.require_auth(None)
        assert exc.value.status_code == 401
    finally:
        main.STATE["mode"] = old_mode


def test_local_demo_bypass_preserved():
    """Le comportement local DEMO (sans AUTH_ENABLED, sans PORT/RAILWAY) reste
    inchangé : bypass ADMIN pour la démo locale."""
    old_mode = main.STATE["mode"]
    main.STATE["mode"] = "DEMO"
    try:
        res = main.require_auth(None)
        assert res.get("role") == main.Roles.ADMIN
        assert res.get("username") == "local-demo"
    finally:
        main.STATE["mode"] = old_mode


def test_auth_enabled_true_still_enforced(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    old_mode = main.STATE["mode"]
    main.STATE["mode"] = "DEMO"
    try:
        with pytest.raises(HTTPException) as exc:
            main.require_auth(None)
        assert exc.value.status_code == 401
    finally:
        main.STATE["mode"] = old_mode


def test_real_mode_always_enforced():
    """Le mode REAL exige l'auth même en local (règle institutionnelle
    existante, préservée)."""
    old_mode = main.STATE["mode"]
    main.STATE["mode"] = "REAL"
    try:
        with pytest.raises(HTTPException) as exc:
            main.require_auth(None)
        assert exc.value.status_code == 401
    finally:
        main.STATE["mode"] = old_mode


# ---------------------------------------------------------------- P0-3 ----

class _FakeDB:
    """Mini-DB factice : aucun hash existant -> force la génération d'un
    nouveau mot de passe admin (chemin P0-3 à tester)."""
    def get_setting(self, key, user_id=1, decrypt=False):
        return None

    def save_setting(self, key, value, user_id=1, encrypt=False):
        return True

    def get_user(self, username):
        return None

    def upsert_admin(self, password_hash, role="ADMIN"):
        return True


@pytest.fixture
def fake_db(monkeypatch):
    db = _FakeDB()
    monkeypatch.setattr(main, "db", db)
    return db


def _run_ensure(creds_path):
    """Exécute la génération de secrets et retourne (admin_pass, caplog_text)."""
    old_adm = os.environ.get("ADMIN_PASSWORD")
    old_jwt = os.environ.get("JWT_SECRET_KEY")
    try:
        main._ensure_auth_secrets(creds_path=creds_path)
        return os.environ.get("ADMIN_PASSWORD", "")
    finally:
        if old_adm is None:
            os.environ.pop("ADMIN_PASSWORD", None)
        else:
            os.environ["ADMIN_PASSWORD"] = old_adm
        if old_jwt is None:
            os.environ.pop("JWT_SECRET_KEY", None)
        else:
            os.environ["JWT_SECRET_KEY"] = old_jwt


def test_admin_password_never_logged(fake_db, tmp_path, caplog):
    """P0-3 : le mot de passe auto-généré n'apparaît dans AUCUN log ;
    il est écrit dans le fichier .admin_credentials (0600)."""
    creds_path = tmp_path / ".admin_credentials"
    caplog.set_level(logging.WARNING, logger="InstitutionalTradingBot")
    gen = _run_ensure(str(creds_path))
    assert gen and len(gen) >= 10, "un mot de passe doit avoir été généré"
    assert gen not in caplog.text, "SECRET LOGGÉ EN CLAIR !"
    assert creds_path.exists(), "le fichier de livraison doit exister"
    content = creds_path.read_text(encoding="utf-8")
    assert f"password={gen}" in content
    mode = stat.S_IMODE(os.stat(creds_path).st_mode)
    assert mode == 0o600, f"permissions attendues 0600, obtenues {oct(mode)}"


def test_admin_password_delivered_via_telegram(fake_db, tmp_path, caplog, monkeypatch):
    """P0-3 : si Telegram est configuré, le secret part en DM (jamais dans les
    logs) et aucun fichier n'est créé."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    sent = {}

    def fake_send(token, chat_id, text):
        sent["token"], sent["chat_id"], sent["text"] = token, chat_id, text
        return True

    monkeypatch.setattr(main, "_telegram_send_sync", fake_send)
    creds_path = tmp_path / ".admin_credentials"
    caplog.set_level(logging.WARNING, logger="InstitutionalTradingBot")
    gen = _run_ensure(str(creds_path))
    assert sent.get("text") and f"password={gen}" in sent["text"]
    assert gen not in caplog.text, "SECRET LOGGÉ EN CLAIR !"
    assert not creds_path.exists(), "aucun fichier requis si Telegram a fonctionné"


def test_admin_password_fallback_file_when_telegram_fails(fake_db, tmp_path, caplog, monkeypatch):
    """P0-3 : si Telegram échoue, repli sur le fichier 0600 — toujours aucun
    secret dans les logs."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setattr(main, "_telegram_send_sync", lambda *a, **k: False)
    creds_path = tmp_path / ".admin_credentials"
    caplog.set_level(logging.WARNING, logger="InstitutionalTradingBot")
    gen = _run_ensure(str(creds_path))
    assert gen not in caplog.text, "SECRET LOGGÉ EN CLAIR !"
    assert creds_path.exists()
    assert f"password={gen}" in creds_path.read_text(encoding="utf-8")


def test_existing_admin_password_not_generated(fake_db, monkeypatch, tmp_path):
    """Un ADMIN_PASSWORD fourni par env n'est jamais régénéré ni loggé."""
    monkeypatch.setenv("ADMIN_PASSWORD", "MyEnvPass123!")
    # pas de hash en DB -> avec env fourni, on ne génère rien de nouveau
    _run_ensure(str(tmp_path / "creds_test"))
    assert os.environ.get("ADMIN_PASSWORD") == "MyEnvPass123!"
