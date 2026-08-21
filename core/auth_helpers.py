"""
HELPERS D'AUTHENTIFICATION (LOT 8 — architecture, découplage).
Extraits de main.py (AST, corps inchangés) ; symboles partagés importés de
main de façon EXPLICITE (main est complet quand ce module est importé, en fin
de main.py).
"""
import logging
import os
from pathlib import Path

import httpx
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

from database.auth import AuthManager, Roles
from main import STATE, db  # noqa: E402

logger = logging.getLogger("InstitutionalTradingBot")

auth_security_optional = HTTPBearer(auto_error=False)


def _is_remote_deployment() -> bool:
    """
    P0-2 (audit indépendant §2.9): détecte un déploiement non-local.
    Considéré non-local si une variable RAILWAY_* est présente (Railway) ou si
    PORT est défini dans l'environnement (convention PaaS/Railway) — une URL
    publique est alors exposée et l'authentification devient OBLIGATOIRE.
    """
    if any(k.startswith("RAILWAY_") for k in os.environ):
        return True
    port = os.getenv("PORT", "").strip()
    return port.isdigit() and int(port) > 0


def auth_enforced() -> bool:
    """
    L'authentification est-elle exigée sur les routes d'action ?
     - AUTH_ENABLED=true explicite -> OUI (priorité absolue)
     - AUTH_ENABLED=false explicite -> NON (choix assumé de l'opérateur,
       ex. sandbox de dev/preview ; prioritaire sur la détection d'environnement)
     - sinon : mode REAL (argent réel : jamais contrôlable sans session) OU
       déploiement non-local (PORT/RAILWAY_* -> URL publique, P0-2) -> OUI.
    """
    explicit = os.getenv("AUTH_ENABLED", "").strip().lower()
    if explicit in ("1", "true", "yes", "on"):
        return True
    if STATE["mode"] == "REAL":
        return True
    if explicit in ("0", "false", "no", "off"):
        return False
    return _is_remote_deployment()


def require_auth(credentials=Depends(auth_security_optional)):
    """
    Protects state-changing endpoints.
    Enforced when AUTH_ENABLED=true, OR in REAL mode, OR on any non-local
    deployment (P0-2: PORT/RAILWAY_* -> an exposed URL must never be open
    without authentication, even in DEMO mode).
    """
    if not auth_enforced():
        return {"role": Roles.ADMIN, "username": "local-demo", "sub": "1"}
    if credentials is None or not getattr(credentials, "credentials", None):
        raise HTTPException(status_code=401, detail="Authentication required")
    return AuthManager.verify_jwt_token(credentials.credentials)


def require_admin(credentials=Depends(auth_security_optional)):
    """ADMIN-gated dependency for user management endpoints (audit C7).
    STRICT: always validates a real JWT - never the DEMO local-bypass, because
    user management must be protected even when AUTH_ENABLED is off."""
    if credentials is None or not getattr(credentials, "credentials", None):
        raise HTTPException(status_code=401, detail="Authentication required.")
    user = AuthManager.verify_jwt_token(credentials.credentials)
    if user.get("role") != Roles.ADMIN:
        raise HTTPException(status_code=403, detail="Admin role required.")
    return user


def _telegram_send_sync(token: str, chat_id: str, text: str) -> bool:
    """Envoi Telegram best-effort synchrone (canal dédié pour un secret au boot).
    Ne lève jamais : retourne False en cas d'échec."""
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text},
            )
            return resp.status_code == 200
    except Exception:
        return False


def _deliver_admin_password_once(admin_pass: str, creds_path: str = None) -> str:
    """
    P0-3 (audit indépendant §2.10): livre le mot de passe admin auto-généré par
    un canal dédié — JAMAIS dans les logs applicatifs (les logs sont souvent
    centralisés/exportés vers des tiers).
    Ordre : 1) Telegram DM si TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID configurés,
            2) fichier .admin_credentials en mode 0600, 3) aucun canal.
    Retourne "telegram" | "file" | "none".
    """
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if tg_token and tg_chat:
        try:
            ok = _telegram_send_sync(
                tg_token,
                tg_chat,
                "🔐 QUANT-PORTAL : mot de passe admin auto-généré "
                "(secret — ne pas partager, ne pas logger).\n"
                "username=admin_quant\n"
                f"password={admin_pass}\n"
                "Définissez ADMIN_PASSWORD (env) puis redémarrez pour le remplacer.",
            )
            if ok:
                return "telegram"
        except Exception:
            pass
    path = Path(creds_path) if creds_path else Path(os.path.dirname(os.path.abspath(__file__))) / ".admin_credentials"
    try:
        path.write_text(
            "# QUANT-PORTAL — mot de passe admin auto-généré (SECRET).\n"
            "# Ne jamais committer, logger ni partager. Supprimez ce fichier après\n"
            "# la première connexion et définissez ADMIN_PASSWORD (env) à la place.\n"
            f"username=admin_quant\npassword={admin_pass}\n",
            encoding="utf-8",
        )
        os.chmod(path, 0o600)
        return "file"
    except Exception:
        return "none"


def _ensure_auth_secrets(creds_path: str = None) -> None:
    """
    Audit B3-3/B3-4 + P0-2/P0-3 : secrets forts auto-générés au premier boot,
    persistés chiffrés en DB, jamais loggués en clair (admin livré une fois).
    """
    import secrets as _secrets

    import bcrypt as _bcrypt

    # ---- JWT secret: reuse persisted or generate strong ----
    jwt_secret = os.getenv("JWT_SECRET_KEY", "")
    if len(jwt_secret) < 24:
        persisted = db.get_setting("jwt_secret_key", decrypt=True)
        if persisted and len(persisted) >= 24:
            jwt_secret = persisted
        else:
            jwt_secret = _secrets.token_urlsafe(48)
            db.save_setting("jwt_secret_key", jwt_secret, encrypt=True)
            logger.warning(
                "🔐 AUTH: auto-generated a strong JWT_SECRET_KEY and stored it "
                "encrypted in the DB. Set JWT_SECRET_KEY env to override."
            )
        os.environ["JWT_SECRET_KEY"] = jwt_secret

    # ---- Admin password: reuse persisted hash or generate + upsert ----
    admin_pass = os.getenv("ADMIN_PASSWORD", "")
    if not admin_pass or admin_pass == "ChangeMe!Institutionnel2026":
        persisted_hash = db.get_user("admin_quant")
        if persisted_hash and persisted_hash.get("password_hash", "").startswith("$2"):
            # a real bcrypt hash already exists in the DB -> rely on it
            logger.warning(
                "🔐 AUTH: ADMIN_PASSWORD env not set - using the bcrypt hash "
                "already stored in the users table. Set ADMIN_PASSWORD to override."
            )
        else:
            admin_pass = _secrets.token_urlsafe(12)
            hashed = _bcrypt.hashpw(admin_pass.encode(), _bcrypt.gensalt()).decode()
            db.upsert_admin(hashed, Roles.ADMIN)
            os.environ["ADMIN_PASSWORD"] = admin_pass
            # P0-3: le secret n'apparaît JAMAIS dans les logs — canal dédié.
            delivered = _deliver_admin_password_once(admin_pass, creds_path)
            if delivered == "telegram":
                logger.warning(
                    "🔐 AUTH: auto-generated admin password — livré par Telegram "
                    "(jamais loggé). Définissez ADMIN_PASSWORD env pour le remplacer."
                )
            elif delivered == "file":
                logger.warning(
                    "🔐 AUTH: auto-generated admin password — écrit dans "
                    ".admin_credentials (0600, jamais loggé). Supprimez ce fichier "
                    "après la première connexion et définissez ADMIN_PASSWORD env."
                )
            else:
                logger.warning(
                    "🔐 AUTH: auto-generated admin password (jamais loggé, aucun "
                    "canal de livraison configuré). Définissez ADMIN_PASSWORD env "
                    "pour garder la main sur l'accès."
                )
    else:
        # env password provided: make sure the DB hash is in sync
        try:
            hashed = _bcrypt.hashpw(admin_pass.encode(), _bcrypt.gensalt()).decode()
            db.upsert_admin(hashed, Roles.ADMIN)
        except Exception:
            pass
