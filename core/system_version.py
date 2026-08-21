"""
SYSTEM VERSION & GOUVERNANCE (LOT 9 du mandat, §20).

Objectif : « Une décision doit permettre de savoir exactement avec quelle
version du système elle a été prise. »

Ce module construit et expose un SNAPSHOT DE VERSION du système :
  - config_hash  : SHA-256 du fichier config.yaml (les paramètres qui
                   pilotent le risque/la conviction/l'allocation)
  - git_commit   : dernier commit git (best-effort, None si indisponible)
  - system_version : identifiant composite `qp-{commit}-{hash[:8]}`
  - composants   : versions logicielles clés (python, fastapi) + registre
                   des modules (module_honesty) + modèle déployé MLOps

Chaque décision du decision_journal enregistre system_version + config_hash :
on peut rejouer « avec quelle version cette décision a été prise » (mandat).

Principes :
  1. JAMAIS bloquant (git absent, config illisible -> valeurs None/fallback).
  2. Le hash de config est calculé au démarrage (fichier statique) — les
     changements de config.yaml produisent un hash différent.
  3. Le registre des modèles déployés est lu depuis la DB (settings
     model_status_*_DEPLOYED) — honnête : ce qui est réellement déployé.
"""
import hashlib
import logging
import os
import subprocess

logger = logging.getLogger("InstitutionalTradingBot")

CONFIG_PATH = os.getenv("CONFIG_PATH", os.path.join(os.getcwd(), "config.yaml"))


def config_hash() -> str:
    """SHA-256 du contenu de config.yaml (None si illisible)."""
    try:
        if not os.path.exists(CONFIG_PATH):
            return None
        with open(CONFIG_PATH, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        logger.warning(f"config_hash failed ({e})")
        return None


def git_commit() -> str | None:
    """Dernier commit git (best-effort)."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
            cwd=os.path.dirname(os.path.abspath(__file__)))
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def system_version() -> str:
    """Identifiant composite : qp-{commit}-{config_hash[:8]}."""
    c = config_hash() or "noconfig"
    g = git_commit() or "nocommit"
    return f"qp-{g}-{c[:8]}"


def deployed_models(db) -> dict:
    """Modèles DÉPLOYÉS depuis le registre MLOps (settings model_status_*).
    Retourne {symbol: {model_type: version_id}} pour les status DEPLOYED."""
    out: dict = {}
    try:
        # lister les settings de la DB
        conn = db.get_connection()
        try:
            cur = conn.cursor()
            if db.is_postgres:
                cur.execute("SELECT key, value FROM system_settings WHERE key LIKE 'model_status_%'")
            else:
                cur.execute("SELECT key, value FROM system_settings WHERE key LIKE 'model_status_%'")
            rows = cur.fetchall()
        finally:
            conn.close()
        for r in rows:
            key = str(r[0])
            status = str(r[1])
            # model_status_{symbol}_{model_type}_{version_id}
            parts = key.split("_")
            if len(parts) >= 4 and status.upper() == "DEPLOYED":
                symbol = parts[2]
                model_type = parts[3]
                version_id = "_".join(parts[4:])
                out.setdefault(symbol, {})[model_type] = version_id
    except Exception as e:
        logger.debug(f"deployed_models failed ({e})")
    return out


def component_versions() -> dict:
    """Versions logicielles clés (best-effort)."""
    out = {}
    try:
        import sys
        out["python"] = sys.version.split()[0]
    except Exception:
        pass
    try:
        import fastapi
        out["fastapi"] = fastapi.__version__
    except Exception:
        pass
    try:
        import numpy as np
        out["numpy"] = np.__version__
    except Exception:
        pass
    return out


def build_system_snapshot(db=None) -> dict:
    """Snapshot complet de la version du système (télémétrie / API / journal)."""
    ch = config_hash()
    return {
        "system_version": system_version(),
        "config_hash": ch,
        "config_hash_short": (ch[:12] + "...") if ch else None,
        "git_commit": git_commit(),
        "components": component_versions(),
        "deployed_models": deployed_models(db) if db is not None else {},
        "note": "Chaque décision du journal enregistre system_version + config_hash : "
                "on sait exactement avec quelle version elle a été prise.",
    }


def decision_version() -> dict:
    """{system_version, config_hash} à injecter dans chaque décision."""
    return {
        "system_version": system_version(),
        "config_hash": config_hash(),
    }
