"""
LOT 8 — P2-16 / P2-17 / P2-18 (audit indépendant §5-16/17/18).

P2-16 : l'ambiguïté market_data/macro_calendar.py vs models/macro_calendar.py
        est résolue : market_data/macro_calendar.py est un shim de dépréciation
        (pattern ai/) vers l'implémentation RÉELLE (données datées, plus de
        FOMC simulé « dans 4h » en boucle).
P2-17 : models/telegram_bot.py a déménagé dans bot/telegram_bot.py (ce n'est
        pas un modèle quantitatif).
P2-18 : le shim db_manager.py racine est supprimé ; les 5 importeurs utilisent
        database.db_manager.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- P2-16 ----

def test_market_data_macro_calendar_is_deprecation_shim():
    """market_data/macro_calendar.py ne contient PLUS la version simulée :
    c'est un shim vers models.macro_calendar (l'implémentation réelle)."""
    src = (ROOT / "market_data" / "macro_calendar.py").read_text(encoding="utf-8")
    assert "from models.macro_calendar import" in src
    # l'ancien code simulé (FOMC « dans 4h » en boucle) a disparu
    assert "time.time() + 3600 * 4" not in src
    assert "simulating real calendar timelines" not in src


def test_macro_shim_exposes_real_engine():
    """Importer depuis market_data.macro_calendar donne le moteur RÉEL."""
    from market_data.macro_calendar import MacroeconomicCalendarEngine, IMPACT_REDUCTION
    assert MacroeconomicCalendarEngine.__module__ == "models.macro_calendar"
    assert IMPACT_REDUCTION["HIGH"] == 0.40


def test_no_old_simulated_events_anywhere():
    """Aucun événement macro simulé (time.time() + 3600*X) ne subsiste."""
    import re
    for rel in ("market_data/macro_calendar.py", "models/macro_calendar.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "time.time() + 3600" not in src, f"{rel} contient encore du simulé"


# ---------------------------------------------------------------- P2-17 ----

def test_telegram_bot_moved_out_of_models():
    """bot/telegram_bot.py existe ; models/telegram_bot.py a disparu."""
    assert (ROOT / "bot" / "telegram_bot.py").exists()
    assert not (ROOT / "models" / "telegram_bot.py").exists()


def test_telegram_bot_import_via_bot():
    """L'import se fait via bot.telegram_bot et fonctionne."""
    from bot.telegram_bot import TelegramBotManager
    assert TelegramBotManager.__module__ == "bot.telegram_bot"
    # main importe la nouvelle localisation
    main_src = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "from bot.telegram_bot import TelegramBotManager" in main_src
    assert "from models.telegram_bot import" not in main_src


def test_telegram_bot_commands_preserved():
    """Les commandes Telegram sont préservées après le déplacement."""
    src = (ROOT / "bot" / "telegram_bot.py").read_text(encoding="utf-8")
    for cmd in ('"/status"', '"/history"', '"/modes"', '"/risk"',
                '"/pause"', '"/resume"', '"/kill"', '"/honesty"'):
        assert cmd in src, f"commande {cmd} manquante après déplacement"


# ---------------------------------------------------------------- P2-18 ----

def test_root_db_manager_shim_removed():
    """Le shim db_manager.py racine n'existe plus."""
    assert not (ROOT / "db_manager.py").exists()


def test_all_importers_use_database_db_manager():
    """Les 5 importeurs utilisent database.db_manager (plus le shim racine)."""
    importers = ["main.py", "run_walk_forward.py",
                 "tests/oms/test_capital_persistence.py",
                 "tests/oms/test_oms.py",
                 "tests/reconciliation/test_reconciliation.py"]
    for rel in importers:
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "from database.db_manager import DBManager" in src, \
            f"{rel} n'importe pas database.db_manager"
        assert "from db_manager import" not in src


def test_database_db_manager_importable():
    """database.db_manager fonctionne (utilisé par main au boot)."""
    from database.db_manager import DBManager
    db = DBManager()
    assert db is not None
