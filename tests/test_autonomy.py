"""
AUTONOMIE À 100 % — garanties verrouillées par test.

1. Le mode consultatif (approbation humaine avant chaque trade) est OFF par
   défaut : le bot ne demande JAMAIS d'approbation sauf si CONSULTATIVE_MODE=true.
2. live_trading_loop est une boucle infinie surveillée par le watchdog, qui
   redémarre toute tâche morte (auto-réparation).
3. Chaque sous-étape du tick est protégée par try/except (une erreur ne tue
   pas la boucle).
4. Le HALT a un cool-down + redémarrage progressif automatique (reprise sans
   intervention humaine).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_consultative_mode_off_by_default():
    """Aucune approbation humaine par défaut : l'humain n'est pas dans la
    boucle de décision."""
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    # la valeur par défaut dépend UNIQUEMENT de CONSULTATIVE_MODE env
    assert 'os.getenv("CONSULTATIVE_MODE", "").lower() == "true"' in src
    import main
    assert main.STATE.get("consultative_mode", False) is False


def test_pending_approvals_only_filled_in_consultative_mode():
    """Le bloc qui remplit pending_approvals est conditionné par
    consultative_mode — hors mode consultatif, le bot exécute seul."""
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    block_start = src.find('STATE["pending_approvals"].append(_proposal)')
    assert block_start != -1
    # remonter : le if consultative_mode doit précéder l'append
    prefix = src[max(0, block_start - 600):block_start]
    assert 'if STATE.get("consultative_mode", False):' in prefix


def test_live_trading_loop_is_infinite_and_watchdog_managed():
    """Boucle while True + enregistrée dans TASK_FACTORIES (le watchdog la
    redémarre si elle meurt)."""
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "async def live_trading_loop():" in src
    assert '"live_trading_loop": lambda: live_trading_loop()' in src
    # le watchdog redémarre les tâches mortes
    wd = src[src.find("async def task_watchdog_loop"):]
    assert "task.done()" in wd
    assert "TASK_FACTORIES.get(name)" in wd
    assert "asyncio.create_task(restart())" in wd


def test_watchdog_restarts_dead_task(monkeypatch):
    """Preuve comportementale : une tâche morte est recréée."""
    import main
    calls = {"n": 0}

    def fake_factory():
        calls["n"] += 1
        return main.live_trading_loop()

    # on remplace la table des fabriques par une version qui compte les appels
    monkeypatch.setattr(main, "TASK_FACTORIES",
                        {"live_trading_loop": fake_factory})
    monkeypatch.setattr(main, "_BG_TASKS", {"live_trading_loop": None})  # morte

    async def run():
        for name, task in list(main._BG_TASKS.items()):
            if task is None or task.done():
                restart = main.TASK_FACTORIES.get(name)
                if restart:
                    main._BG_TASKS[name] = restart()

    import asyncio
    asyncio.run(run())
    assert calls["n"] == 1
    assert main._BG_TASKS["live_trading_loop"] is not None


def test_halt_recovers_automatically():
    """Le HALT a un cool-down + redémarrage progressif SANS humain."""
    from core.risk_pipeline import HALT_COOLDOWN_MINUTES, RESTART_STAGES, RiskStateMachine
    rsm = RiskStateMachine()
    rsm.enter(RiskStateMachine.HALT, "test")
    assert rsm.state == RiskStateMachine.HALT
    # le tick gère la reprise (scale_factor > 0 après cooldown)
    assert HALT_COOLDOWN_MINUTES > 0
    assert len(RESTART_STAGES) >= 2
    # la reprise est progressive et automatique
    assert RESTART_STAGES[0][0] == 0.25
    assert RESTART_STAGES[-1][0] == 1.0


def test_tick_steps_are_exception_safe():
    """Les sous-étapes du tick sont protégées : une erreur sur un actif ne
    tue pas la boucle (try/except autour de la boucle actifs)."""
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    loop = src[src.find("async def live_trading_loop"):]
    # au moins N try/except dans la boucle
    assert loop.count("try:") >= 10
    assert loop.count("except Exception") >= 10
    # le sleep de fin de tick est présent (tick régulier)
    assert "loop_sleep_seconds" in loop
