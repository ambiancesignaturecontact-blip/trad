"""
LOT F (F6 — autonomie opérationnelle) : alertes automatiques + runbook.

Vérifié ici :
  1. OpsAlertManager : déduplication par type (cooldown), transition d'état
     drift (sévère détecté / résorbé), messages actionnables, silencieux
     sans config Telegram, JAMAIS bloquant (erreur -> log, pas d'exception).
  2. Intégration main.py : alerte circuit breaker au trip + alerte drift
     après run_drift_check (câblage présent).
  3. RUNBOOK : sections drift / drawdown / HALT / autonomie présentes.
  4. DÉMO == RÉAL : aucun flag de mode dans les alertes.
"""
import asyncio
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from core.ops_alerts import (  # noqa: E402
    ALERT_COOLDOWNS,
    LAST_ALERT_KEY,
    circuit_breaker_text,
    drift_recovered_text,
    drift_severe_text,
    drift_transition,
    maybe_alert_drift_transition,
    send_ops_alert,
)


# --------------------------------------------------------------------------- #
# 1. Transition d'état drift
# --------------------------------------------------------------------------- #
class TestDriftTransition:
    def test_entering_severe_alerts(self):
        alert, atype = drift_transition("STABLE", "SEVERE")
        assert alert is True and atype == "drift_severe"
        alert, atype = drift_transition("MODERATE", "SEVERE")
        assert alert is True and atype == "drift_severe"

    def test_recovery_alerts(self):
        alert, atype = drift_transition("SEVERE", "STABLE")
        assert alert is True and atype == "drift_recovered"
        alert, atype = drift_transition("SEVERE", "MODERATE")
        assert alert is True and atype == "drift_recovered"

    def test_no_alert_on_persistent_state(self):
        assert drift_transition("SEVERE", "SEVERE") == (False, None)
        assert drift_transition("STABLE", "STABLE") == (False, None)
        assert drift_transition("MODERATE", "MODERATE") == (False, None)

    def test_first_computation(self):
        """Premier calcul : alerte seulement si d'emblée SEVERE (l'opérateur
        doit le savoir), sinon rien (pas de bruit au démarrage)."""
        alert, atype = drift_transition(None, "SEVERE")
        assert alert is True and atype == "drift_severe"
        assert drift_transition(None, "STABLE") == (False, None)
        assert drift_transition(None, "MODERATE") == (False, None)


# --------------------------------------------------------------------------- #
# 2. Messages actionnables
# --------------------------------------------------------------------------- #
class TestMessages:
    def test_circuit_breaker_text_includes_action(self):
        t = circuit_breaker_text("DAILY DRAWDOWN BREACHED (12.0%)", 3, "DEMO")
        assert "KILL SWITCH" in t
        assert "3" in t
        assert "DEMO" in t
        assert "runbook" in t.lower()

    def test_drift_severe_text_includes_action(self):
        t = drift_severe_text({"max_psi": 0.97, "sources": {"psi": "SEVERE", "cusum": "OK"},
                               "bandit_decay_recommended": 0.92})
        assert "0.97" in t
        assert "0.92" in t
        assert "runbook" in t.lower()

    def test_drift_recovered_text(self):
        t = drift_recovered_text({"max_psi": 0.2})
        assert "RÉSORBÉ" in t


# --------------------------------------------------------------------------- #
# 3. Déduplication + cooldown + jamais bloquant
# --------------------------------------------------------------------------- #
class FakeBot:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    async def send_push_notification(self, text):
        if self.fail:
            raise RuntimeError("telegram down")
        self.sent.append(text)
        return True


class TestSendOpsAlert:
    def test_cooldown_deduplicates(self):
        state = {}
        bot = FakeBot()
        async def run():
            r1 = await send_ops_alert(bot, state, "drift_severe", "A", force=True)
            r2 = await send_ops_alert(bot, state, "drift_severe", "B")
            return r1, r2
        r1, r2 = asyncio.run(run())
        assert r1 is True and r2 is False
        assert len(bot.sent) == 1
        assert "drift_severe" in state[LAST_ALERT_KEY]

    def test_force_bypasses_cooldown(self):
        state = {}
        bot = FakeBot()
        async def run():
            await send_ops_alert(bot, state, "circuit_breaker", "A", force=True)
            return await send_ops_alert(bot, state, "circuit_breaker", "B", force=True)
        r = asyncio.run(run())
        assert r is True
        assert len(bot.sent) == 2

    def test_different_types_independent_cooldowns(self):
        state = {}
        bot = FakeBot()
        async def run():
            await send_ops_alert(bot, state, "drift_severe", "A", force=True)
            return await send_ops_alert(bot, state, "drift_recovered", "B")
        r = asyncio.run(run())
        assert r is True
        assert len(bot.sent) == 2

    def test_no_bot_is_noop(self):
        async def run():
            return await send_ops_alert(None, {}, "drift_severe", "A")
        assert asyncio.run(run()) is False

    def test_bot_error_never_raises(self):
        state = {}
        bot = FakeBot(fail=True)
        async def run():
            return await send_ops_alert(bot, state, "drift_severe", "A", force=True)
        assert asyncio.run(run()) is False   # jamais levée, jamais bloquant

    def test_cooldowns_are_sane(self):
        assert ALERT_COOLDOWNS["drift_severe"] >= 1800
        assert ALERT_COOLDOWNS["circuit_breaker"] >= 1800


class TestMaybeAlertDriftTransition:
    def test_alerts_on_transition_to_severe(self):
        state = {"drift_psi": {"unified": {"status": "SEVERE", "max_psi": 0.9,
                                           "sources": {"psi": "SEVERE", "cusum": "OK"},
                                           "bandit_decay_recommended": 0.92}}}
        bot = FakeBot()
        async def run():
            await maybe_alert_drift_transition(bot, state, "STABLE")
        asyncio.run(run())
        assert len(bot.sent) == 1
        assert "DRIFT SÉVÈRE" in bot.sent[0]

    def test_alerts_on_recovery(self):
        state = {"drift_psi": {"unified": {"status": "STABLE", "max_psi": 0.2,
                                           "sources": {"psi": "STABLE", "cusum": "OK"},
                                           "bandit_decay_recommended": 0.98}}}
        bot = FakeBot()
        async def run():
            await maybe_alert_drift_transition(bot, state, "SEVERE")
        asyncio.run(run())
        assert len(bot.sent) == 1
        assert "RÉSORBÉ" in bot.sent[0]

    def test_no_alert_on_persistent(self):
        state = {"drift_psi": {"unified": {"status": "SEVERE", "max_psi": 0.9,
                                           "sources": {}, "bandit_decay_recommended": 0.92}}}
        bot = FakeBot()
        async def run():
            await maybe_alert_drift_transition(bot, state, "SEVERE")
        asyncio.run(run())
        assert bot.sent == []

    def test_never_raises_on_broken_state(self):
        bot = FakeBot()
        async def run():
            await maybe_alert_drift_transition(bot, {"drift_psi": None}, None)
        asyncio.run(run())   # ne doit pas lever


# --------------------------------------------------------------------------- #
# 4. Intégration main.py (câblage réel)
# --------------------------------------------------------------------------- #
class TestMainWiring:
    def test_circuit_breaker_alert_wired(self):
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        assert 'send_ops_alert(telegram_bot, STATE, "circuit_breaker"' in src
        assert "circuit_breaker_text(msg, _flattened_count, active_mode)" in src

    def test_drift_alert_wired_after_check(self):
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        assert "run_drift_check(STATE" in src
        assert "maybe_alert_drift_transition(telegram_bot, STATE" in src
        # l'alerte suit le check (même tick)
        i_check = src.find("run_drift_check(STATE")
        i_alert = src.find("maybe_alert_drift_transition(telegram_bot, STATE")
        assert 0 < i_check < i_alert

    def test_no_mode_flag_in_ops_alerts(self):
        """DÉMO == RÉAL : aucune branche de mode dans les alertes."""
        src = inspect.getsource(send_ops_alert) + inspect.getsource(drift_transition)
        assert "active_mode" not in src
        assert '"DEMO"' not in src and '"REAL"' not in src


# --------------------------------------------------------------------------- #
# 5. Runbook
# --------------------------------------------------------------------------- #
class TestRunbook:
    def test_runbook_has_drift_section(self):
        rb = (ROOT / "RUNBOOK.md").read_text(encoding="utf-8")
        assert "## 7. DRIFT" in rb
        assert "DRIFT SÉVÈRE DÉTECTÉ" in rb

    def test_runbook_has_drawdown_section(self):
        rb = (ROOT / "RUNBOOK.md").read_text(encoding="utf-8")
        assert "## 8. DRAWDOWN / KILL SWITCH" in rb
        assert "réarmement manuel" in rb or "Réarmer" in rb

    def test_runbook_has_halt_section(self):
        rb = (ROOT / "RUNBOOK.md").read_text(encoding="utf-8")
        assert "## 9. HALT" in rb
        assert "redémarrage progressif" in rb

    def test_runbook_has_autonomy_table(self):
        rb = (ROOT / "RUNBOOK.md").read_text(encoding="utf-8")
        assert "## 10. AUTONOMIE OPÉRATIONNELLE" in rb
        assert "LOT B" in rb and "LOT D" in rb
