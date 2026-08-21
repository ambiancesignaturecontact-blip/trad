"""
ALERTES OPÉRATIONNELLES AUTOMATIQUES (LOT F / F6 — autonomie opérationnelle).

F6 de l'audit : « dépendance à la surveillance humaine ». Le bot est autonome
sur l'exécution mais les événements critiques doivent être VISIBLES sans que
l'opérateur surveille le terminal en continu. Ce module envoie des alertes
Telegram automatiques pour :

  - CIRCUIT BREAKER / KILL SWITCH (drawdown quotidien ou lifetime atteint)
  - DRIFT PSI SÉVÈRE (distribution des features changée) + retour à la normale
  - transitions d'état risque (HALT/CAUTION -> alerte contextuelle)

Principes (les mêmes qui protègent le système en production) :
  1. JAMAIS bloquant : toute erreur d'envoi est loggée, jamais levée — le
     trading continue même si Telegram est down.
  2. DÉDUPLICATION : une alerte par ÉPISODE — cooldown par type (l'état de
     chaque type est mémorisé dans STATE) + alerte sur TRANSITION d'état
     seulement (un drift SEVERE persistant ne spamme pas toutes les 15 min).
  3. Silencieux sans configuration : pas de token/chat_id -> no-op propre
     (mode alert-silent existant).
  4. DÉMO == RÉAL : même chemin de code, aucun flag de mode.

Utilisation (main.py) :
    await send_ops_alert(telegram_bot, STATE, "circuit_breaker", texte, ...)
ou, dans la boucle :
    asyncio.create_task(send_ops_alert(telegram_bot, STATE, "drift_severe", ...))
"""
import logging
import time

logger = logging.getLogger("InstitutionalTradingBot")

# Cooldowns par type d'alerte (secondes). Une alerte de même type ne peut
# repartir qu'après ce délai — jamais de spam en cas d'état persistant.
ALERT_COOLDOWNS: dict[str, float] = {
    "circuit_breaker": 3600.0,   # 1h max (le kill switch est un événement rare)
    "drift_severe": 3600.0,      # 1h max (épisode, pas de répétition 15 min)
    "drift_recovered": 7200.0,   # 2h (le retour à la normale mérite un message,
                                 # mais pas plus d'une fois par épisode)
    "risk_state_halt": 3600.0,
    "risk_state_caution": 1800.0,
}

# Clé d'état persistante (dans STATE) pour les horodatages des alertes
LAST_ALERT_KEY = "ops_alerts_last_ts"


def _now() -> float:
    return time.time()


def drift_transition(previous_status: str | None, new_status: str) -> tuple[bool, str | None]:
    """
    Décide si une alerte drift est nécessaire, sur TRANSITION d'état :
      - STABLE/MODERATE/None -> SEVERE  : alerte « drift sévère détecté »
      - SEVERE -> STABLE/MODERATE       : alerte « drift résorbé »
      - sinon                           : pas d'alerte (état stable persistant)
    Retourne (alerter, alert_type) — alert_type dans {drift_severe,
    drift_recovered} ou None.
    """
    if previous_status is None:
        # premier calcul : si le tout premier état est déjà SEVERE, on alerte
        # (l'opérateur doit le savoir) ; sinon rien.
        return (True, "drift_severe") if new_status == "SEVERE" else (False, None)
    if new_status == "SEVERE" and previous_status != "SEVERE":
        return True, "drift_severe"
    if previous_status == "SEVERE" and new_status != "SEVERE":
        return True, "drift_recovered"
    return False, None


def circuit_breaker_text(reason: str, n_positions_flattened: int, mode: str) -> str:
    """Message actionnable pour un circuit breaker / kill switch."""
    return (
        "🚨 *KILL SWITCH ENGAGÉ — CIRCUIT BREAKER*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📉 Raison : `{reason[:120]}`\n"
        f"📦 Positions liquidées : *{n_positions_flattened}*\n"
        f"🛑 Mode : *{mode}* — trading arrêté\n"
        f"⛔ Nouveaux ordres bloqués automatiquement.\n"
        f"📋 Action requise : consulter le runbook §5 (DRAWDOWN) — "
        f"diagnostiquer la cause avant de réarmer via /api/toggle-bot."
    )


def drift_severe_text(detail: dict) -> str:
    """Message pour un drift PSI sévère (distribution des features changée)."""
    sources = detail.get("sources", {}) if isinstance(detail, dict) else {}
    psi = detail.get("max_psi", 0.0)
    return (
        "🧨 *DRIFT SÉVÈRE DÉTECTÉ*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 PSI max : *{float(psi):.3f}* (seuil 0.60)\n"
        f"🔍 Sources : PSI={sources.get('psi', '?')} / CUSUM={sources.get('cusum', '?')}\n"
        f"🔄 Oubli du bandit accéléré (decay {detail.get('bandit_decay_recommended', 0.92):.3f}) — "
        f"le système ré-explore plutôt que de récompenser un edge mort.\n"
        f"📋 Action : consulter le runbook §4 (DRIFT)."
    )


def drift_recovered_text(detail: dict) -> str:
    """Message de retour à la normale après un épisode de drift."""
    return (
        "✅ *DRIFT RÉSORBÉ*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Le PSI est revenu sous le seuil sévère (max {float(detail.get('max_psi', 0.0)):.3f}).\n"
        f"🔄 L'oubli du bandit revient au rythme nominal (0.98).\n"
        f"📈 Le système continue en autonomie."
    )


async def maybe_alert_drift_transition(telegram_bot, state: dict,
                                       previous_status: str | None) -> None:
    """
    LOT F (F6) : alerte automatique sur TRANSITION du statut de drift
    (unifié PSI + CUSUM) :
      - STABLE/MODERATE -> SEVERE : « drift sévère détecté »
      - SEVERE -> STABLE/MODERATE : « drift résorbé »
    Dédupliquée par épisode (cooldown), jamais bloquante, silencieuse sans
    config Telegram. previous_status = statut AVANT le dernier run_drift_check
    (None = premier calcul).
    """
    try:
        unified = state.get("drift_psi", {}).get("unified", {})
        new_status = unified.get("status")
        if not new_status:
            return
        alert, alert_type = drift_transition(previous_status, new_status)
        if not alert:
            return
        text = (drift_severe_text(unified) if alert_type == "drift_severe"
                else drift_recovered_text(unified))
        await send_ops_alert(telegram_bot, state, alert_type, text)
    except Exception as e:
        # JAMAIS bloquant : le trading continue même si l'alerte échoue
        logger.warning(f"Drift ops alert failed ({e}) — non bloquant")


async def send_ops_alert(telegram_bot, state: dict, alert_type: str,
                         text: str, force: bool = False) -> bool:
    """
    Envoie une alerte opérationnelle DÉDUPLIQUÉE (cooldown par type, mémorisé
    dans state["ops_alerts_last_ts"]). Retourne True si envoyée.

    - telegram_bot None ou non configuré -> no-op (mode alert-silent).
    - Jamais bloquant : toute erreur est loggée et retourne False.
    - force=True ignore le cooldown (événements vraiment critiques).
    """
    if telegram_bot is None:
        return False
    cooldown = ALERT_COOLDOWNS.get(alert_type, 3600.0)
    if not force:
        last = (state.get(LAST_ALERT_KEY) or {}).get(alert_type, 0.0)
        if _now() - float(last) < cooldown:
            logger.debug(f"Ops alert '{alert_type}' supprimée (cooldown {cooldown:.0f}s)")
            return False
    try:
        sent = await telegram_bot.send_push_notification(text)
        if sent:
            state.setdefault(LAST_ALERT_KEY, {})[alert_type] = _now()
            logger.info(f"📣 Alerte opérationnelle envoyée : {alert_type}")
        return bool(sent)
    except Exception as e:
        # JAMAIS bloquant : le trading continue même si l'alerte échoue
        logger.warning(f"Ops alert '{alert_type}' échouée ({e}) — non bloquant")
        return False
