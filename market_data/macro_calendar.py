"""
SHIM DE DÉPRÉCIATION — P2-16 (audit indépendant §5-16).

Ce fichier contenait l'ANCIENNE version du calendrier macro avec des
événements SIMULÉS (annonces « dans 4 heures » recréées en boucle éternelle
sur l'horloge du process). Cette faille a été corrigée dans
`models/macro_calendar.py` (données 100% réelles, `data/macro_events.json`
daté UTC, règle absolue UNAVAILABLE sans source réelle).

Conformément au pattern `ai/` (shims de dépréciation), ce module n'est plus
qu'un passe-plat vers l'implémentation réelle : tout import de
`market_data.macro_calendar` reçoit le moteur RÉEL, jamais l'ancien code.
À supprimer quand aucun import ne subsistera.
"""
from models.macro_calendar import (  # noqa: F401
    MacroeconomicCalendarEngine,
    IMPACT_REDUCTION,
    EVENT_ACTIVE_WINDOW,
    EVENT_AFTERMATH_WINDOW,
)
from models.macro_calendar import *  # noqa: F401,F403
