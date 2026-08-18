"""
Macroeconomic Scheduled Events Calendar Engine — DONNÉES 100% RÉELLES.

Faille 1 corrigée (PDF — mentalité n°5 : « la confiance dans le signal compte
autant que le signal ») : l'ancienne version simulait des événements avec
`time.time() + 3600 * 4`, ce qui déclenchait des réductions de risque sur des
annonces inventées (FOMC « dans 4h » en boucle éternelle).

Nouveau comportement (honnête et sûr) :
1. Source PRIMAIRE : fichier `data/macro_events.json` (daté UTC, maintenu à la
   main à partir des calendriers officiels Fed / BLS / BCE / BoE).
2. Source SECONDAIRE (optionnelle) : API Finnhub « economic calendar » si la
   variable d'environnement FINNHUB_API_KEY est définie.
3. RÈGLE ABSOLUE : si aucune source réelle n'est disponible, le moteur renvoie
   UNAVAILABLE et N'APPLIQUE AUCUNE réduction de risque (principe : « en cas de
   doute, réduire ou s'arrêter » — mais jamais inventer de l'information).
"""
import json
import logging
import os
import time
from typing import Dict, List, Optional

logger = logging.getLogger("MacroCalendar")

# Facteurs de réduction par impact (documentés, cohérents avec le pipeline de risque)
IMPACT_REDUCTION = {
    "HIGH": 0.40,     # -60% de taille avant un événement majeur (Fed/BLS/BCE)
    "MEDIUM": 0.70,   # -30% avant un événement moyen
    "LOW": 0.90,      # -10% avant un événement mineur
}


class MacroeconomicCalendarEngine:
    """
    Calendrier macro événementiel alimenté par des données RÉELLES datées UTC.

    Ne génère JAMAIS d'événement. Si aucune source n'est disponible,
    `check_upcoming_macro_shocks()` renvoie UNAVAILABLE et ne réduit rien.
    """

    def __init__(self, calendar_file: Optional[str] = None):
        # Fichier de calendrier : variable d'env documentée, défaut sûr
        self.calendar_file = calendar_file or os.getenv(
            "MACRO_CALENDAR_FILE", "data/macro_events.json"
        )
        self.scheduled_events: List[dict] = []
        self.source_status = "UNAVAILABLE"
        self._finnhub_cache_ts = 0.0
        self._finnhub_cache = None
        self.load_calendar()

    # ------------------------------------------------------------------ #
    # Chargement des sources réelles
    # ------------------------------------------------------------------ #
    def load_calendar(self) -> None:
        """Charge le calendrier depuis le fichier JSON réel (source primaire)."""
        try:
            if not os.path.exists(self.calendar_file):
                logger.warning(
                    f"MacroCalendar: fichier {self.calendar_file} absent -> "
                    f"UNAVAILABLE (aucune réduction de risque appliquée)."
                )
                self.scheduled_events = []
                self.source_status = "UNAVAILABLE"
                return

            with open(self.calendar_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            raw = payload.get("events", []) if isinstance(payload, dict) else payload

            now = time.time()
            future = []
            for ev in raw:
                try:
                    ts = float(ev.get("timestamp", 0.0))
                except (TypeError, ValueError):
                    continue
                # On ne garde que les événements RÉELS et FUTURS (pas de date passée)
                if ts > now and ev.get("event"):
                    future.append({
                        "event": str(ev["event"]),
                        "timestamp": ts,
                        "impact": str(ev.get("impact", "MEDIUM")).upper(),
                        "source": str(ev.get("source", self.calendar_file)),
                    })

            future.sort(key=lambda e: e["timestamp"])
            self.scheduled_events = future
            self.source_status = f"file:{self.calendar_file} ({len(future)} événements futurs)"
            logger.info(
                f"MacroCalendar: {len(future)} événements réels chargés depuis "
                f"{self.calendar_file} (source {self.source_status})."
            )
        except Exception as e:
            logger.error(f"MacroCalendar: échec chargement {self.calendar_file}: {e}")
            self.scheduled_events = []
            self.source_status = "UNAVAILABLE"

    async def try_finnhub_calendar(self) -> bool:
        """
        Source secondaire optionnelle : Finnhub Economic Calendar (clé gratuite).
        Ne remplace PAS le fichier JSON ; vient en complément si la clé existe.
        """
        api_key = os.getenv("FINNHUB_API_KEY", "")
        if not api_key:
            return False
        # Cache 6h (le calendrier macro ne change pas à chaque tick)
        if time.time() - self._finnhub_cache_ts < 6 * 3600 and self._finnhub_cache:
            return True
        try:
            import httpx
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.get(
                    "https://api.finnhub.io/api/v1/calendar/economic",
                    params={"token": api_key},
                )
            if resp.status_code != 200:
                return False
            items = resp.json().get("economicCalendar", [])
            now = time.time()
            for it in items:
                # Finnhub fournit date (YYYY-MM-DD) et time (HH:MM, approximatif UTC)
                date_str = it.get("date", "")
                time_str = (it.get("time") or "12:00").strip()
                try:
                    import datetime as _dt
                    parsed = _dt.datetime.strptime(
                        f"{date_str} {time_str}", "%Y-%m-%d %H:%M"
                    ).replace(tzinfo=_dt.timezone.utc)
                    ts = parsed.timestamp()
                except Exception:
                    continue
                if ts <= now:
                    continue
                event_name = str(it.get("event", "")).upper().replace(" ", "_")
                impact = str(it.get("impact", "MEDIUM")).upper()
                if impact not in IMPACT_REDUCTION:
                    impact = "MEDIUM"
                # Évite les doublons avec le fichier JSON
                if not any(abs(e["timestamp"] - ts) < 3600 and e["event"] == event_name
                           for e in self.scheduled_events):
                    self.scheduled_events.append({
                        "event": event_name, "timestamp": ts,
                        "impact": impact, "source": "finnhub",
                    })
            self.scheduled_events.sort(key=lambda e: e["timestamp"])
            self._finnhub_cache = True
            self._finnhub_cache_ts = time.time()
            self.source_status += " + finnhub"
            return True
        except Exception as e:
            logger.warning(f"MacroCalendar: finnhub indisponible ({e})")
            return False

    # ------------------------------------------------------------------ #
    # API publique (signature conservée : appelée par main.py)
    # ------------------------------------------------------------------ #
    def check_upcoming_macro_shocks(self, warning_window_seconds=14400) -> dict:
        """
        Scanne le calendrier RÉEL. Si un événement à fort impact approche
        (fenêtre d'avertissement, défaut 4h), renvoie une demande de réduction
        préventive de l'exposition avec le facteur adapté à l'impact.

        Retour (contrat conservé pour main.py) :
            upcoming_shock, event, time_to_event_minutes, impact,
            scale_reduction_factor + source & status honnêtes.
        """
        current_epoch = time.time()

        if not self.scheduled_events:
            # RÈGLE D'HONNÊTETÉ : sans source réelle -> AUCUNE réduction,
            # jamais de faux événement (mentalité n°20 : zéro ego, honnêteté).
            return {
                "upcoming_shock": False,
                "scale_reduction_factor": 1.0,
                "source": self.source_status,
                "status": "UNAVAILABLE",
                "events_loaded": 0,
            }

        for ev in self.scheduled_events:
            time_to_event = ev["timestamp"] - current_epoch
            if 0 < time_to_event <= warning_window_seconds:
                impact = ev["impact"]
                factor = IMPACT_REDUCTION.get(impact, 1.0)
                logger.warning(
                    f"⏰ PRÉVENTION RÉELLE : événement macro '{ev['event']}' "
                    f"(impact {impact}, source {ev['source']}) dans "
                    f"{time_to_event/60:.1f} min -> facteur de taille {factor}."
                )
                return {
                    "upcoming_shock": True,
                    "event": ev["event"],
                    "time_to_event_minutes": time_to_event / 60.0,
                    "impact": impact,
                    "scale_reduction_factor": factor,
                    "source": ev["source"],
                    "status": "LIVE",
                    "events_loaded": len(self.scheduled_events),
                }

        return {
            "upcoming_shock": False,
            "scale_reduction_factor": 1.0,
            "source": self.source_status,
            "status": "LIVE",
            "events_loaded": len(self.scheduled_events),
        }

    def get_calendar(self, limit: int = 20) -> dict:
        """Exposition du calendrier réel pour l'API / dashboard (pilotage humain)."""
        now = time.time()
        return {
            "source": self.source_status,
            "status": "UNAVAILABLE" if not self.scheduled_events else "LIVE",
            "events": [
                {
                    "event": e["event"],
                    "timestamp_utc": e["timestamp"],
                    "in_minutes": round((e["timestamp"] - now) / 60.0, 1),
                    "impact": e["impact"],
                    "source": e["source"],
                }
                for e in self.scheduled_events[:limit]
            ],
        }

    def reload(self) -> None:
        """Recharge le calendrier (utilisable par un endpoint d'administration)."""
        self.load_calendar()
