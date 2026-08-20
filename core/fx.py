"""
Moteur FX — devise de compte configurable (multi-devise DEMO == REAL).

Principe institutionnel (base currency) : un portefeuille multi-devises se
mesure dans UNE devise de référence — celle du compte. Chaque actif est
libellé dans sa devise de cotation (USD pour BTCUSDT/XAUUSD/AAPL..., EURUSD
est l'exception : sa cotation EST un taux de change) et TOUTE valeur affichée
(balance, équité, PnL, min notional) est convertie dans la devise du compte.

Sources : open.er-api.com (taux réels gratuits, base USD, ~60 devises,
mise à jour quotidienne) — la même source déjà utilisée pour EURUSD dans
market_data/multi_source.py. Cache 5 min. Honnêteté : si la source est
indisponible, on retourne None et l'affichage reste en USD avec un
avertissement (jamais de taux inventé).

Devise de compte : config.yaml -> account.currency (ou env ACCOUNT_CURRENCY),
défaut "USD".
"""
import logging
import os
import time

import httpx

logger = logging.getLogger("FX")

# Devises supportées (celles réelement disponibles sur er-api + raisonnables
# pour un compte de trading)
SUPPORTED_CURRENCIES = ("USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "CNY")

# Cache court : les taux bougent peu dans la minute, et on ne doit pas
# marteler l'API gratuite (rate limit ~1500 req/mois)
_CACHE_TTL_SECONDS = 300.0
_cache: dict[str, dict] = {"ts": 0.0, "rates": {}}


def get_account_currency() -> str:
    """Devise de base du compte (configurable, défaut USD)."""
    ccy = os.getenv("ACCOUNT_CURRENCY", "").strip().upper()
    if ccy in SUPPORTED_CURRENCIES:
        return ccy
    try:
        from core.config import settings
        ccy = str(settings.get("account", "currency", "USD")).strip().upper()
        if ccy in SUPPORTED_CURRENCIES:
            return ccy
    except Exception:
        pass
    return "USD"


def fetch_usd_rates() -> dict[str, float] | None:
    """Taux réels USD -> toutes devises (er-api). None si source indisponible."""
    now = time.time()
    if now - _cache["ts"] < _CACHE_TTL_SECONDS and _cache["rates"]:
        return _cache["rates"]
    try:
        resp = httpx.get("https://open.er-api.com/v6/latest/USD", timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("result") == "success":
                rates = {k: float(v) for k, v in (data.get("rates") or {}).items()}
                if rates:
                    _cache["ts"] = now
                    _cache["rates"] = rates
                    return rates
    except Exception as e:
        logger.warning(f"FX: er-api indisponible ({e}) — conversion impossible")
    return None


def usd_to(currency: str) -> float | None:
    """Taux de conversion 1 USD -> `currency`. None si devise inconnue ou
    source indisponible (jamais de taux inventé)."""
    if currency == "USD":
        return 1.0
    rates = fetch_usd_rates()
    if not rates:
        return None
    rate = rates.get(currency)
    return float(rate) if rate else None


def convert(value_usd: float, currency: str = None) -> float | None:
    """Convertit une valeur USD dans la devise du compte (ou celle demandée).
    Retourne None si la conversion n'est pas possible (source indisponible)."""
    ccy = currency or get_account_currency()
    rate = usd_to(ccy)
    if rate is None:
        return None
    return value_usd * rate


def display(value_usd: float, currency: str = None) -> dict:
    """Représentation d'affichage honnête : valeur convertie + devise + taux.
    Si la conversion échoue, on affiche la valeur USD avec le taux inconnu
    (l'utilisateur voit que la devise n'a pas pu être convertie)."""
    ccy = currency or get_account_currency()
    rate = usd_to(ccy)
    if rate is None:
        return {"value": value_usd, "currency": "USD", "fx_rate": None,
                "note": "FX indisponible — affiché en USD"}
    return {"value": value_usd * rate, "currency": ccy, "fx_rate": rate}
