"""
HELPERS D'ÉTAT (LOT 8 — architecture, découplage).
Extraits de main.py (AST, corps inchangés) ; symboles partagés importés de
main de façon EXPLICITE (main est complet quand ce module est importé, en fin
de main.py).
"""
import logging
import time

from main import STATE, DataQualityStatus, order_flow, platform_metrics  # noqa: E402

logger = logging.getLogger("InstitutionalTradingBot")


def set_data_quality(status):
    """Tracks market-data quality per source into STATE + Prometheus gauge."""
    STATE["data_quality_status"] = status
    try:
        mapping = {
            DataQualityStatus.LIVE: 4.0,
            DataQualityStatus.DELAYED: 3.0,
            DataQualityStatus.STALE: 2.0,
            DataQualityStatus.INVALID: 1.0,
            DataQualityStatus.DISCONNECTED: 0.0,
            DataQualityStatus.UNAVAILABLE: 0.0,
        }
        platform_metrics.DATA_QUALITY.labels(source="market").set(mapping.get(status, 0.0))
    except Exception:
        pass


def set_asset_quality(symbol: str, status: str):
    """
    Qualité de données PAR ACTIF (faille 1 corrigée — mentalité n°5 : chaque
    donnée doit avoir un score de confiance). Un actif dont la source est
    indisponible est marqué UNAVAILABLE et NE PEUT PAS être tradé.
    """
    STATE.setdefault("asset_data_status", {})[symbol] = status
    STATE["assets"].setdefault(symbol, {})["data_status"] = status
    if status == DataQualityStatus.UNAVAILABLE:
        STATE["assets"][symbol]["has_real_price"] = False


def _neutral(value, default: float = 0.0) -> float:
    """
    Convertit un indicateur éventuellement indisponible en valeur NEUTRE.
    Contrairement à une donnée inventée, 0.0 = « aucune information » et
    n'apporte aucune direction à la décision (mentalité n°20 : je ne sais pas).
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def record_open_position(symbol: str, strategy: str, entry_price: float) -> None:
    """LOT 2 : mémorise la stratégie responsable d'une position ouverte
    (alimente le win rate RÉEL par stratégie au moment de la clôture)."""
    if strategy:
        STATE.setdefault("position_strategies", {})[symbol] = {
            "strategy": strategy,
            "entry_price": float(entry_price),
            "ts": time.time(),
        }


def mark_real_price(symbol: str, price: float, volume_24h=None):
    """
    Enregistre un prix RÉEL reçu d'une source de marché. Met à jour le flag
    has_real_price (seul vrai « feu vert » pour trader cet actif).
    """
    STATE["assets"][symbol]["price"] = float(price)
    STATE["assets"][symbol]["has_real_price"] = True
    STATE["assets"][symbol]["data_status"] = DataQualityStatus.LIVE
    STATE.setdefault("last_known_prices", {})[symbol] = float(price)
    if volume_24h is not None:
        STATE["assets"][symbol]["volume_24h"] = float(volume_24h)
    # Le dernier prix global réel (BTC) alimente le dashboard
    if symbol == "BTCUSDT":
        STATE["last_price"] = float(price)
        STATE["price_history"].append(float(price))
        if len(STATE["price_history"]) > 120:
            STATE["price_history"] = STATE["price_history"][-120:]


def update_asset_order_book(symbol: str, bids: list, asks: list, exchange: str = "bybit"):
    """
    Met à jour le carnet d'ordres RÉEL d'un actif (multi-assets, multi-exchange).
    Stocke le carnet PAR exchange puis consolide le BEST BOOK (meilleur spread)
    dans order_books[symbol] — `order_book` reste l'alias historique pour BTCUSDT.
    """
    STATE.setdefault("exchange_order_books", {}).setdefault(exchange, {})[symbol] = {
        "bids": bids, "asks": asks, "_ts": time.time(),
    }
    # Consolidation BBO : le carnet de l'exchange avec le meilleur spread
    best = None
    for ex, books in STATE.get("exchange_order_books", {}).items():
        b = books.get(symbol)
        if not b or not b.get("bids") or not b.get("asks"):
            continue
        try:
            spread = float(b["asks"][0][0]) - float(b["bids"][0][0])
        except Exception:
            continue
        if best is None or spread < best[0]:
            best = (spread, ex, b)
    if best is not None:
        consolidated = {"bids": best[2]["bids"], "asks": best[2]["asks"],
                        "exchange": best[1]}
        STATE.setdefault("order_books", {})[symbol] = consolidated
        if symbol == "BTCUSDT":
            STATE["order_book"] = consolidated
    STATE.setdefault("asset_data_status", {})[symbol] = DataQualityStatus.LIVE
    STATE["assets"][symbol]["data_status"] = DataQualityStatus.LIVE
    # LOT 3 : alimente l'OFI du moteur d'order flow (pression bid vs ask)
    try:
        _bd = sum(float(b[1]) for b in bids if b and len(b) > 1)
        _ad = sum(float(a[1]) for a in asks if a and len(a) > 1)
        order_flow.update_book(symbol, _bd, _ad)
    except Exception:
        pass
