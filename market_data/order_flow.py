"""
ORDER FLOW ENGINE — Données 100% RÉELLES (PROMPT MAÎTRE, Pilier H / Faille 4).

Compétence signature des pros (PDF, section G) : le bot lit l'INTENTION réelle
derrière les ordres, au-delà des indicateurs de prix :
  • Delta / CVD      : volume agressif ACHAT - volume agressif VENTE (tape
                       reading via flux de trades publics réels).
  • OFI              : Order Flow Imbalance du carnet (pression bid vs ask).
  • Absorption       : gros flux agressif absorbé SANS mouvement de prix
                       (signe d'un acteur déterminé qui défend un niveau).
  • Liquidations     : cascades de liquidation (flux !forceOrder / liquidation)
                       -> ne JAMAIS acheter dans la panique, attendre la fin.

Exploitation dans la décision (PDF Pilier H) :
  (a) refuser d'entrer CONTRE un flux agressif dominant ;
  (b) attendre la fin d'une cascade de liquidation avant d'acheter la panique ;
  (c) ne JAMAIS placer un stop dans une zone de stops évidente (stop hunting) ;
  (d) réduire la taille quand le flux est toxique (VPIN/Kyle/delta extrême).

Aucune donnée inventée : si aucun trade réel n'est reçu, tous les facteurs
renvoient leur valeur NEUTRE (1.0 / False) — jamais un signal fabriqué.
"""
import logging
import time
from collections import deque

from core.config import settings

logger = logging.getLogger("OrderFlow")

# P1-8 (audit §3) : constantes branchées sur core/config.py (config.yaml).
# Fenêtres de calcul (secondes)
DELTA_WINDOW = settings.get_float("orderflow", "delta_window_seconds", 60.0)
CASCADE_WINDOW = settings.get_float("orderflow", "cascade_window_seconds", 30.0)
ABSORPTION_WINDOW = settings.get_float("orderflow", "absorption_window_seconds", 30.0)

# Seuils (calibrés, documentés)
TOXIC_DELTA_RATIO = settings.get_float("orderflow", "toxic_delta_ratio", 0.35)
MIN_TOXIC_VOLUME_USD = settings.get_float("orderflow", "min_toxic_volume_usd", 100000.0)
CASCADE_MIN_EVENTS = settings.get_int("orderflow", "cascade_min_events", 3)
CASCADE_MIN_NOTIONAL = settings.get_float("orderflow", "cascade_min_notional_usd", 200000.0)
ABSORPTION_VOLUME_USD = settings.get_float("orderflow", "absorption_volume_usd", 300000.0)


class OrderFlowEngine:
    """Moteur d'order flow multi-actifs, alimenté par les flux WS réels."""

    def __init__(self):
        # symbol -> deque de (ts, side(+1 buy/-1 sell), qty, price)
        self.trades: dict[str, deque] = {}
        # symbol -> dict de liquidations récentes
        self.liquidations: dict[str, deque] = {}
        # symbol -> CVD cumulé (départ à 0, delta signé)
        self.cvd: dict[str, float] = {}
        # symbol -> (book_bid_qty, book_ask_qty, ts)
        self.book: dict[str, tuple[float, float, float]] = {}
        # symbol -> snapshot prix pour l'absorption
        self.price_ref: dict[str, tuple[float, float]] = {}  # (prix, ts)

    # ------------------------------------------------------------------ #
    # INGESTION DES DONNÉES RÉELLES
    # ------------------------------------------------------------------ #
    def update_trade(self, symbol: str, price: float, qty: float, side: str) -> None:
        """
        Enregistre un trade EXÉCUTÉ réel (flux public). `side` = côté AGRESSIF
        (maker taker) : "buy" = achat agressif, "sell" = vente agressive.
        """
        if price <= 0 or qty <= 0:
            return
        d = self.trades.setdefault(symbol, deque())
        d.append((time.time(), 1.0 if side == "buy" else -1.0, qty, price))
        # purge des vieux trades (> 2 fenêtres)
        cutoff = time.time() - 2 * DELTA_WINDOW
        while d and d[0][0] < cutoff:
            d.popleft()
        # CVD cumulé
        self.cvd[symbol] = self.cvd.get(symbol, 0.0) + (qty if side == "buy" else -qty)

    def update_trade_unknown_side(self, symbol: str, price: float, qty: float,
                                  prev_price: float | None = None) -> None:
        """
        Trade sans indication de côté (certains flux) : on applique la tick rule
        (le prix montait -> acheteur agressif). Sans prix précédent, NEUTRE.
        """
        if prev_price is not None and prev_price > 0:
            side = "buy" if price >= prev_price else "sell"
            self.update_trade(symbol, price, qty, side)
        # sinon : on ignore (pas d'info -> pas de signal fabriqué)

    def update_book(self, symbol: str, bid_qty: float, ask_qty: float) -> None:
        """Met à jour la profondeur agrégée (5 niveaux) pour l'OFI."""
        self.book[symbol] = (float(bid_qty), float(ask_qty), time.time())

    def update_liquidation(self, symbol: str, side: str, qty: float, price: float) -> None:
        """Enregistre une liquidation RÉELLE (flux forceOrder/liquidation)."""
        if qty <= 0 or price <= 0:
            return
        d = self.liquidations.setdefault(symbol, deque())
        notional = qty * price
        d.append((time.time(), side, notional))
        cutoff = time.time() - CASCADE_WINDOW
        while d and d[0][0] < cutoff:
            d.popleft()

    # ------------------------------------------------------------------ #
    # INDICATEURS (tous neutres si pas de données réelles)
    # ------------------------------------------------------------------ #
    def get_delta(self, symbol: str, window: float = DELTA_WINDOW) -> tuple[float, float]:
        """
        Delta agressif (volume achat - volume vente) et volume total sur la
        fenêtre. Retourne (delta, volume_total) ; (0,0) si aucune donnée.
        """
        d = self.trades.get(symbol)
        if not d:
            return 0.0, 0.0
        cutoff = time.time() - window
        delta = 0.0
        vol = 0.0
        for ts, side, qty, _ in d:
            if ts >= cutoff:
                delta += side * qty
                vol += qty
        return delta, vol

    def get_cvd(self, symbol: str) -> float:
        """Cumulative Volume Delta signé (accumulation/distribution)."""
        return float(self.cvd.get(symbol, 0.0))

    def compute_ofi(self, symbol: str) -> float | None:
        """
        Order Flow Imbalance du carnet : (bid_qty - ask_qty)/(bid_qty + ask_qty).
        +1 = pression achat massive, -1 = pression vente, None = pas de carnet.
        """
        b = self.book.get(symbol)
        if not b:
            return None
        bid_qty, ask_qty, ts = b
        if time.time() - ts > 30.0:
            return None  # carnet trop vieux -> neutre
        total = bid_qty + ask_qty
        if total <= 0:
            return None
        return (bid_qty - ask_qty) / total

    def detect_absorption(self, symbol: str) -> bool:
        """
        Absorption : gros volume agressif échangé SANS mouvement de prix
        significatif (< 0.15 %). Signe d'un acteur qui défend un niveau.
        """
        d = self.trades.get(symbol)
        if not d or len(d) < 20:
            return False
        cutoff = time.time() - ABSORPTION_WINDOW
        recent = [t for t in d if t[0] >= cutoff]
        if len(recent) < 10:
            return False
        vol_usd = sum(q * p for _, _, q, p in recent)
        if vol_usd < ABSORPTION_VOLUME_USD:
            return False
        prices = [p for _, _, _, p in recent]
        move_pct = (max(prices) - min(prices)) / (sum(prices) / len(prices)) * 100.0
        return move_pct < 0.15

    def liquidation_cascade_active(self, symbol: str) -> bool:
        """Cascade de liquidations active (>= 3 événements / >= $200k notional
        dans les 30 dernières secondes)."""
        d = self.liquidations.get(symbol)
        if not d:
            return False
        cutoff = time.time() - CASCADE_WINDOW
        recent = [liq for liq in d if liq[0] >= cutoff]
        if len(recent) < CASCADE_MIN_EVENTS:
            return False
        total_notional = sum(n for _, _, n in recent)
        return total_notional >= CASCADE_MIN_NOTIONAL

    # ------------------------------------------------------------------ #
    # EXPLOITATION DANS LA DÉCISION (Pilier H a/b/c/d)
    # ------------------------------------------------------------------ #
    def toxicity_factor(self, symbol: str, vpin: float | None = None) -> float:
        """
        Facteur de taille (0..1) quand le flux est TOXIQUE (informed trading) :
          - delta agressif extrême dans UNE direction (déséquilibre > 35 %)
          - VPIN élevé (probabilité de flux informé)
          - OFI extrême
        Retourne 1.0 (neutre) si aucune donnée réelle. Jamais de signal inventé.
        """
        factor = 1.0
        delta, vol = self.get_delta(symbol)
        # Volume minimum en USD : sans échantillon suffisant, pas de jugement
        # de toxicité (le bruit ne doit jamais réduire la taille).
        if vol > 0:
            vol_usd = 0.0
            for ts, side, qty, price in self.trades.get(symbol, ()):
                if ts >= time.time() - DELTA_WINDOW:
                    vol_usd += qty * price
            if vol_usd >= MIN_TOXIC_VOLUME_USD:
                ratio = abs(delta) / vol
                if ratio > TOXIC_DELTA_RATIO:
                    # flux très déséquilibré -> on réduit (le marché « sait » quelque chose)
                    factor *= max(0.4, 1.0 - (ratio - TOXIC_DELTA_RATIO) * 2.0)
        if vpin is not None:
            # FIX (logs prod) : VPIN est une probabilité NORMALISÉE -> il est
            # mathématiquement borné à [0,1]. Une valeur > 1.0 (ex: 4276, 6.9M
            # observés sur barres à volume constant) est une ERREUR de calcul,
            # PAS un signal de toxicité. On l'ignore (neutre) au lieu de la
            # clamper à 1.0 — sinon TOUS les actifs étaient réduits à 0.6 en
            # permanence (mentalité n°5 : ne pas réagir à une donnée douteuse).
            vpin_f = float(vpin)
            if 0.0 <= vpin_f <= 1.0:
                if vpin_f > 0.75:
                    factor *= 0.6
                elif vpin_f > 0.65:
                    factor *= 0.8
        ofi = self.compute_ofi(symbol)
        if ofi is not None and abs(ofi) > 0.6:
            factor *= 0.8
        return max(0.2, min(1.0, factor))

    def should_avoid_entry(self, symbol: str, side: str) -> tuple[bool, str]:
        """
        (a) Refuser d'entrer CONTRE un flux agressif dominant.
        BUY alors que le delta est fortement négatif (vente agressive) ->
        on ne prend pas le couteau. SELL alors que le delta est fortement
        positif -> idem. Sans données réelles : False (neutre).
        """
        delta, vol = self.get_delta(symbol)
        if vol <= 0:
            return False, ""
        # Volume minimum en USD : ne jamais bloquer un trade sur un échantillon
        # minuscule (bruit) — la confiance dans le signal exige des données.
        vol_usd = 0.0
        for ts, side2, qty, price in self.trades.get(symbol, ()):
            if ts >= time.time() - DELTA_WINDOW:
                vol_usd += qty * price
        if vol_usd < MIN_TOXIC_VOLUME_USD:
            return False, ""
        ratio = delta / vol  # -1..+1
        if side == "BUY" and ratio < -TOXIC_DELTA_RATIO:
            return True, (f"flux vendeur agressif dominant (delta {ratio*100:.0f}% "
                          f"du volume) -> pas d'achat contre le flux")
        if side == "SELL" and ratio > TOXIC_DELTA_RATIO:
            return True, (f"flux acheteur agressif dominant (delta {ratio*100:.0f}% "
                          f"du volume) -> pas de vente contre le flux")
        return False, ""

    def wait_cascade_end(self, symbol: str) -> tuple[bool, str]:
        """
        (b) Cascade de liquidations active -> ne pas entrer (surtout pas
        « acheter la panique ») ; attendre la fin de la cascade.
        """
        if self.liquidation_cascade_active(symbol):
            return True, "cascade de liquidations en cours -> entrée différée"
        return False, ""

    def stop_hunting_zone(self, symbol: str, recent_high: float,
                          recent_low: float, atr: float) -> tuple[float | None, float | None]:
        """
        (c) Zones de stops évidentes (stop hunting) : les stops longs se
        concentrent SOUS le plus bas récent, les stops shorts AU-DESSUS du
        plus haut récent. Retourne (zone_haute, zone_basse) en prix, ou None.
        """
        if recent_high <= 0 or recent_low <= 0 or atr <= 0:
            return None, None
        # bande de ~0.5 ATR autour des extrêmes récents
        zone_high = recent_high + 0.5 * atr   # au-dessus du plus haut
        zone_low = recent_low - 0.5 * atr     # sous le plus bas
        return zone_high, zone_low

    def adjust_stop_against_hunting(self, symbol: str, stop_price: float,
                                    direction: str, recent_high: float,
                                    recent_low: float, atr: float) -> float:
        """
        (c) Déplace le stop HORS de la zone de chasse si le stop calculé y
        tomberait (un stop trop évident se fait « chasser » avant le retour).
        direction : 'long' (stop sous le prix) ou 'short' (stop au-dessus).
        """
        zone_high, zone_low = self.stop_hunting_zone(symbol, recent_high, recent_low, atr)
        if zone_high is None:
            return stop_price
        if direction == "long" and zone_low is not None:
            # stop long dans la zone de chasse -> on le pousse PLUS BAS, hors
            # de la zone (un stop trop évident se fait « chasser »)
            if stop_price >= zone_low:
                return min(stop_price - 1.0 * atr, zone_low - 1.0 * atr)
        elif direction == "short" and zone_high is not None:
            # stop short dans la zone -> poussé PLUS HAUT, hors de la zone
            if stop_price <= zone_high:
                return max(stop_price + 1.0 * atr, zone_high + 1.0 * atr)
        return stop_price

    # ------------------------------------------------------------------ #
    # SANTÉ / TÉLÉMÉTRIE
    # ------------------------------------------------------------------ #
    def status(self, symbol: str) -> dict:
        delta, vol = self.get_delta(symbol)
        ofi = self.compute_ofi(symbol)
        return {
            "symbol": symbol,
            "delta": round(delta, 4),
            "volume_60s": round(vol, 4),
            "cvd": round(self.get_cvd(symbol), 4),
            "ofi": round(ofi, 4) if ofi is not None else None,
            "absorption": self.detect_absorption(symbol),
            "liquidation_cascade": self.liquidation_cascade_active(symbol),
            "toxicity_factor": round(self.toxicity_factor(symbol), 4),
            "n_trades": len(self.trades.get(symbol, ())),
        }
