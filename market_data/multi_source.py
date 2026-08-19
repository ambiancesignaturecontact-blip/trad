"""
Multi-Source Price Consensus Engine — DONNÉES 100% RÉELLES, MULTI-EXCHANGE.

PDF (section « Redondance & fiabilité ») : chaque catégorie critique (prix,
funding, OI) doit avoir AU MOINS 2 sources indépendantes CROISÉES. Une
divergence anormale entre sources = alarme + GEL du trading (données non
fiables). Mentalité n°5 : la confiance dans le signal compte autant que le
signal — on ne trade JAMAIS sur un prix non corroboré.

Sources prix crypto (rapides, ~chaque tick) :
    Binance REST, Bybit REST, Coinbase, Kraken, OKX
Sources prix crypto (lentes, batch 60s) :
    CoinGecko, CryptoCompare
Sources prix Or/FX/Actions :
    Yahoo Finance (toutes) + gold-api.com (XAUUSD) + open.er-api.com (EURUSD)
    AAPL/TSLA : Yahoo seule -> statut SINGLE_SOURCE honnête (pas de 2e source
    gratuite fiable, donc on le DIT au lieu d'inventer).

Consensus : MÉDIANE robuste (les outliers sont écartés par construction).
Divergence : (max - min) / médiane × 100. Si > seuil par actif -> DIVERGENT.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("MultiSourcePrice")

# Seuils de divergence par actif (en %) — au-delà, GEL du trading
# NOTE XAUUSD : Yahoo GC=F est le CONTRAT FUTURES or (COMEX), gold-api est le
# SPOT XAU. Un écart de ~1% entre futures et spot est structurel et normal
# (base), donc seuil élargi à 1.5% avec cette justification documentée.
DIVERGENCE_THRESHOLDS = {
    "BTCUSDT": 0.30,
    "ETHUSDT": 0.50,
    "SOLUSDT": 0.75,
    "XAUUSD": 1.50,
    "EURUSD": 0.50,
    "AAPL": 1.00,
    "TSLA": 1.00,
}
DEFAULT_THRESHOLD_PCT = 1.00

# Mapping des identifiants par source
_COINBASE_SYM = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD"}
_KRAKEN_SYM = {"BTCUSDT": "XBTUSD", "ETHUSDT": "ETHUSD", "SOLUSDT": "SOLUSD"}
_OKX_SYM = {"BTCUSDT": "BTC-USDT", "ETHUSDT": "ETH-USDT", "SOLUSDT": "SOL-USDT"}
_COINGECKO_ID = {"BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "SOLUSDT": "solana"}
_CC_SYM = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL"}
_YAHOO_SYM = {"XAUUSD": "GC=F", "EURUSD": "EURUSD=X", "AAPL": "AAPL", "TSLA": "TSLA"}


class MultiSourcePriceEngine:
    """Moteur de consensus de prix multi-sources (asyncio, non bloquant)."""

    def __init__(self):
        self._cache: Dict[str, dict] = {}        # symbol -> consensus récent
        self._slow_cache: Dict[str, dict] = {}   # symbol -> prix sources lentes
        self._funding_cache: Dict[str, dict] = {}
        self._yahoo_cache: Dict[str, Tuple[float, dict]] = {}

    # ------------------------------------------------------------------ #
    # HELPERS
    # ------------------------------------------------------------------ #
    @staticmethod
    def threshold(symbol: str) -> float:
        return DIVERGENCE_THRESHOLDS.get(symbol, DEFAULT_THRESHOLD_PCT)

    async def _safe_fetch(self, name: str, coro):
        """Exécute un fetch source avec garde-fou (jamais bloquant)."""
        try:
            return name, await asyncio.wait_for(coro, timeout=5.0)
        except Exception as e:
            logger.debug(f"MultiSource: {name} échec: {e}")
            return name, None

    # ------------------------------------------------------------------ #
    # FETCH PAR SOURCE (chacune renvoie un float ou None)
    # ------------------------------------------------------------------ #
    async def _fetch_binance(self, symbol: str):
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}")
            if r.status_code == 200:
                return float(r.json()["price"])
        except Exception:
            pass
        return None

    async def _fetch_bybit(self, symbol: str):
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(
                    f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}")
            if r.status_code == 200:
                t = r.json().get("result", {}).get("list", [{}])[0]
                return float(t.get("lastPrice") or 0.0) or None
        except Exception:
            pass
        return None

    async def _fetch_coinbase(self, symbol: str):
        sym = _COINBASE_SYM.get(symbol)
        if not sym:
            return None
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"https://api.coinbase.com/v2/prices/{sym}/spot")
            if r.status_code == 200:
                return float(r.json()["data"]["amount"])
        except Exception:
            pass
        return None

    async def _fetch_kraken(self, symbol: str):
        pair = _KRAKEN_SYM.get(symbol)
        if not pair:
            return None
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"https://api.kraken.com/0/public/Ticker?pair={pair}")
            if r.status_code == 200:
                result = r.json().get("result", {})
                for v in result.values():
                    if isinstance(v, dict) and v.get("c"):
                        return float(v["c"][0])
        except Exception:
            pass
        return None

    async def _fetch_okx(self, symbol: str):
        inst = _OKX_SYM.get(symbol)
        if not inst:
            return None
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"https://www.okx.com/api/v5/market/ticker?instId={inst}")
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    return float(data[0].get("last") or 0.0) or None
        except Exception:
            pass
        return None

    async def _fetch_coingecko(self, symbol: str):
        cid = _COINGECKO_ID.get(symbol)
        if not cid:
            return None
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(
                    f"https://api.coingecko.com/api/v3/simple/price?ids={cid}&vs_currencies=usd")
            if r.status_code == 200:
                price = r.json().get(cid, {}).get("usd")
                return float(price) if price else None
        except Exception:
            pass
        return None

    async def _fetch_cryptocompare(self, symbol: str):
        fsym = _CC_SYM.get(symbol)
        if not fsym:
            return None
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(
                    f"https://min-api.cryptocompare.com/data/price?fsym={fsym}&tsyms=USD")
            if r.status_code == 200:
                price = r.json().get("USD")
                return float(price) if price else None
        except Exception:
            pass
        return None

    async def _fetch_yahoo(self, symbol: str):
        """Prix Yahoo Finance (1m, cache TTL 20s) — utilisé pour tous les actifs."""
        y_ticker = _YAHOO_SYM.get(symbol)
        if not y_ticker:
            return None
        now = time.time()
        cached = self._yahoo_cache.get(symbol)
        if cached and now - cached[0] < 20.0:
            return cached[1]
        try:
            url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{y_ticker}"
                   f"?interval=1m&range=1d")
            headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                                      "Chrome/120.0.0.0 Safari/537.36")}
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(url, headers=headers)
            if r.status_code == 200:
                result = r.json().get("chart", {}).get("result", [])
                if not result:
                    return None
                closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
                closes = [x for x in closes if x is not None]
                if closes:
                    price = float(closes[-1])
                    self._yahoo_cache[symbol] = (now, price)
                    return price
        except Exception:
            pass
        return None

    async def _fetch_goldapi(self, symbol: str):
        if symbol != "XAUUSD":
            return None
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get("https://api.gold-api.com/price/XAU")
            if r.status_code == 200:
                price = r.json().get("price")
                return float(price) if price else None
        except Exception:
            pass
        return None

    async def _fetch_erapi(self, symbol: str):
        """open.er-api.com : taux EUR/USD réel (gratuit, sans clé).

        L'API renvoie rates.EUR = quantité d'EUR pour 1 USD (≈0.86).
        Le prix EURUSD = 1 / rates.EUR (≈1.16). Conversion obligatoire.
        """
        if symbol != "EURUSD":
            return None
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get("https://open.er-api.com/v6/latest/USD")
            if r.status_code == 200:
                rate = r.json().get("rates", {}).get("EUR")
                if rate and float(rate) > 0:
                    return 1.0 / float(rate)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    # SOURCES PAR ACTIF
    # ------------------------------------------------------------------ #
    def _source_coros(self, symbol: str) -> List[Tuple[str, object]]:
        """Liste (nom, coroutine) des sources RAPIDES pour un symbole."""
        if symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            return [
                ("binance", self._fetch_binance(symbol)),
                ("bybit", self._fetch_bybit(symbol)),
                ("coinbase", self._fetch_coinbase(symbol)),
                ("kraken", self._fetch_kraken(symbol)),
                ("okx", self._fetch_okx(symbol)),
            ]
        return [
            ("yahoo", self._fetch_yahoo(symbol)),
            ("gold-api", self._fetch_goldapi(symbol)),
            ("er-api", self._fetch_erapi(symbol)),
        ]

    def _slow_source_coros(self, symbol: str) -> List[Tuple[str, object]]:
        if symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            return [
                ("coingecko", self._fetch_coingecko(symbol)),
                ("cryptocompare", self._fetch_cryptocompare(symbol)),
            ]
        return []

    # ------------------------------------------------------------------ #
    # CONSENSUS
    # ------------------------------------------------------------------ #
    async def get_consensus(self, symbol: str, max_age_seconds: float = 10.0) -> dict:
        """Retourne le consensus (cache court), refresh si nécessaire."""
        cached = self._cache.get(symbol)
        if cached and time.time() - cached["ts"] < max_age_seconds:
            return cached["data"]
        data = await self.refresh(symbol)
        return data

    async def refresh(self, symbol: str) -> dict:
        """Collecte toutes les sources en parallèle et calcule le consensus."""
        fast = self._source_coros(symbol)
        tasks = [self._safe_fetch(name, coro) for name, coro in fast]

        # Sources lentes (CoinGecko/CryptoCompare) : rafraîchies au plus toutes
        # les 60s — on ne crée les coroutines que si nécessaire (évite les
        # coroutines jamais awaited quand le cache est frais).
        sc = self._slow_cache.get(symbol)
        need_slow = not sc or time.time() - sc.get("ts", 0.0) > 60.0
        if need_slow:
            for name, coro in self._slow_source_coros(symbol):
                tasks.append(self._safe_fetch(name, coro))

        results = await asyncio.gather(*tasks)

        prices: Dict[str, float] = {}
        for name, price in results:
            if price is not None and float(price) > 0:
                prices[name] = float(price)

        # Conserver les sources lentes précédentes si pas rafraîchies
        if sc:
            for name, price in sc.get("sources", {}).items():
                prices.setdefault(name, price)
        new_slow = {k: v for k, v in prices.items()
                    if k in ("coingecko", "cryptocompare")}
        if new_slow:
            self._slow_cache[symbol] = {"sources": new_slow, "ts": time.time()}

        data = self._compute_consensus(symbol, prices)
        self._cache[symbol] = {"data": data, "ts": time.time()}
        return data

    @staticmethod
    def _compute_consensus(symbol: str, prices: Dict[str, float]) -> dict:
        n = len(prices)
        now = time.time()
        threshold = MultiSourcePriceEngine.threshold(symbol)
        if n == 0:
            return {
                "symbol": symbol, "price": None, "sources": {},
                "n_sources": 0, "divergence_pct": None,
                "status": "UNAVAILABLE", "threshold_pct": threshold, "ts": now,
            }
        vals = sorted(prices.values())
        median = vals[len(vals) // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
        divergence = ((vals[-1] - vals[0]) / median * 100.0) if median > 0 else 0.0

        if n == 1:
            status = "SINGLE_SOURCE"
        elif divergence > threshold:
            status = "DIVERGENT"
        else:
            status = "OK"

        return {
            "symbol": symbol,
            "price": round(median, 8),
            "sources": {k: round(v, 8) for k, v in prices.items()},
            "n_sources": n,
            "divergence_pct": round(divergence, 4),
            "status": status,
            "threshold_pct": threshold,
            "ts": now,
        }

    # ------------------------------------------------------------------ #
    # FUNDING RATE CONSENSUS (Binance <-> Bybit, PDF : 2 sources croisées)
    # ------------------------------------------------------------------ #
    async def _fetch_binance_funding(self, symbol: str):
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}")
            if r.status_code == 200:
                fr = r.json().get("lastFundingRate")
                return float(fr) if fr is not None else None
        except Exception:
            pass
        return None

    async def _fetch_bybit_funding(self, symbol: str):
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(
                    f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}")
            if r.status_code == 200:
                t = r.json().get("result", {}).get("list", [{}])[0]
                fr = t.get("fundingRate")
                return float(fr) if fr is not None else None
        except Exception:
            pass
        return None

    async def get_funding_consensus(self, symbol: str, max_age_seconds: float = 30.0) -> dict:
        """Funding 8h croisé Binance + Bybit (médiane si OK, None si divergent)."""
        cached = self._funding_cache.get(symbol)
        if cached and time.time() - cached["ts"] < max_age_seconds:
            return cached["data"]

        b, y = await asyncio.gather(
            self._safe_fetch("binance_funding", self._fetch_binance_funding(symbol)),
            self._safe_fetch("bybit_funding", self._fetch_bybit_funding(symbol)),
        )
        rates = {}
        if b[1] is not None:
            rates["binance"] = b[1]
        if y[1] is not None:
            rates["bybit"] = y[1]

        if not rates:
            data = {"funding_rate_8h": None, "sources": {}, "status": "UNAVAILABLE",
                    "ts": time.time()}
        elif len(rates) == 1:
            data = {"funding_rate_8h": list(rates.values())[0], "sources": rates,
                    "status": "SINGLE_SOURCE", "ts": time.time()}
        else:
            a, bb = rates["binance"], rates["bybit"]
            # FIX (logs prod) : seuil réaliste — voir docstring. Un écart de
            # 2-3x entre les taux des deux venues est NORMAL (calculs/périodes
            # différents) ; seule une vraie anomalie bloque le consensus.
            signs_opposed = (a > 0 > bb) or (bb > 0 > a)
            abs_gap = abs(a - bb)
            if signs_opposed and abs_gap > 1e-6:
                data = {"funding_rate_8h": None, "sources": rates,
                        "status": "DIVERGENT", "ts": time.time()}
            elif abs_gap > 0.0001:  # écart > 1 bp : anomalie réelle
                data = {"funding_rate_8h": None, "sources": rates,
                        "status": "DIVERGENT", "ts": time.time()}
            else:
                data = {"funding_rate_8h": (a + bb) / 2.0, "sources": rates,
                        "status": "OK", "ts": time.time()}
        self._funding_cache[symbol] = {"data": data, "ts": time.time()}
        return data
