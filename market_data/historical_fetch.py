"""
Fetch OHLCV RÉELS (Binance -> Bybit -> Yahoo) extrait de main.py
(LOT C, F3). Corps inchangés ; jamais de barres fabriquées.
"""

import logging

import time

import httpx
import pandas as pd

from core.config import settings
from core.rate_limits import bybit_limiter, yahoo_limiter
from main import CRYPTO_SYMBOLS  # noqa: E402

logger = logging.getLogger("InstitutionalTradingBot")  # LOT C : même canal de logs que main

# AUDIT B6-1: short TTL cache for Yahoo chart calls (rate-limit friendly)
_yahoo_cache: dict = {}


async def fetch_yahoo_finance_candles(ticker: str, interval="1h", range_str="5d") -> pd.DataFrame:
    """
    Queries Yahoo Finance API with a secure browser User-Agent
    to fetch 100% genuine real-time and historical candles for Gold, Forex, and Stocks!
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={interval}&range={range_str}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    # AUDIT B6-1: serve fresh-enough cached bars instead of hammering Yahoo
    _cache_key = f"{ticker}|{interval}|{range_str}"
    _cached = _yahoo_cache.get(_cache_key)
    if _cached is not None and (time.time() - _cached[0]) < settings.get_float("data", "yahoo_cache_ttl_seconds", 20.0):
        return _cached[1].copy()

    try:
        async with yahoo_limiter:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            result = resp.json().get("chart", {}).get("result", [])[0]
            timestamps = result.get("timestamp", [])
            if timestamps is None:
                logger.info(f"Yahoo Finance: Market for {ticker} is currently closed or has no active trades (Weekend/Closed).")
                return pd.DataFrame()

            indicators = result.get("indicators", {}).get("quote", [])[0]

            opens = indicators.get("open", [])
            highs = indicators.get("high", [])
            lows = indicators.get("low", [])
            closes = indicators.get("close", [])
            volumes = indicators.get("volume", [])

            data = []
            for idx, t in enumerate(timestamps):
                if opens[idx] is not None and closes[idx] is not None:
                    data.append({
                        "timestamp": pd.to_datetime(t, unit='s'),
                        "open": float(opens[idx]),
                        "high": float(highs[idx]),
                        "low": float(lows[idx]),
                        "close": float(closes[idx]),
                        "volume": float(volumes[idx]) if volumes[idx] else 10.0
                    })
            df = pd.DataFrame(data).set_index("timestamp")
            _yahoo_cache[_cache_key] = (time.time(), df)
            if len(_yahoo_cache) > 64:
                _yahoo_cache.pop(next(iter(_yahoo_cache)))
            logger.info(f"Successfully loaded {len(df)} actual real-world market bars from Yahoo Finance for {ticker}!")
            return df
    except Exception as e:
        logger.error(f"Failed to fetch Yahoo Finance candles for {ticker}: {str(e)}")
    return pd.DataFrame()


def _klines_to_df(data: list) -> pd.DataFrame:
    """Convertit une réponse klines (Binance ou Bybit) en DataFrame OHLCV réel."""
    bars = []
    for b in data:
        bars.append({
            "timestamp": pd.to_datetime(b[0], unit='ms'),
            "open": float(b[1]),
            "high": float(b[2]),
            "low": float(b[3]),
            "close": float(b[4]),
            "volume": float(b[5])
        })
    df = pd.DataFrame(bars).set_index("timestamp")
    return df


async def fetch_bybit_klines(symbol: str, interval: str = "1h", limit: int = 120) -> pd.DataFrame:
    """Barres OHLCV RÉELLES via l'API publique Bybit v5 (secours Binance)."""
    try:
        async with bybit_limiter:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
                    f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}"
                    f"&interval={interval}&limit={limit}"
                )
        if resp.status_code == 200 and resp.json().get("retCode") == 0:
            rows = resp.json().get("result", {}).get("list", [])
            if rows:
                # Bybit renvoie les barres de la plus récente à la plus ancienne
                rows = list(reversed(rows))
                bars = []
                for b in rows:
                    bars.append({
                        "timestamp": pd.to_datetime(int(b[0]), unit='ms'),
                        "open": float(b[1]), "high": float(b[2]),
                        "low": float(b[3]), "close": float(b[4]),
                        "volume": float(b[5]),
                    })
                df = pd.DataFrame(bars).set_index("timestamp")
                logger.info(f"Fetched {len(df)} barres RÉELLES Bybit pour {symbol} ({interval}).")
                return df
    except Exception as e:
        logger.warning(f"Bybit klines failed for {symbol}: {e}")
    return pd.DataFrame()


async def fetch_historical_market_data(symbol="BTCUSDT"):
    """
    Fetches real historical price candles (OHLCV) from real APIs (Binance, puis
    Bybit en secours, puis Yahoo pour les actifs non-crypto). AUCUNE donnée
    simulée : si toutes les sources réelles échouent, renvoie un DataFrame vide
    (l'appelant marque l'actif UNAVAILABLE et ne trade pas).
    """
    # 1) Binance (source primaire)
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=120"
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(url)
        if response.status_code == 200:
            df = _klines_to_df(response.json())
            if not df.empty:
                logger.info(f"Successfully fetched {len(df)} real bars from Binance for {symbol}.")
                return df
    except Exception as e:
        logger.warning(f"Binance historical fetch failed for {symbol}: {e}")

    # 2) Bybit (secours réel pour les cryptos)
    if symbol in CRYPTO_SYMBOLS:
        df = await fetch_bybit_klines(symbol, interval="1h", limit=120)
        if not df.empty:
            return df

    # 3) Yahoo Finance (secours réel pour Or/FX/Actions — et cryptos en dernier recours)
    try:
        y_ticker = "GC=F" if symbol == "XAUUSD" else "EURUSD=X" if symbol == "EURUSD" else \
                   ("BTC-USD" if symbol == "BTCUSDT" else "ETH-USD" if symbol == "ETHUSDT"
                    else "SOL-USD" if symbol == "SOLUSDT" else symbol)
        df_y = await fetch_yahoo_finance_candles(y_ticker, interval="1h", range_str="5d")
        if not df_y.empty:
            logger.info(f"Fetched {len(df_y)} real bars from Yahoo Finance for {symbol}.")
            return df_y
    except Exception as e:
        logger.warning(f"Yahoo historical fetch failed for {symbol}: {e}")

    # HONNÊTETÉ (mentalité n°5) : aucune source réelle -> vide, pas de simulé.
    logger.warning(f"NO REAL HISTORICAL DATA AVAILABLE for {symbol} — marked UNAVAILABLE.")
    return pd.DataFrame()
