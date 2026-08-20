"""
Market Data Module - Extrait de live_trading_loop
Gère la récupération des prix en temps réel (Bybit + Yahoo)
"""
import asyncio
import logging

import httpx

logger = logging.getLogger("MarketData")

class MarketDataFetcher:
    def __init__(self, state: dict):
        self.state = state

    async def fetch_crypto_price(self, symbol: str) -> float | None:
        """Récupère le prix crypto via Bybit (rapide et fiable)"""
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.get(
                    f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}"
                )
                if resp.status_code == 200:
                    price = float(resp.json()["result"]["list"][0]["lastPrice"])
                    self.state["assets"][symbol]["price"] = price
                    if symbol == "BTCUSDT":
                        self.state["last_price"] = price
                    return price
        except Exception as e:
            logger.warning(f"Bybit price fetch failed for {symbol}: {e}")
        return None

    async def fetch_yahoo_price(self, symbol: str) -> float | None:
        """Récupère les prix Yahoo Finance (Gold, Forex, Stocks)"""
        ticker_map = {
            "XAUUSD": "GC=F",
            "EURUSD": "EURUSD=X",
            "AAPL": "AAPL",
            "TSLA": "TSLA"
        }
        y_ticker = ticker_map.get(symbol, symbol)

        try:
            from main import fetch_yahoo_finance_candles
            df = await fetch_yahoo_finance_candles(y_ticker, interval="1m", range_str="1d")
            if not df.empty:
                price = float(df['close'].iloc[-1])
                self.state["assets"][symbol]["price"] = price
                return price
        except Exception as e:
            logger.warning(f"Yahoo fetch failed for {symbol}: {e}")
        return None

    async def update_all_prices(self):
        """Met à jour les prix de tous les actifs"""
        tasks = []
        for symbol in self.state["assets"]:
            if symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
                tasks.append(self.fetch_crypto_price(symbol))
            else:
                tasks.append(self.fetch_yahoo_price(symbol))

        await asyncio.gather(*tasks, return_exceptions=True)
