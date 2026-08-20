import logging
import time

import httpx

from market_data.base import MarketDataProvider
from market_data.models import MarketTick

logger = logging.getLogger("BinanceDataProvider")

class BinanceDataProvider(MarketDataProvider):
    """
    Binance Market Data Adapter querying real public CEX endpoints (Phase 6).
    """
    def __init__(self):
        self.base_url = "https://api.binance.com"

    def get_ticker(self, symbol: str) -> MarketTick:
        # Binance symbol formatting (e.g. BTCUSDT)
        url = f"{self.base_url}/api/v3/ticker/bookTicker?symbol={symbol}"
        try:
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                bid = float(data.get("bidPrice"))
                ask = float(data.get("askPrice"))
                last = (bid + ask) / 2.0 # Mid-price proxy

                # Create and return validated MarketTick
                tick = MarketTick(
                    symbol=symbol,
                    exchange="Binance",
                    timestamp=time.time(),
                    bid=bid,
                    ask=ask,
                    last=last,
                    volume=float(data.get("bidQty", 1.0)) # actual real liquidity volume
                )
                return tick
        except Exception as e:
            logger.error(f"BinanceDataProvider: Failed to fetch ticker for {symbol}: {str(e)}")

        # Return none if API is down - STRICTLY NO FAKE FALLBACKS (Phase 34)
        return None
