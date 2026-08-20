import logging
import time

import httpx

from market_data.base import MarketDataProvider
from market_data.models import MarketTick

logger = logging.getLogger("BybitDataProvider")

class BybitDataProvider(MarketDataProvider):
    """
    Bybit Market Data Adapter querying real public CEX endpoints (Phase 6).
    """
    def __init__(self):
        self.base_url = "https://api.bybit.com"

    def get_ticker(self, symbol: str) -> MarketTick:
        url = f"{self.base_url}/v5/market/tickers?category=spot&symbol={symbol}"
        try:
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code == 200:
                list_data = resp.json().get("result", {}).get("list", [])
                if list_data:
                    data = list_data[0]
                    last = float(data.get("lastPrice"))
                    bid = float(data.get("bid1Price", last * 0.9995))
                    ask = float(data.get("ask1Price", last * 1.0005))
                    volume = float(data.get("volume24h", 1.0))

                    tick = MarketTick(
                        symbol=symbol,
                        exchange="Bybit",
                        timestamp=time.time(),
                        bid=bid,
                        ask=ask,
                        last=last,
                        volume=volume
                    )
                    return tick
        except Exception as e:
            logger.error(f"BybitDataProvider: Failed to fetch ticker for {symbol}: {str(e)}")

        # Return none if API is down - STRICTLY NO FAKE FALLBACKS
        return None
