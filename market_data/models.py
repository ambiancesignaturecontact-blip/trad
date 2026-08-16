import time
from market_data.quality import MarketDataQuality

class MarketTick:
    """
    Normalized Market Tick Data structure (Phase 7).
    Guarantees 100% genuine data tracking with origin source and receipt timestamps.
    """
    def __init__(self, symbol: str, exchange: str, timestamp: float, bid: float, ask: float, last: float, volume: float, source: str = "REST"):
        self.symbol = symbol
        self.exchange = exchange
        self.timestamp = timestamp
        self.received_at = time.time()
        self.source = source
        self.quality = MarketDataQuality.LIVE
        
        self.bid = float(bid) if bid is not None else None
        self.ask = float(ask) if ask is not None else None
        self.last = float(last) if last is not None else None
        self.volume = float(volume) if volume is not None else None
        self.sequence = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp,
            "received_at": self.received_at,
            "source": self.source,
            "quality": self.quality,
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "volume": self.volume
        }
