import logging
import time

from market_data.models import MarketTick
from market_data.quality import MarketDataQuality

logger = logging.getLogger("DataQualityGate")

class MarketDataProvider:
    """
    Unified Abstract Interface representing Market Data Providers (Phase 4 & 23).
    """
    def get_ticker(self, symbol: str) -> MarketTick:
        raise NotImplementedError

    def get_order_book(self, symbol: str) -> dict:
        raise NotImplementedError

    def get_trades(self, symbol: str) -> list:
        raise NotImplementedError

    def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> list:
        raise NotImplementedError


class DataQualityGate:
    """
    Sovereign Data Quality Validator (Phase 2 & Lot 2).
    Enforces strict freshness and quality bounds before allowing trading actions.
    """
    def __init__(self, stale_timeout_seconds=15.0):
        self.stale_timeout_seconds = stale_timeout_seconds

    def validate_tick(self, tick: MarketTick, mode: str) -> bool:
        """
        Validates a single market tick against strict freshness and mode boundaries.
        In REAL mode, only 100% 'LIVE' quality is allowed.
        """
        if not tick or tick.last is None:
            logger.error(f"DATA_GATE: Rejected null tick for {getattr(tick, 'symbol', 'Unknown')}.")
            return False

        current_epoch = time.time()

        # 1. Freshness Validation (Stale Timeout)
        # Check time elapsed since receipt or tick timestamp
        elapsed_since_receipt = current_epoch - tick.received_at
        if elapsed_since_receipt > self.stale_timeout_seconds:
            tick.quality = MarketDataQuality.STALE
            logger.error(f"DATA_GATE: Rejected {tick.symbol} tick. Data is STALE (Elapsed: {elapsed_since_receipt:.1f}s).")
            return False

        # 2. Quality Validation based on Trading Mode
        if mode == "REAL":
            # For REAL trading: strictly ONLY 'LIVE' quality is permitted!
            if tick.quality != MarketDataQuality.LIVE:
                logger.error(f"DATA_GATE: Real trading rejected. Data quality is {tick.quality} (Only LIVE allowed).")
                return False
        else:
            # In DEMO/PAPER/BACKTEST modes, we allow 'LIVE' or 'DELAYED'
            if tick.quality not in [MarketDataQuality.LIVE, MarketDataQuality.DELAYED]:
                logger.error(f"DATA_GATE: Rejected {tick.symbol} tick. Quality is {tick.quality}.")
                return False

        return True
