"""
Execution router (VISION §3.1) and per-venue slippage model (VISION §3.2, §7).

- decide_style(): market vs limit vs TWAP based on urgency, spread and size
- ExecutionAlpha: tracks realized-vs-arrival slippage to measure execution quality
- SlippageModel: per-venue slippage statistics, recalibrated daily from real fills
"""
import logging
import time
from collections import defaultdict
from typing import Dict, Optional

logger = logging.getLogger("ExecutionRouter")


def decide_style(spread_bps: float, urgency: float, size_vs_liquidity: float,
                 twap_max_size_ratio: float = 0.02) -> str:
    """
    Returns "market" | "limit" | "twap".
    - market   : tight spread + high urgency (signal is strong/decaying)
    - limit    : wide spread + low urgency (patience pays)
    - twap     : large size vs available liquidity -> slice over time
    """
    if size_vs_liquidity > twap_max_size_ratio:
        return "twap"
    if spread_bps > 15.0 and urgency < 0.5:
        return "limit"
    return "market"


class ExecutionAlpha:
    """Measures execution quality: realized price vs arrival price (VISION §3.1)."""

    def __init__(self):
        self.samples: list = []

    def record(self, symbol: str, side: str, arrival_price: float,
               realized_price: float, style: str) -> float:
        """Returns slippage in bps (positive = adverse)."""
        if arrival_price <= 0 or realized_price <= 0:
            return 0.0
        if side == "BUY":
            slip = (realized_price - arrival_price) / arrival_price * 1e4
        else:
            slip = (arrival_price - realized_price) / arrival_price * 1e4
        self.samples.append({
            "symbol": symbol, "side": side, "style": style,
            "slippage_bps": round(slip, 3), "ts": time.time(),
        })
        if len(self.samples) > 2000:
            self.samples = self.samples[-2000:]
        return float(slip)

    def avg_slippage_bps(self, style: Optional[str] = None) -> float:
        pool = [s["slippage_bps"] for s in self.samples if (style is None or s["style"] == style)]
        return float(sum(pool) / len(pool)) if pool else 0.0


class SlippageModel:
    """Per-venue/per-symbol slippage stats, recalibrated from real fills (VISION §3.2)."""

    def __init__(self):
        self.stats: Dict[str, dict] = defaultdict(lambda: {"n": 0, "sum_bps": 0.0, "last": 0.0})

    def update(self, venue: str, symbol: str, slippage_bps: float) -> None:
        key = f"{venue}:{symbol}"
        s = self.stats[key]
        s["n"] += 1
        s["sum_bps"] += slippage_bps
        s["last"] = slippage_bps
        s["avg_bps"] = s["sum_bps"] / s["n"]

    def expected_slippage_bps(self, venue: str, symbol: str, fallback: float = 3.0) -> float:
        key = f"{venue}:{symbol}"
        s = self.stats.get(key)
        return s["avg_bps"] if s and s["n"] >= 5 else fallback

    def recalibrate(self) -> None:
        """Daily recalibration hook (called by the autonomous scheduler)."""
        total = sum(v["n"] for v in self.stats.values())
        logger.info(f"Slippage model: {total} real fill samples across {len(self.stats)} venue/symbol keys")
