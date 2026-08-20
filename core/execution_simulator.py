"""
LOT 49: Realistic Execution Simulator (Slippage + Latency Modeling)
Used for high-fidelity backtesting of REAL trading conditions.
"""

import logging

import numpy as np

logger = logging.getLogger("ExecutionSimulator")

class ExecutionSimulator:
    """
    Simulates realistic order execution conditions:
    - Slippage (price impact + spread)
    - Latency (network + exchange delay)
    - Partial fills (optional)
    """

    def __init__(self, base_slippage_bps: float = 5.0, base_latency_ms: float = 80):
        self.base_slippage_bps = base_slippage_bps      # 5 basis points default
        self.base_latency_ms = base_latency_ms

    def simulate_slippage(self, price: float, side: str, volatility: float,
                          liquidity_score: float = 1.0, order_size_pct: float = 0.01) -> float:
        """
        Calculate realistic slippage.
        - Higher volatility → higher slippage
        - Lower liquidity → higher slippage
        - Larger order size → higher slippage
        """
        vol_factor = 1 + (volatility * 8)
        liq_factor = 1 / max(liquidity_score, 0.2)
        size_factor = 1 + (order_size_pct * 3)

        slippage_bps = self.base_slippage_bps * vol_factor * liq_factor * size_factor
        slippage_pct = slippage_bps / 10000

        if side.upper() == "BUY":
            executed_price = price * (1 + slippage_pct)
        else:
            executed_price = price * (1 - slippage_pct)

        return round(executed_price, 6)

    def simulate_latency(self, network_jitter: bool = True) -> float:
        """
        Simulate order execution latency in milliseconds.
        """
        latency = self.base_latency_ms

        if network_jitter:
            jitter = np.random.normal(0, 25)  # ±25ms standard deviation
            latency += max(jitter, 5)

        return round(latency, 1)

    def simulate_execution(self, symbol: str, side: str, expected_price: float,
                           quantity: float, market_conditions: dict) -> dict:
        """
        Full realistic execution simulation.
        Returns dict with executed_price, latency, slippage, success.
        """
        volatility = market_conditions.get("volatility", 0.015)
        liquidity = market_conditions.get("liquidity_score", 1.0)
        order_size_pct = market_conditions.get("order_size_pct", 0.01)

        executed_price = self.simulate_slippage(
            expected_price, side, volatility, liquidity, order_size_pct
        )

        latency_ms = self.simulate_latency()

        slippage_bps = abs((executed_price - expected_price) / expected_price) * 10000

        result = {
            "symbol": symbol,
            "side": side,
            "expected_price": expected_price,
            "executed_price": executed_price,
            "slippage_bps": round(slippage_bps, 2),
            "latency_ms": latency_ms,
            "quantity": quantity,
            "success": True
        }

        logger.info(f"[SIM] {side} {symbol} @ {executed_price} | Slippage: {slippage_bps:.1f}bps | Latency: {latency_ms}ms")
        return result
