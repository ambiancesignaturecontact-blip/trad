"""
LOT 47++: Complete Multi-Exchange Smart Order Router
Real latency + Real liquidity (order book depth) + Volume + Fees
"""

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger("MultiExchangeSOR")

@dataclass
class ExchangeQuote:
    exchange: str
    price: float
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    fee_rate: float
    latency_ms: float
    liquidity_usd: float
    net_cost_buy: float
    net_cost_sell: float
    final_score: float = 0.0


class MultiExchangeSmartOrderRouter:
    """
    Complete Multi-Exchange SOR with real metrics.
    Supports: Binance, Bybit (easily extensible)
    """

    def __init__(self):
        self.exchanges = ["binance", "bybit"]
        self.fee_rates = {
            "binance": 0.0004,
            "bybit": 0.0006
        }
        self.timeout = 4.0

    async def get_all_quotes(self, symbol: str) -> list[ExchangeQuote]:
        """Fetch real-time quotes from all exchanges"""
        tasks = [
            self._fetch_binance(symbol),
            self._fetch_bybit(symbol)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, ExchangeQuote)]

    async def _fetch_binance(self, symbol: str) -> ExchangeQuote | None:
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Price
                price_resp = await client.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}")
                price = float(price_resp.json()["price"])

                # Depth
                depth_resp = await client.get(f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=10")
                depth = depth_resp.json()

                bid = float(depth["bids"][0][0])
                ask = float(depth["asks"][0][0])
                bid_qty = float(depth["bids"][0][1])
                ask_qty = float(depth["asks"][0][1])

                latency = (time.time() - start) * 1000
                fee = self.fee_rates["binance"]
                liquidity = min(bid_qty, ask_qty) * price

                return ExchangeQuote(
                    exchange="binance",
                    price=price,
                    bid=bid,
                    ask=ask,
                    bid_qty=bid_qty,
                    ask_qty=ask_qty,
                    fee_rate=fee,
                    latency_ms=latency,
                    liquidity_usd=liquidity,
                    net_cost_buy=ask * (1 + fee),
                    net_cost_sell=bid * (1 - fee)
                )
        except Exception as e:
            logger.warning(f"Binance fetch failed: {e}")
            return None

    async def _fetch_bybit(self, symbol: str) -> ExchangeQuote | None:
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}")
                data = resp.json().get("result", {}).get("list", [{}])[0]

                price = float(data.get("lastPrice", 0))
                bid = float(data.get("bid1Price", price))
                ask = float(data.get("ask1Price", price))
                bid_qty = float(data.get("bid1Size", 0))
                ask_qty = float(data.get("ask1Size", 0))

                latency = (time.time() - start) * 1000
                fee = self.fee_rates["bybit"]
                liquidity = min(bid_qty, ask_qty) * price

                return ExchangeQuote(
                    exchange="bybit",
                    price=price,
                    bid=bid,
                    ask=ask,
                    bid_qty=bid_qty,
                    ask_qty=ask_qty,
                    fee_rate=fee,
                    latency_ms=latency,
                    liquidity_usd=liquidity,
                    net_cost_buy=ask * (1 + fee),
                    net_cost_sell=bid * (1 - fee)
                )
        except Exception as e:
            logger.warning(f"Bybit fetch failed: {e}")
            return None

    def select_best_venue(self, quotes: list[ExchangeQuote], side: str) -> ExchangeQuote | None:
        """Select the best exchange based on multiple real metrics"""
        if not quotes:
            return None

        for q in quotes:
            # Multi-factor scoring
            # AUDIT: for SELL the best venue is the one with the HIGHEST net proceeds,
            # so score with net_cost_sell directly (previous 1/net_cost_sell inverted
            # the ranking and picked the worst venue for sells).
            cost_score = 1 / q.net_cost_buy if side == "BUY" else q.net_cost_sell
            latency_score = 1 / (q.latency_ms + 15)
            liquidity_score = min(q.liquidity_usd / 100000, 1.0)

            # Weighted scoring (Price is most important)
            q.final_score = (
                cost_score * 0.50 +
                latency_score * 0.25 +
                liquidity_score * 0.25
            )

        best = max(quotes, key=lambda x: x.final_score)
        logger.info(f"[SOR] Best venue: {best.exchange} | Score: {best.final_score:.4f} | Liquidity: ${best.liquidity_usd:,.0f}")
        return best

    async def route_order(self, symbol: str, side: str, quantity: float) -> dict:
        """Main routing function"""
        quotes = await self.get_all_quotes(symbol)
        best = self.select_best_venue(quotes, side)

        if not best:
            return {"success": False, "reason": "No venue available"}

        return {
            "success": True,
            "exchange": best.exchange,
            "price": best.price,
            "estimated_cost": best.net_cost_buy if side == "BUY" else best.net_cost_sell,
            "latency_ms": best.latency_ms,
            "liquidity_usd": best.liquidity_usd
        }
