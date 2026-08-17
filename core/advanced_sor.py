"""
LOT 47+: Advanced Smart Order Router with Real Latency, Liquidity & Volume
"""

import asyncio
import time
import httpx
import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger("AdvancedSOR")

@dataclass
class VenueQuote:
    exchange: str
    price: float
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    fee_rate: float
    latency_ms: float
    net_cost_buy: float
    net_cost_sell: float
    liquidity_usd: float
    final_score: float = 0.0


class AdvancedSmartOrderRouter:
    def __init__(self):
        self.fee_rates = {"binance": 0.0004, "bybit": 0.0006}

    async def get_all_quotes(self, symbol: str) -> List[VenueQuote]:
        tasks = [self._fetch_binance(symbol), self._fetch_bybit(symbol)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, VenueQuote)]

    async def _fetch_binance(self, symbol: str) -> Optional[VenueQuote]:
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=3.5) as client:
                price_resp = await client.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}")
                depth_resp = await client.get(f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=10")

                price = float(price_resp.json()["price"])
                depth = depth_resp.json()

                bid = float(depth["bids"][0][0])
                ask = float(depth["asks"][0][0])
                bid_qty = float(depth["bids"][0][1])
                ask_qty = float(depth["asks"][0][1])

                latency = (time.time() - start) * 1000
                fee = self.fee_rates["binance"]

                return VenueQuote(
                    exchange="binance",
                    price=price,
                    bid=bid, ask=ask,
                    bid_qty=bid_qty, ask_qty=ask_qty,
                    fee_rate=fee,
                    latency_ms=latency,
                    net_cost_buy=ask * (1 + fee),
                    net_cost_sell=bid * (1 - fee),
                    liquidity_usd=min(bid_qty, ask_qty) * price
                )
        except Exception as e:
            logger.warning(f"Binance SOR error: {e}")
            return None

    async def _fetch_bybit(self, symbol: str) -> Optional[VenueQuote]:
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=3.5) as client:
                resp = await client.get(f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}")
                data = resp.json().get("result", {}).get("list", [{}])[0]

                price = float(data.get("lastPrice", 0))
                bid = float(data.get("bid1Price", price))
                ask = float(data.get("ask1Price", price))
                bid_qty = float(data.get("bid1Size", 0))
                ask_qty = float(data.get("ask1Size", 0))

                latency = (time.time() - start) * 1000
                fee = self.fee_rates["bybit"]

                return VenueQuote(
                    exchange="bybit",
                    price=price,
                    bid=bid, ask=ask,
                    bid_qty=bid_qty, ask_qty=ask_qty,
                    fee_rate=fee,
                    latency_ms=latency,
                    net_cost_buy=ask * (1 + fee),
                    net_cost_sell=bid * (1 - fee),
                    liquidity_usd=min(bid_qty, ask_qty) * price
                )
        except Exception as e:
            logger.warning(f"Bybit SOR error: {e}")
            return None

    def select_best_venue(self, quotes: List[VenueQuote], side: str) -> Optional[VenueQuote]:
        if not quotes:
            return None

        for q in quotes:
            cost_score = 1 / q.net_cost_buy if side == "BUY" else 1 / q.net_cost_sell
            latency_score = 1 / (q.latency_ms + 10)
            liquidity_score = min(q.liquidity_usd / 50000, 1.0)

            q.final_score = (cost_score * 0.55) + (latency_score * 0.25) + (liquidity_score * 0.20)

        best = max(quotes, key=lambda x: x.final_score)
        logger.info(f"[SOR] Best venue: {best.exchange} | Score: {best.final_score:.4f}")
        return best
