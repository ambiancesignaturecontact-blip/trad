import logging

logger = logging.getLogger("DexCexArbitrage")

class DexCexArbitrageEngine:
    """
    Cross-Venue DEX-CEX Arbitrage Engine (Phase 6 & Lot 4).
    Calculates gross spreads, strict exchange fees, order book depth slippages,
    and net executable spreads on real-world order book feeds.
    """
    def __init__(self, min_profit_spread_pct=0.003):
        self.min_profit_spread_pct = min_profit_spread_pct

    def calculate_executable_arbitrage(self, symbol: str, book_a: dict, book_b: dict, fee_a: float, fee_b: float, max_order_usd: float = 1000.0) -> dict:
        """
        Executes order-book walking to calculate actual gross spreads, exact slippages,
        and net executable profits.

        book_a: dict with 'bids' and 'asks', each list of [price, qty]
        book_b: dict with 'bids' and 'asks', each list of [price, qty]
        """
        # Ensure both order books are fully populated and valid
        if not book_a or not book_b or not book_a.get("bids") or not book_b.get("bids"):
            return {"action": "HOLD", "reason": "One or both exchange order books are missing."}

        bids_a = book_a["bids"]
        asks_a = book_a["asks"]
        bids_b = book_b["bids"]
        asks_b = book_b["asks"]

        # Route 1: Buy Exchange A (asks_a), Sell Exchange B (bids_b)
        # We walk down the books to fill up to max_order_usd
        qty_a, avg_price_a = self._walk_book_to_fill(asks_a, max_order_usd)
        qty_b, avg_price_b = self._walk_book_to_fill(bids_b, max_order_usd)

        # Ensure we have valid fills on both sides
        if qty_a == 0 or qty_b == 0:
            return {"action": "HOLD", "reason": "Insufficient order book depth to execute minimum lot."}

        # Actual size is limited by the smallest execution size to remain delta-neutral!
        executable_qty = min(qty_a, qty_b)

        # Gross Spread: Buy at A, Sell at B
        gross_spread_pct = (avg_price_b - avg_price_a) / avg_price_a

        # Deduct total trading fees (Exchange A + Exchange B)
        total_fees = fee_a + fee_b

        # Net Spread after slippage and fees
        net_spread_pct = gross_spread_pct - total_fees

        if net_spread_pct >= self.min_profit_spread_pct:
            logger.info(
                f"🏆 ARBITRAGE OPPORTUNITY FOUND ({symbol}): Buy A (${avg_price_a:.2f}) -> Sell B (${avg_price_b:.2f}). "
                f"Executable Qty: {executable_qty:.4f}. Net Profit: {net_spread_pct*100:.3f}%"
            )
            return {
                "action": "EXECUTE_ARBITRAGE",
                "symbol": symbol,
                "route": "BUY_A_SELL_B",
                "executable_qty": executable_qty,
                "buy_price": avg_price_a,
                "sell_price": avg_price_b,
                "gross_spread_pct": gross_spread_pct,
                "net_spread_pct": net_spread_pct
            }

        # Route 2: Buy Exchange B (asks_b), Sell Exchange A (bids_a)
        qty_b_2, avg_price_b_2 = self._walk_book_to_fill(asks_b, max_order_usd)
        qty_a_2, avg_price_a_2 = self._walk_book_to_fill(bids_a, max_order_usd)

        if qty_b_2 == 0 or qty_a_2 == 0:
            return {"action": "HOLD", "reason": "Insufficient order book depth to execute minimum lot."}

        executable_qty_2 = min(qty_b_2, qty_a_2)
        gross_spread_pct_2 = (avg_price_a_2 - avg_price_b_2) / avg_price_b_2
        net_spread_pct_2 = gross_spread_pct_2 - total_fees

        if net_spread_pct_2 >= self.min_profit_spread_pct:
            logger.info(
                f"🏆 ARBITRAGE OPPORTUNITY FOUND ({symbol}): Buy B (${avg_price_b_2:.2f}) -> Sell A (${avg_price_a_2:.2f}). "
                f"Executable Qty: {executable_qty_2:.4f}. Net Profit: {net_spread_pct_2*100:.3f}%"
            )
            return {
                "action": "EXECUTE_ARBITRAGE",
                "symbol": symbol,
                "route": "BUY_B_SELL_A",
                "executable_qty": executable_qty_2,
                "buy_price": avg_price_b_2,
                "sell_price": avg_price_a_2,
                "gross_spread_pct": gross_spread_pct_2,
                "net_spread_pct": net_spread_pct_2
            }

        return {"action": "HOLD", "reason": "No profitable arbitrage spread after accounting for slippages and fees."}

    def _walk_book_to_fill(self, book_levels: list, max_usd: float) -> tuple:
        """
        Simulates walking down the order book levels to fill an order of size max_usd.
        Returns: (filled_qty, average_price)
        """
        filled_qty = 0.0
        spent_usd = 0.0

        for lvl in book_levels:
            price = float(lvl[0])
            qty = float(lvl[1])

            level_usd = price * qty
            if spent_usd + level_usd >= max_usd:
                # Fill the remaining fraction
                needed_usd = max_usd - spent_usd
                needed_qty = needed_usd / price
                filled_qty += needed_qty
                spent_usd = max_usd
                break
            else:
                filled_qty += qty
                spent_usd += level_usd

        if spent_usd == 0:
            return 0.0, 0.0

        avg_price = spent_usd / filled_qty if filled_qty > 0 else 0.0
        return filled_qty, avg_price

    def detect_arbitrage_opportunities(self, symbol: str, dex_price: float, cex_price: float, estimated_gas_usd: float = 0.05) -> dict:
        """
        Simplified price-based fallback for cross-venue spreads (DEX vs CEX).
        Calculates spread and net profit after deducting fees and estimated gas.
        """
        if not dex_price or not cex_price:
            return {"action": "HOLD", "reason": "Price feeds unavailable."}

        # Route 1: Buy DEX, Sell CEX
        spread = (cex_price - dex_price) / dex_price
        fee_pct = 0.002 # standard taker fee
        net_profit_pct = spread - fee_pct - (estimated_gas_usd / dex_price)

        if net_profit_pct >= self.min_profit_spread_pct:
            return {
                "action": "EXECUTE_ARBITRAGE",
                "route": "BUY_DEX_SELL_CEX",
                "spread_pct": spread,
                "net_profit_pct": net_profit_pct
            }

        # Route 2: Buy CEX, Sell DEX
        spread_rev = (dex_price - cex_price) / cex_price
        net_profit_pct_rev = spread_rev - fee_pct - (estimated_gas_usd / cex_price)
        if net_profit_pct_rev >= self.min_profit_spread_pct:
            return {
                "action": "EXECUTE_ARBITRAGE",
                "route": "BUY_CEX_SELL_DEX",
                "spread_pct": spread_rev,
                "net_profit_pct": net_profit_pct_rev
            }

        return {"action": "HOLD", "reason": "No profitable spread detected."}
