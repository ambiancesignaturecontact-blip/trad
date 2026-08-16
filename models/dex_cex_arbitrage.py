import logging
import random

logger = logging.getLogger("DexCexArbitrage")

class DexCexArbitrageEngine:
    """
    Cross-Venue DEX-CEX Arbitrage Engine.
    Monitors price discrepancies between Decentralized Exchanges (Uniswap/1inch on L2s)
    and Centralized Exchanges (Binance/Bybit), capturing instant risk-free spreads.
    """
    def __init__(self, min_profit_spread_pct=0.005, execution_fee_pct=0.0015):
        # Default minimum profitable spread: 0.50% net of gas
        self.min_profit_spread_pct = min_profit_spread_pct
        self.execution_fee_pct = execution_fee_pct

    def detect_arbitrage_opportunities(self, symbol: str, dex_price: float, cex_price: float, estimated_gas_usd: float) -> dict:
        """
        Scans price feeds and checks if the spread between DEX and CEX covers transaction fees & gas:
        - Route 1: Buy DEX, Sell CEX (if DEX is cheaper than CEX).
        - Route 2: Buy CEX, Sell DEX (if CEX is cheaper than DEX).
        """
        if dex_price <= 0 or cex_price <= 0:
            return {"action": "HOLD", "reason": "DEX or CEX prices offline."}
            
        # Calculate raw spread %
        spread_pct = (cex_price - dex_price) / dex_price
        
        # Total transaction fees (CEX fee + DEX fee + estimated gas fee scaled by a standard $1000 order size)
        standard_order_size_usd = 1000.0
        scaled_gas_pct = estimated_gas_usd / standard_order_size_usd
        total_costs_pct = (self.execution_fee_pct * 2.0) + scaled_gas_pct
        
        # Route 1: DEX is cheaper than CEX (Positive spread)
        if spread_pct > 0:
            net_profit_pct = spread_pct - total_costs_pct
            if net_profit_pct >= self.min_profit_spread_pct:
                logger.info(
                    f"🏆 DEX-CEX ARBITRAGE DETECTED ({symbol}): DEX Price (${dex_price:.2f}) < CEX Price (${cex_price:.2f}). "
                    f"Net expected profit: {net_profit_pct*100:.3f}% (net of L2 gas)."
                )
                return {
                    "action": "EXECUTE_ARBITRAGE",
                    "symbol": symbol,
                    "route": "BUY_DEX_SELL_CEX",
                    "dex_action": "BUY",
                    "cex_action": "SELL",
                    "spread_pct": spread_pct,
                    "net_profit_pct": net_profit_pct
                }
                
        # Route 2: CEX is cheaper than DEX (Negative spread)
        else:
            abs_spread_pct = abs(spread_pct)
            net_profit_pct = abs_spread_pct - total_costs_pct
            if net_profit_pct >= self.min_profit_spread_pct:
                logger.info(
                    f"🏆 DEX-CEX ARBITRAGE DETECTED ({symbol}): CEX Price (${cex_price:.2f}) < DEX Price (${dex_price:.2f}). "
                    f"Net expected profit: {net_profit_pct*100:.3f}% (net of L2 gas)."
                )
                return {
                    "action": "EXECUTE_ARBITRAGE",
                    "symbol": symbol,
                    "route": "BUY_CEX_SELL_DEX",
                    "dex_action": "SELL",
                    "cex_action": "BUY",
                    "spread_pct": abs_spread_pct,
                    "net_profit_pct": net_profit_pct
                }
                
        return {"action": "HOLD", "reason": "No profitable cross-venue arbitrage spread."}
