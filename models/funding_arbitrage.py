import logging
import random

logger = logging.getLogger("FundingArbitrage")

class FundingRateArbitrageEngine:
    """
    Delta-Neutral Funding Rate Cash-and-Carry Arbitrage Engine.
    Monitors Perpetual Futures funding rates and Spot rates to capture 
    risk-free cash payment yield streams paid by leveraged market participants.
    """
    def __init__(self, min_funding_threshold=0.0005, transaction_fee_pct=0.001):
        # Default minimum positive funding rate to trigger arbitrage: 0.05% per 8 hours
        self.min_funding_threshold = min_funding_threshold
        self.transaction_fee_pct = transaction_fee_pct
        
        # Open arbitrage positions tracker: symbol -> {"qty": qty, "entry_spot_price": price, "entry_perp_price": price, "accumulated_funding": value}
        self.active_arbitrages = {}

    def analyze_funding_opportunities(self, symbol: str, spot_price: float, perp_price: float, funding_rate_8h: float) -> dict:
        """
        Evaluates funding arbitrage metrics and emits clear execution instructions:
        - ENTRY: If funding rate is highly positive, buy Spot (Long) and Sell Perpetual (Short) delta-neutral.
        - EXIT: If funding rate drops below threshold or becomes negative, close/unwind both positions to lock in interest profits.
        """
        if spot_price <= 0 or perp_price <= 0:
            return {"action": "HOLD", "reason": "Market prices offline."}
            
        is_already_open = symbol in self.active_arbitrages
        
        # 1. Evaluate ENTRY conditions
        # If funding rate is extremely positive, shorts are paid by longs.
        # We short Perp and buy Spot -> delta-neutral, capturing the funding fee!
        if funding_rate_8h >= self.min_funding_threshold and not is_already_open:
            # Check if Spot/Perp spread is favorable
            spread_pct = (perp_price - spot_price) / spot_price
            
            # Entry cost in fees (spot taker + perp taker fees)
            entry_cost_pct = self.transaction_fee_pct * 2.0
            
            # Expected net yield for 1 day (3 funding payments of 8h)
            expected_daily_yield_pct = funding_rate_8h * 3.0
            
            if expected_daily_yield_pct > entry_cost_pct:
                logger.info(
                    f"FUNDING ARBITRAGE OPPORTUNITY: {symbol} Funding Rate is {funding_rate_8h*100:.3f}% per 8h. "
                    f"Entering Delta-Neutral Cash-and-Carry position!"
                )
                return {
                    "action": "ENTER_ARBITRAGE",
                    "symbol": symbol,
                    "funding_rate": funding_rate_8h,
                    "spot_action": "BUY",
                    "perp_action": "SELL_SHORT",
                    "spread_pct": spread_pct,
                    "expected_daily_yield_pct": expected_daily_yield_pct
                }
                
        # 2. Evaluate EXIT conditions
        elif is_already_open:
            active_pos = self.active_arbitrages[symbol]
            # Exit if funding rate drops significantly below our threshold
            # or if it becomes negative (shorts pay longs, which would erode our interest profit!)
            if funding_rate_8h < (self.min_funding_threshold * 0.3):
                logger.info(
                    f"FUNDING ARBITRAGE WINDDOWN: {symbol} Funding Rate fell to {funding_rate_8h*100:.3f}%. "
                    f"Closing Delta-Neutral positions to lock in accumulated interest yield!"
                )
                return {
                    "action": "EXIT_ARBITRAGE",
                    "symbol": symbol,
                    "funding_rate": funding_rate_8h,
                    "spot_action": "SELL_CLOSE",
                    "perp_action": "BUY_COVER",
                    "accumulated_funding": active_pos.get("accumulated_funding", 0.0)
                }
                
        return {"action": "HOLD", "reason": "No actionable funding rate divergence."}

    def simulate_funding_payment_tick(self, symbol: str, current_price: float, funding_rate_8h: float):
        """
        Simulates the 8-hour epoch tick, adding the actual funding fee payment
        directly to our accumulated arbitrage yield cash register.
        """
        if symbol in self.active_arbitrages:
            pos = self.active_arbitrages[symbol]
            position_value = pos["qty"] * current_price
            
            # Funding payment paid to shorts: Position_Value * Funding_Rate
            payment_received = position_value * funding_rate_8h
            pos["accumulated_funding"] += payment_received
            logger.info(f"FUNDING RECEIVED: Collected ${payment_received:.2f} USD interest on delta-neutral {symbol} position.")
            return payment_received
        return 0.0
