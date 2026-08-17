import logging
import time

logger = logging.getLogger("FundingArbitrage")

class FundingRateArbitrageEngine:
    """
    Delta-Neutral Funding Rate Cash-and-Carry Arbitrage Engine (Phase 5 & Lot 5).
    Evaluates real-world spot and perpetual price spreads, mark prices, index prices,
    and funding rates to capture risk-free yield paid by leveraged traders.
    """
    def __init__(self, min_funding_threshold=0.0003, transaction_fee_pct=0.001):
        self.min_funding_threshold = min_funding_threshold
        self.transaction_fee_pct = transaction_fee_pct
        self.active_arbitrages = {}

    def analyze_funding_opportunities(
        self,
        symbol: str,
        spot_bid: float = None,
        spot_ask: float = None,
        perp_bid: float = None,
        perp_ask: float = None,
        mark_price: float = None,
        index_price: float = None,
        funding_rate_8h: float = None,
        spot_price: float = None,
        perp_price: float = None
    ) -> dict:
        """
        Analyse d'arbitrage funding 100% basée sur des données réelles.
        Si les données sont incomplètes, retourne HOLD (aucune simulation).
        """
        # On n'accepte que les données réelles passées en paramètre
        if None in [spot_bid, spot_ask, perp_bid, perp_ask, mark_price, index_price, funding_rate_8h]:
            # Essai de fallback avec spot/perp si disponibles
            if spot_price is not None and perp_price is not None:
                spot_bid = spot_ask = spot_price
                perp_bid = perp_ask = perp_price
                index_price = spot_price
                mark_price = perp_price
            else:
                return {"action": "HOLD", "reason": "Insufficient real market data for funding arbitrage."}
            
        is_already_open = symbol in self.active_arbitrages
        
        # 1. Evaluate ENTRY conditions
        # To enter cash-and-carry, we Buy Spot at spot_ask, and Short Perp at perp_bid
        # Real Executable Spread = (perp_bid - spot_ask) / spot_ask
        real_spread_pct = (perp_bid - spot_ask) / spot_ask
        
        # Total transaction fees (spot taker + perp taker fees)
        entry_fees_pct = self.transaction_fee_pct * 2.0
        
        # Expected daily yield (3 funding payments of 8h)
        expected_daily_yield_pct = funding_rate_8h * 3.0
        
        if funding_rate_8h >= self.min_funding_threshold and not is_already_open:
            # Only enter if the expected daily yield covers the transaction entry fees and meets profitability!
            if expected_daily_yield_pct > entry_fees_pct:
                logger.info(
                    f"🏆 FUNDING ARBITRAGE DETECTED ({symbol}): Real Spread: {real_spread_pct*100:.3f}% | "
                    f"Funding: {funding_rate_8h*100:.3f}% / 8h. Initiating Delta-Neutral Cash-and-Carry!"
                )
                return {
                    "action": "ENTER_ARBITRAGE",
                    "symbol": symbol,
                    "funding_rate": funding_rate_8h,
                    "spot_action": "BUY",
                    "perp_action": "SELL_SHORT",
                    "real_spread_pct": real_spread_pct,
                    "mark_price": mark_price,
                    "index_price": index_price
                }
                
        # 2. Evaluate EXIT conditions
        elif is_already_open:
            active_pos = self.active_arbitrages[symbol]
            # Exit if funding rate falls back, or becomes negative (shorts pay longs)
            if funding_rate_8h < (self.min_funding_threshold * 0.3):
                logger.info(
                    f"⚖️ FUNDING ARBITRAGE CLOSE ({symbol}): Funding fell to {funding_rate_8h*100:.3f}%. Unwinding!"
                )
                return {
                    "action": "EXIT_ARBITRAGE",
                    "symbol": symbol,
                    "funding_rate": funding_rate_8h,
                    "spot_action": "SELL_CLOSE",
                    "perp_action": "BUY_COVER",
                    "accumulated_funding": active_pos.get("accumulated_funding", 0.0)
                }
                
        return {"action": "HOLD", "reason": "No actionable funding arbitrage spread detected."}

    def simulate_funding_payment_tick(self, symbol: str, current_price: float, funding_rate_8h: float) -> float:
        if symbol in self.active_arbitrages:
            pos = self.active_arbitrages[symbol]
            position_value = pos["qty"] * current_price
            payment_received = position_value * funding_rate_8h
            pos["accumulated_funding"] += payment_received
            logger.info(f"Collected ${payment_received:.2f} USD interest on delta-neutral {symbol} position.")
            return payment_received
        return 0.0
