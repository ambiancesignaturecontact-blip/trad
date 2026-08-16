import logging
import math

logger = logging.getLogger("VolatilityArbitrage")

class OptionsVolatilityArbitrageEngine:
    """
    Derivative Options Volatility Arbitrage Engine (Deribit / Bybit Options format).
    Formulates optimal options structures (Covered Calls, Cash-Secured Puts, 
    Straddles, and Iron Condors) based on HMM volatility regimes and volatility skews.
    """
    def __init__(self):
        pass

    def evaluate_optimal_options_strategy(self, current_price: float, iv_annual: float, regime_id: int) -> dict:
        """
        Formulates the mathematical optimal strategy based on the current regime:
        - Regime 0 (Bull Low Vol) -> covered call writing (yield generation).
        - Regime 1 (Bear High Vol) -> buy protective puts (insurance).
        - Regime 2 (Range Low Vol) -> write iron condors or sell straddles (capture theta decay).
        - Regime 3 (Erratic High Vol) -> buy long straddles or strangles (capture breakout delta/gamma).
        """
        if current_price is None or current_price <= 0:
            return {"strategy": "PASSIVE", "details": "Asset price offline."}
            
        # Standard deviation proxy for option strikes (e.g. 1 month duration, 30 days)
        time_days = 30
        t_years = time_days / 365.0
        one_sd_move = current_price * iv_annual * math.sqrt(t_years)
        
        strategy_name = "PASSIVE"
        legs = []
        expected_premium_pct = 0.0
        
        if regime_id == 0:
            # Bullish Low Vol: Covered Call (Write out-of-the-money Call at +1 SD)
            strike_call = current_price + one_sd_move
            strategy_name = "COVERED_CALL_WRITE"
            legs = [
                {"type": "LONG", "asset": "SPOT", "strike": current_price},
                {"type": "SHORT", "asset": "CALL_OPTION", "strike": strike_call, "premium_est": iv_annual * current_price * 0.05}
            ]
            expected_premium_pct = 2.5
            
        elif regime_id == 1:
            # Bearish High Vol: Long Put (Protective put at -1 SD)
            strike_put = current_price - one_sd_move
            strategy_name = "PROTECTIVE_PUT_BUY"
            legs = [
                {"type": "LONG", "asset": "PUT_OPTION", "strike": strike_put, "cost_est": iv_annual * current_price * 0.04}
            ]
            expected_premium_pct = -4.0
            
        elif regime_id == 2:
            # Low Volatility Range: Short Straddle (Write ATM Call & Write ATM Put)
            # Capitalizes on high theta decay
            strategy_name = "SHORT_STRADDLE_WRITE"
            legs = [
                {"type": "SHORT", "asset": "CALL_OPTION", "strike": current_price, "premium_est": iv_annual * current_price * 0.08},
                {"type": "SHORT", "asset": "PUT_OPTION", "strike": current_price, "premium_est": iv_annual * current_price * 0.08}
            ]
            expected_premium_pct = 16.0
            
        elif regime_id == 3:
            # High Volatility Breakout: Long Straddle (Buy ATM Call & Buy ATM Put)
            # Capitalizes on massive price breakouts (Gamma/Vega squeeze)
            strategy_name = "LONG_STRADDLE_BUY"
            legs = [
                {"type": "LONG", "asset": "CALL_OPTION", "strike": current_price, "cost_est": iv_annual * current_price * 0.09},
                {"type": "LONG", "asset": "PUT_OPTION", "strike": current_price, "cost_est": iv_annual * current_price * 0.09}
            ]
            expected_premium_pct = -18.0
            
        logger.info(f"VOLATILITY ARBITRAGE: Formulated options structure: {strategy_name} (IV: {iv_annual*100:.1f}%)")
        return {
            "strategy": strategy_name,
            "implied_volatility_pct": iv_annual * 100.0,
            "one_sd_price_range": [current_price - one_sd_move, current_price + one_sd_move],
            "legs": legs,
            "estimated_yield_pct": expected_premium_pct
        }
