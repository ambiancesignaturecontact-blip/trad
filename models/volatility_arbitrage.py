import logging
import math
import os
import time

logger = logging.getLogger("VolatilityArbitrage")

# Correspondance symboles internes -> paires Deribit (options BTC/ETH réelles)
DERIBIT_CURRENCY = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}


class OptionsVolatilityArbitrageEngine:
    """
    Derivative Options Volatility Arbitrage Engine (Deribit / Bybit Options format).
    Formulates optimal options structures (Covered Calls, Cash-Secured Puts,
    Straddles, and Iron Condors) based on HMM volatility regimes and volatility skews.

    HONNÊTETÉ (faille 1 corrigée — mentalité n°5) : l'implied volatility doit être
    RÉELLE (DVOL Deribit). Si aucune source réelle n'est joignable, le moteur
    renvoie UNAVAILABLE — plus JAMAIS de valeur d'IV inventée (ancien iv_map).
    """
    def __init__(self):
        self._iv_cache = {}
        self._iv_cache_ts = {}

    async def fetch_real_iv(self, symbol: str, max_age_seconds: float = 900.0):
        """
        Récupère l'implied volatility annualisée RÉELLE (indice DVOL Deribit)
        pour BTC/ETH. Retourne None si indisponible (cache 15 min).
        """
        currency = DERIBIT_CURRENCY.get(symbol)
        if not currency:
            return None  # SOL / or / fx / actions : pas de marché d'options gratuit fiable

        now = time.time()
        if self._iv_cache.get(symbol) is not None and \
           now - self._iv_cache_ts.get(symbol, 0.0) < max_age_seconds:
            return self._iv_cache[symbol]

        try:
            import httpx
            start_ts = int((now - 86400) * 1000)
            end_ts = int(now * 1000)
            url = ("https://www.deribit.com/api/v2/public/get_volatility_index_data"
                   f"?currency={currency}&start_timestamp={start_ts}"
                   f"&end_timestamp={end_ts}&resolution=1D")
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.get(url)
            if resp.status_code != 200:
                logger.warning(f"VolArb: Deribit DVOL {symbol} HTTP {resp.status_code}")
                return None
            data = resp.json().get("result", {}).get("data", [])
            if not data:
                return None
            # Format Deribit : [timestamp, open, high, low, close] (listes, pas dicts)
            last = data[-1]
            if isinstance(last, dict):
                close = last.get("close") or last.get("open") or 0.0
            else:
                close = last[4] if len(last) > 4 else last[1]
            # DVOL est exprimé en % -> on divise par 100 pour une IV annualisée
            dvol = float(close) / 100.0
            if not (0.01 <= dvol <= 3.0):  # garde de plausibilité (1%..300%)
                logger.warning(f"VolArb: DVOL {symbol} hors plage plausible ({dvol}) -> ignoré")
                return None
            self._iv_cache[symbol] = dvol
            self._iv_cache_ts[symbol] = now
            logger.info(f"VolArb: IV réelle Deribit {symbol} = {dvol*100:.1f}%")
            return dvol
        except Exception as e:
            logger.warning(f"VolArb: IV réelle indisponible pour {symbol}: {e}")
            return None

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
        if iv_annual is None or iv_annual <= 0:
            # HONNÊTETÉ : pas d'IV réelle -> pas de stratégie d'options calculée
            return {
                "strategy": "UNAVAILABLE",
                "details": "Implied volatility source indisponible — aucune stratégie d'options calculée.",
                "implied_volatility_pct": None,
                "legs": [],
                "estimated_yield_pct": 0.0,
            }

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
