"""
External Indicators Module
Gère les données avancées (Sentiment, OnChain, Macro, Funding, etc.)
"""
import logging

import httpx

logger = logging.getLogger("Indicators")

class ExternalIndicators:
    def __init__(self, state: dict, news_analyzer, onchain_tracker, macro_calendar, funding_arb_engine):
        self.state = state
        self.news = news_analyzer
        self.onchain = onchain_tracker
        self.macro = macro_calendar
        self.funding = funding_arb_engine

    async def update_sentiment(self, loop_count: int) -> float:
        """Met à jour le sentiment de marché"""
        if loop_count % 3 == 1:
            try:
                res = await self.news.get_market_sentiment_index()
                self.state["sentiment_index"] = res["sentiment_index"]
                return res.get("shock_status", {}).get("shock_detected", False)
            except Exception as e:
                logger.warning(f"Sentiment fetch failed: {e}")
        return False

    async def update_onchain(self, loop_count: int):
        """Met à jour les données on-chain"""
        if loop_count % 5 == 1:
            try:
                onchain_data = await self.onchain.get_exchange_netflows()
                self.state["onchain_risk_score"] = self.onchain.compute_onchain_risk_score(onchain_data)
            except Exception as e:
                logger.warning(f"OnChain fetch failed: {e}")

    async def update_macro(self):
        """Vérifie les événements macroéconomiques"""
        try:
            macro_res = self.macro.check_upcoming_macro_shocks()
            if macro_res.get("upcoming_shock"):
                # Logique d'alerte Telegram déjà dans le main
                pass
            return macro_res.get("scale_reduction_factor", 1.0)
        except Exception as e:
            logger.warning(f"Macro calendar error: {e}")
            return 1.0

    async def update_funding_arbitrage(self, symbol: str, current_price: float):
        """Analyse les opportunités de funding rate arbitrage"""
        if symbol not in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            return None

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}")
                if resp.status_code == 200:
                    funding_8h = float(resp.json().get("lastFundingRate", 0.0001))
                    return self.funding.analyze_funding_opportunities(
                        symbol=symbol,
                        spot_price=current_price,
                        perp_price=current_price,
                        funding_rate_8h=funding_8h
                    )
        except Exception as e:
            logger.warning(f"Funding rate fetch failed: {e}")
        return None
