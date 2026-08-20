import asyncio
import logging
import re

import httpx

logger = logging.getLogger("SentimentAnalyzer")

# LOT 5 (PDF Pilier I) : pondération des sources — toutes les actualités ne se
# valent pas : Google News (couverture mondiale) et CryptoCompare (marchés)
# pèsent plus que Reddit (bruit communautaire).
SOURCE_WEIGHTS = {
    "cryptocompare": 0.30,
    "google_news": 0.35,
    "alpha_vantage": 0.20,
    "reddit": 0.15,
}

# Intensificateurs (PDF Pilier I) : amplifient le sentiment du mot de lexique
# qui suit ("record rally", "massive selloff"...). Multiplicateur borné 1.0-2.0.
INTENSIFIERS = {
    "record": 1.5, "all-time": 1.5, "unprecedented": 1.5, "historic": 1.4,
    "massive": 1.5, "huge": 1.4, "major": 1.3, "sharp": 1.3, "dramatic": 1.3,
    "extreme": 1.3, "collapse": 1.4, "plunge": 1.3, "surge": 1.3, "soar": 1.3,
    "slump": 1.3, "tank": 1.3, "crisis": 1.4, "stunning": 1.3, "epic": 1.3,
    "panic": 1.3,
}

class NewsSentimentAnalyzer:
    """
    Advanced Context-Aware Financial Natural Language Processing (NLP) Sentiment Analyzer.
    Aggregates headlines across multiple sources (CryptoCompare, Reddit, and Alpha Vantage),
    detects negation contexts (e.g. 'crash avoided' -> positive), and includes
    an Extreme Event Spike Detector for immediate emergency portfolio risk reductions.
    """
    def __init__(self):
        # Lexicon of financial sentiment words with base scores
        self.lexicon = {
            "bullish": 0.8, "breakout": 0.7, "rally": 0.8, "surge": 0.6,
            "gain": 0.4, "profit": 0.5, "adoption": 0.6, "partnership": 0.5,
            "halving": 0.7, "accumulation": 0.6, "ath": 0.8, "optimistic": 0.5,
            "bearish": -0.8, "breakdown": -0.7, "crash": -0.9, "plunge": -0.7,
            "dump": -0.8, "loss": -0.4, "regulation": -0.5, "ban": -0.8,
            "scam": -0.9, "hack": -0.8, "sec": -0.6, "liquidation": -0.7,
            "panic": -0.8, "fud": -0.7, "selloff": -0.7, "collapse": -0.9,
            "illegal": -0.6, "insolvency": -0.9, "bankruptcy": -0.9
        }

        # Negation modifiers that invert the sentiment of the following word
        self.negations = ["not", "no", "never", "avoided", "prevented", "rejected", "deny", "denies", "false", "without"]

        # High-impact emergency shock tokens (trigger instant risk limits reduction)
        self.shock_tokens = {
            "hack": "SECURITY_HACK_ALERT",
            "exploit": "SECURITY_HACK_ALERT",
            "insolvency": "INSOLVENCY_ALERT",
            "bankruptcy": "INSOLVENCY_ALERT",
            "ban": "REGULATORY_BAN_ALERT",
            "illegal": "REGULATORY_BAN_ALERT",
            "arrested": "CRIMINAL_CHARGES_ALERT",
            "shutdown": "REGULATORY_BAN_ALERT"
        }

    async def fetch_cryptocompare_news(self) -> list:
        try:
            # Optional CryptoCompare API key (roadmap ecosystem #3) removes 401s
            import os
            api_key = os.getenv("CRYPTOCOMPARE_API_KEY", "")
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
            params = {"extraParams": "quant-portal"}
            if api_key:
                params["api_key"] = api_key
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json().get("Data", [])
                    return [art.get("title", "") for art in data[:10]]
        except Exception as e:
            logger.warning(f"CryptoCompare News fetch failed: {str(e)}")
        return []

    async def fetch_reddit_news(self) -> list:
        try:
            # Fetch public hot RSS feed of r/cryptocurrency converted to JSON
            url = "https://www.reddit.com/r/cryptocurrency/hot.json?limit=10"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) QuantPortal/3.0"}
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    children = resp.json().get("data", {}).get("children", [])
                    return [post.get("data", {}).get("title", "") for post in children]
        except Exception as e:
            logger.warning(f"Reddit r/cryptocurrency fetch failed: {str(e)}")
        return []

    async def fetch_alpha_vantage_news(self) -> list:
        try:
            # Query Alpha Vantage free public news feed proxy
            url = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=CRYPTO:BTC&limit=10&apikey=demo"
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    feed = resp.json().get("feed", [])
                    return [art.get("title", "") for art in feed]
        except Exception as e:
            logger.warning(f"Alpha Vantage News fetch failed: {str(e)}")
        return []

    async def fetch_all_sources(self) -> list:
        """
        Polls multiple independent news APIs concurrently to avoid any single-point-of-failure.
        LOT 5 (PDF Pilier I) : mémorise la SOURCE de chaque titre (pour la
        pondération et l'endpoint /api/v1/news).
        """
        tasks = [
            ("cryptocompare", self.fetch_cryptocompare_news()),
            ("reddit", self.fetch_reddit_news()),
            ("alpha_vantage", self.fetch_alpha_vantage_news()),
            ("google_news", self.fetch_google_news_rss())
        ]
        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

        # Flatten results list + mémoriser la source de chaque titre
        all_headlines = []
        self._source_map = {}
        for (name, _), r in zip(tasks, results):
            if isinstance(r, list):
                for h in r:
                    if h and h not in self._source_map:
                        all_headlines.append(h)
                        self._source_map[h] = name
        self.last_headlines = [(self._source_map[h], h) for h in all_headlines]

        # RÈGLE D'HONNÊTETÉ (faille 1 corrigée — mentalité n°5 : la confiance dans
        # le signal compte autant que le signal) : si TOUTES les sources réelles
        # sont hors ligne, on renvoie [] et le sentiment devient "UNAVAILABLE".
        # JAMAIS de titres d'actualité inventés qui influenceraient les trades.
        if not all_headlines:
            logger.warning(
                "SentimentAnalyzer: toutes les sources réelles sont hors ligne "
                "-> sentiment UNAVAILABLE (n'influencera AUCUN trade)."
            )
            self.last_headlines = []
        return list(set(all_headlines))  # Deduplicate

    def get_recent_headlines(self, limit: int = 20) -> list:
        """LOT 5 : dernières actualités RÉELLES (source + titre) pour l'API."""
        return [{"source": s, "title": t} for s, t in getattr(self, "last_headlines", [])[:limit]]

    async def fetch_google_news_rss(self) -> list:
        """
        Flux RSS Google News (sans clé, 100% réel) — couverture mondiale
        crypto/marchés en complément des autres sources.
        """
        try:
            import xml.etree.ElementTree as ET
            url = ("https://news.google.com/rss/search?"
                   "q=crypto+OR+bitcoin+OR+ethereum+OR+stock+market&hl=en-US&gl=US&ceid=US:en")
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 QuantPortal/3.0"})
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                titles = []
                for item in root.iter("item"):
                    title = item.findtext("title")
                    if title and not title.startswith("\""):
                        # Format Google News : "Titre - Site" (le site est APRÈS
                        # le dernier séparateur) — on garde le TITRE réel
                        cleaned = title.rsplit(" - ", 1)[0].strip()
                        titles.append(cleaned)
                return titles[:10]
        except Exception as e:
            logger.debug(f"Google News RSS fetch failed: {e}")
        return []

    def analyze_semantic_context(self, text: str) -> float:
        """
        Linguistic Parser with Negation and Valence Shifting (Context-Aware FinBERT-like logic).
        Correctly parses e.g. 'crash avoided' -> positive (+0.9), rather than negative (-0.9).

        LOT 5 (PDF Pilier I) : NÉGATION À DISTANCE (jusqu'à 3 mots avant le mot
        de lexique) + INTENSIFICATEURS ("record rally" amplifié) — le sentiment
        est plus robuste qu'une simple négation au mot précédent.
        """
        cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = cleaned.split()

        score = 0.0
        match_count = 0
        negate_next = False

        for idx, word in enumerate(words):
            # Check if this word is a negation modifier
            if word in self.negations:
                negate_next = True
                continue

            if word in self.lexicon:
                base_sentiment = self.lexicon[word]

                # Négation à distance BIDIRECTIONNELLE (LOT 5, PDF Pilier I) :
                #  - avant : "reports say no crash" (négation dans les 3 mots précédents)
                #  - après : "crash avoided" (négation post-posée, typique anglais)
                negated = negate_next
                for j in range(max(0, idx - 3), idx):
                    if words[j] in self.negations:
                        negated = True
                        break
                if not negated and idx + 1 < len(words) and words[idx + 1] in self.negations:
                    negated = True
                if negated:
                    # Invert and scale down slightly for semantic naturalness
                    base_sentiment = -base_sentiment * 0.9
                    negate_next = False

                # Intensificateur : amplifier si un mot fort précède (borné 1.0-2.0)
                multiplier = 1.0
                for j in range(max(0, idx - 2), idx):
                    if words[j] in INTENSIFIERS:
                        multiplier = max(1.0, min(2.0, INTENSIFIERS[words[j]]))
                        break
                base_sentiment *= multiplier

                score += base_sentiment
                match_count += 1

        if match_count == 0:
            return 0.0
        return max(-1.0, min(1.0, score / match_count))

    def detect_extreme_event_shock(self, headlines: list) -> dict:
        """
        Extreme Event Spike Detector.
        Scans all collected headlines for high-impact systemic shock tokens (HACK, INSOLVENCY, BAN).
        Bypasses smoothed averages to instantly alert the risk manager of black swan events!
        """
        for h in headlines:
            cleaned = re.sub(r'[^a-zA-Z\s]', '', h.lower())
            words = cleaned.split()

            for word in words:
                if word in self.shock_tokens:
                    alert_type = self.shock_tokens[word]
                    logger.critical(f"⚠️ SYSTEMIC SHOCK DETECTED in news headlines! Token: '{word}' -> Triggering {alert_type}!")
                    return {
                        "shock_detected": True,
                        "token": word,
                        "alert_type": alert_type,
                        "headline": h
                    }

        return {"shock_detected": False}

    async def get_market_sentiment_index(self) -> dict:
        """
        Aggregates news, computes context-aware NLP scores, and detects extreme shocks.

        HONNÊTETÉ (faille 1 corrigée) : si aucune actualité RÉELLE n'est disponible,
        renvoie sentiment_index=None et available=False. Le main loop n'influencera
        alors AUCUN trade avec du sentiment (mentalité n°5 + n°20 : savoir dire
        « je ne sais pas »).
        """
        headlines = await self.fetch_all_sources()
        if not headlines:
            return {
                "sentiment_index": None,
                "available": False,
                "confidence": 0.0,
                "num_headlines": 0,
                "shock_status": {"shock_detected": False},
            }

        # LOT 5 (PDF Pilier I) : moyenne PONDÉRÉE par la fiabilité des sources
        # (Google News/CryptoCompare > AlphaVantage > Reddit). Une source sans
        # poids connu = poids 1.0 (neutre).
        scores = []
        weights = []
        for h in headlines:
            scores.append(self.analyze_semantic_context(h))
            weights.append(SOURCE_WEIGHTS.get(self._source_map.get(h, ""), 1.0))
        total_w = sum(weights) or 1.0
        avg_sentiment = float(sum(s * w for s, w in zip(scores, weights)) / total_w)

        # Check for immediate systemic shocks
        shock_status = self.detect_extreme_event_shock(headlines)

        # Score de confiance : plus il y a de titres réels, plus on est confiant
        confidence = min(1.0, len(headlines) / 10.0)

        return {
            "sentiment_index": avg_sentiment,
            "available": True,
            "confidence": confidence,
            "num_headlines": len(headlines),
            "shock_status": shock_status,
        }
