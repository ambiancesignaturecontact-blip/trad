"""
VISION_FUTUR §3/§6 - LLM narrative & assistant via OpenRouter (OpenAI-compatible).

- daily_market_narrative(): why the market moved + how the bot performed
- explain_decision(): plain-language explanation grounded in the REAL journal
- answer_question(): the operator can TALK to the bot ("pourquoi tu n'as pas acheté hier?")
- graceful fallback: without OPENROUTER_API_KEY -> structured text (no LLM)
"""
import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("LLMNarrative")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def _api_key() -> Optional[str]:
    return os.getenv("OPENROUTER_API_KEY", "").strip() or None


def _complete(prompt: str, system: str = "Tu es le narrateur quantitatif d'un bot de trading.", max_tokens: int = 400) -> str:
    key = _api_key()
    if not key:
        return ""
    try:
        import httpx
        resp = httpx.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("LLM_MODEL", DEFAULT_MODEL),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.5,
            },
            timeout=30.0,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        logger.warning(f"OpenRouter HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"OpenRouter call failed: {e}")
    return ""


def daily_market_narrative(report: dict, state: dict) -> str:
    """Daily narrative: market context + bot performance, LLM or structured."""
    mode = report.get("mode", "DEMO")
    equity = report.get("equity", 0.0)
    pnl = report.get("pnl_usd", 0.0)
    pnl_pct = report.get("pnl_pct", 0.0)
    health = report.get("health_score", 0)
    regime = report.get("risk", {}).get("regime", "?")
    orders = report.get("orders_today", 0)
    pos = report.get("positions", [])

    if _api_key():
        prompt = (
            f"Rédige un narratif de marché quotidien (6-8 lignes, en français, ton sobre et "
            f"professionnel, sans jargon inutile) pour un terminal quantitatif. "
            f"Mode {mode}, équité {equity:.2f} USD, P&L du jour {pnl:+.2f} USD ({pnl_pct:+.2f}%), "
            f"santé du bot {health}/100, régime de marché '{regime}', {orders} ordres aujourd'hui, "
            f"{len(pos)} positions ouvertes. Mentionne les points d'attention si la santé est basse."
        )
        narrative = _complete(prompt)
        if narrative:
            return narrative
    # structured fallback (no LLM key or failure)
    lines = [
        f"📊 *Narratif du jour ({mode})*",
        f"━━━━━━━━━━━━━━━━━━",
        f"Régime : *{regime}* | Santé : {health}/100",
        f"Équité : ${equity:,.2f} | P&L : {pnl:+,.2f} $ ({pnl_pct:+.2f}%)",
        f"Ordres aujourd'hui : {orders} | Positions : {len(pos)}",
    ]
    if health < 60:
        lines.append("⚠️ Santé faible : réduction automatique des tailles en cours.")
    return "\n".join(lines)


def explain_decision(decision: dict) -> str:
    """Plain-language explanation of a decision (grounded in real features)."""
    symbol = decision.get("symbol", "?")
    side = decision.get("side", "?")
    reasons = decision.get("reasoning", [])
    regime = decision.get("regime", "?")
    if _api_key():
        prompt = (
            f"Explique en 2-3 phrases claires (français) une décision de trading: "
            f"{side} {symbol}, régime '{regime}'. Raisons (features réelles): {reasons}. "
            f"Style sobre et précis."
        )
        txt = _complete(prompt, max_tokens=120)
        if txt:
            return txt
    top = ", ".join(r.get("feature", "?") for r in reasons[:3]) if reasons else "signaux du méta-modèle"
    return f"{side} {symbol} — {top} (régime {regime})."


def answer_question(question: str, context: dict) -> str:
    """The operator talks to the bot; answers grounded in the REAL journal/state."""
    if not _api_key():
        # structured fallback: answer from live data
        price = context.get("last_price", 0.0)
        equity = context.get("current_equity", 0.0)
        regime = context.get("regime_name", "?")
        q = question.lower()
        if "prix" in q or "bitcoin" in q or "btc" in q:
            return f"BTC est à ${price:,.2f} actuellement (régime '{regime}'). Équité du bot : ${equity:,.2f}."
        if "achet" in q or "pourquoi" in q:
            return ("Je me base sur le méta-modèle (12 stratégies pondérées), le régime, la "
                    "microstructure (VPIN) et la qualité des données. Configurez OPENROUTER_API_KEY "
                    "pour des réponses détaillées.")
        return f"Régime: {regime} | BTC: ${price:,.2f} | Équité: ${equity:,.2f}. (Réponse structurée — clé LLM absente.)"
    prompt = (
        f"Tu es l'assistant d'un terminal de trading quantitatif. Réponds en français, de façon "
        f"précise et honnête, en t'appuyant sur ces données réelles: {json.dumps(context, default=str)[:2000]}. "
        f"Question de l'opérateur: {question}"
    )
    return _complete(prompt, system="Assistant quantitatif sobre, honnête, sans promesse de gain.", max_tokens=350) or "Je n'ai pas pu formuler de réponse (API LLM indisponible)."
