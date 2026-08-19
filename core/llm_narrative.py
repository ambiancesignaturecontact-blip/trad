"""
VISION_FUTUR §3/§6 - LLM narrative & assistant via OpenRouter (OpenAI-compatible).

- daily_market_narrative(): why the market moved + how the bot performed
- explain_decision(): plain-language explanation grounded in the REAL journal
- answer_question(): the operator can TALK to the bot ("pourquoi tu n'as pas acheté hier?")
- graceful fallback: without OPENROUTER_API_KEY -> structured text (no LLM)
- ASYNC-FIRST: uses httpx.AsyncClient so the event loop is never blocked by the
  LLM call. Sync wrappers are provided for tests / non-async contexts.
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


async def _complete_async(prompt: str, system: str = "Tu es le narrateur quantitatif d'un bot de trading.",
                          max_tokens: int = 400) -> str:
    """True async OpenRouter call - must be awaited from async contexts."""
    key = _api_key()
    if not key:
        return ""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
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
            )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        logger.warning(f"OpenRouter HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"OpenRouter call failed: {e}")
    return ""


def _sync_wrapper(coro_factory):
    """
    Runs an async coroutine from a sync context (tests, non-async callers).
    If a running event loop exists (FastAPI), returns "" so the caller must use
    the async variant - we never block the loop.
    """
    try:
        import asyncio as _aio
        try:
            _aio.get_running_loop()
            return ""  # running loop: must use _complete_async / *_async variants
        except RuntimeError:
            return _aio.new_event_loop().run_until_complete(coro_factory())
    except Exception:
        return ""


def _narrative_fallback(report: dict, state: dict) -> str:
    mode = report.get("mode", "DEMO")
    equity = report.get("equity", 0.0)
    pnl = report.get("pnl_usd", 0.0)
    pnl_pct = report.get("pnl_pct", 0.0)
    health = report.get("health_score", 0)
    regime = report.get("risk", {}).get("regime", "?")
    orders = report.get("orders_today", 0)
    pos = report.get("positions", [])
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


def _narrative_prompt(report: dict) -> str:
    mode = report.get("mode", "DEMO")
    equity = report.get("equity", 0.0)
    pnl = report.get("pnl_usd", 0.0)
    pnl_pct = report.get("pnl_pct", 0.0)
    health = report.get("health_score", 0)
    regime = report.get("risk", {}).get("regime", "?")
    orders = report.get("orders_today", 0)
    pos = report.get("positions", [])
    return (
        f"Rédige un narratif de marché quotidien (6-8 lignes, en français, ton sobre et "
        f"professionnel, sans jargon inutile) pour un terminal quantitatif. "
        f"Mode {mode}, équité {equity:.2f} USD, P&L du jour {pnl:+.2f} USD ({pnl_pct:+.2f}%), "
        f"santé du bot {health}/100, régime de marché '{regime}', {orders} ordres aujourd'hui, "
        f"{len(pos)} positions ouvertes. Mentionne les points d'attention si la santé est basse."
    )


async def daily_market_narrative_async(report: dict, state: dict) -> str:
    """Async narrative - for async callers (FastAPI endpoints, scheduler)."""
    if _api_key():
        txt = await _complete_async(_narrative_prompt(report))
        if txt:
            return txt
    return _narrative_fallback(report, state)


def daily_market_narrative(report: dict, state: dict) -> str:
    """Sync narrative (tests / non-async callers). Uses structured fallback + optional LLM via sync wrapper."""
    llm = _sync_wrapper(lambda: _complete_async(_narrative_prompt(report)))
    if llm:
        return llm
    return _narrative_fallback(report, state)


async def explain_decision_async(decision: dict) -> str:
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
        txt = await _complete_async(prompt, max_tokens=120)
        if txt:
            return txt
    top = ", ".join(r.get("feature", "?") for r in reasons[:3]) if reasons else "signaux du méta-modèle"
    return f"{side} {symbol} — {top} (régime {regime})."


def explain_decision(decision: dict) -> str:
    """Sync wrapper for explain_decision_async."""
    return _sync_wrapper(lambda: explain_decision_async(decision))


async def answer_question_async(question: str, context: dict) -> str:
    """Async assistant - for async callers."""
    if not _api_key():
        # structured fallback: answer from live data
        # FIX (mini-app) : last_price peut être None tant qu'aucune donnée
        # réelle n'est arrivée -> on affiche "indisponible" au lieu de crasher
        # (TypeError sur formatage de None) et la mini-app affichait
        # "Pas de réponse" sur une erreur 500.
        try:
            price = context.get("last_price")
            price_txt = f"${price:,.2f}" if isinstance(price, (int, float)) and price > 0 else "indisponible (pas encore de tick réel)"
        except (TypeError, ValueError):
            price_txt = "indisponible"
        try:
            equity = context.get("current_equity") or 0.0
            equity_txt = f"${float(equity):,.2f}"
        except (TypeError, ValueError):
            equity_txt = "indisponible"
        regime = context.get("regime_name", "?")
        q = question.lower()
        if "prix" in q or "bitcoin" in q or "btc" in q:
            return f"BTC est à {price_txt} actuellement (régime '{regime}'). Équité du bot : {equity_txt}."
        if "achet" in q or "pourquoi" in q or "strategie" in q or "trade" in q:
            return ("Je me base sur le méta-modèle (12 stratégies pondérées), le régime HMM, la "
                    "microstructure (VPIN/order flow) et la qualité des données. Toutes les données "
                    "sont réelles ; en cas de doute le bot réduit ou ne trade pas. Configurez "
                    "OPENROUTER_API_KEY pour des réponses détaillées.")
        if "risque" in q or "drawdown" in q or "sécurité" in q or "securite" in q:
            return ("La pyramide de risque : Kelly fractionnaire (win rate réel), CVaR, plafond 25% "
                    "par actif, réserve cash 15%, machine à états NORMAL/CAUTION/HALT. Le bot "
                    "survit d'abord, gagne ensuite.")
        if "position" in q or "portefeuille" in q or "equity" in q:
            return (f"Équité actuelle : {equity_txt}. Positions ouvertes : "
                    f"{', '.join(context.get('positions', [])) or 'aucune'}.")
        return (f"Régime: {regime} | BTC: {price_txt} | Équité: {equity_txt}. "
                f"(Réponse structurée — clé LLM absente, OPENROUTER_API_KEY non configurée.)")
    prompt = (
        f"Tu es l'assistant d'un terminal de trading quantitatif. Réponds en français, de façon "
        f"précise et honnête, en t'appuyant sur ces données réelles: {json.dumps(context, default=str)[:2000]}. "
        f"Question de l'opérateur: {question}"
    )
    return await _complete_async(prompt, system="Assistant quantitatif sobre, honnête, sans promesse de gain.", max_tokens=350) or "Je n'ai pas pu formuler de réponse (API LLM indisponible)."


def answer_question(question: str, context: dict) -> str:
    """Sync wrapper for answer_question_async (tests / non-async callers)."""
    return _sync_wrapper(lambda: answer_question_async(question, context))
