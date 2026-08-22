"""
VISION §4 - DÉCIDER: meta-cognition (knowing when NOT to trade).

- adaptive conviction threshold: the bot demands more evidence when its recent
  performance is worse
- explicit NO_TRADE decisions (logged with a reason, not silent absences)
- integrated hedging decision: open a correlated opposite position when
  portfolio concentration/correlation is extreme
"""
import logging
import time

import numpy as np

logger = logging.getLogger("MetaCognition")


def adaptive_conviction_threshold(recent_signals: list[float], recent_returns: list[float],
                                  base_threshold: float | None = None,
                                  min_threshold: float = 0.08, max_threshold: float = 0.30) -> float:
    """
    VISION §4a + LOT A (F1) : le seuil de conviction s'adapte à DEUX signaux
    réels :
      1. La distribution des |signaux| récents (base = percentile p25, bornée)
         — LOT A : avant, la base était une constante 0.08 (config) ce qui
         bornait le seuil à [0.08, 0.14] et ignorait la conviction réelle du
         marché (p50≈0.27). Désormais, si base_threshold est None, on prend
         le p25 des |signaux| observés : le seuil suit le niveau de signal
         effectivement produit.
      2. La précision directionnelle récente (comportement historique) :
         accuracy élevée -> barre plus basse (on fait confiance), faible ->
         barre plus haute.
    """
    # 1. Base adaptative : percentile p25 des |signaux| récents (borné)
    if base_threshold is None:
        if recent_signals and len(recent_signals) >= 10:
            abs_sig = [abs(float(s)) for s in recent_signals
                       if s is not None and float(s) != 0.0]
            if abs_sig:
                base_threshold = float(np.percentile(abs_sig, 25))
                base_threshold = float(np.clip(base_threshold, min_threshold, max_threshold))
            else:
                base_threshold = min_threshold
        else:
            base_threshold = min_threshold
    n = min(len(recent_signals), len(recent_returns))
    if n < 10:
        return float(np.clip(base_threshold, min_threshold, max_threshold))
    correct = sum(1 for i in range(-n, 0) if np.sign(recent_signals[i]) == np.sign(recent_returns[i]))
    accuracy = correct / n
    # accuracy 0.5 -> neutral; >0.55 lowers the bar, <0.45 raises it
    factor = 1.0 - (accuracy - 0.5) * 1.5
    return float(np.clip(base_threshold * factor, min_threshold, max_threshold))


def decide_no_trade(symbol: str, signal: float, threshold: float, reasons: list[str],
                    event_log=None, db=None) -> bool:
    """
    VISION §4b + LOT 1 (diagnostic) : when the signal is below the conviction
    bar, log an explicit NO_TRADE decision with the reason instead of
    silently not trading.

    - La raison par défaut inclut TOUJOURS |signal| vs seuil (observabilité :
      « pourquoi le bot n'a pas tradé ? » — les raisons passées par l'appelant
      s'y ajoutent).
    - event_log["reasons"] agrège un breakdown par CATÉGORIE de raison
      (conviction / rr_filter / halt / order_flow / cascade / meta_label /
      other) pour le dashboard et la télémétrie.
    Returns True if the bot should abstain.
    """
    if abs(signal) >= threshold:
        return False
    sig_vs_thr = f"|signal| {abs(signal):.3f} < seuil {threshold:.3f}"
    reason = " | ".join(reasons) if reasons else sig_vs_thr
    if reasons:
        reason = f"{sig_vs_thr} | {' | '.join(reasons)}"
    logger.info(f"⏸️ NO_TRADE {symbol}: {reason}")
    try:
        if db is not None:
            db.add_event(time.time(), "no_trade", f'{{"symbol": "{symbol}", "reason": "{reason}"}}')
        if event_log is not None:
            # FIX LOT 1 (diagnostic) : l'état est initialisé avec {"count": 0}
            # et la télémétrie lit "count" — mais l'ancien code incrémentait
            # "no_trades" (jamais lu) : le compteur affiché restait à 0.
            event_log["count"] = event_log.get("count", 0) + 1
            # LOT 1 : breakdown par catégorie (observabilité décisionnelle).
            # FIX : dans `x[k] = y`, y est évalué AVANT le setdefault — écrire
            # le setdefault sur sa propre ligne (l'ancienne forme levait
            # KeyError avalée par le try/except : le breakdown ne se créait
            # jamais silencieusement).
            bucket = _no_trade_bucket(reason)
            reasons_map = event_log.setdefault("reasons", {})
            reasons_map[bucket] = reasons_map.get(bucket, 0) + 1
    except Exception:
        pass
    return True


def _no_trade_bucket(reason: str) -> str:
    """Catégorise une raison NO_TRADE pour l'agrégation (LOT 1)."""
    r = reason.lower()
    if "rr filter" in r or "asymétrie" in r or "rr " in r and "requis" in r:
        return "rr_filter"
    if "halt" in r:
        return "halt"
    if "cascade" in r:
        return "cascade"
    if "order flow" in r or "flux" in r or "toxic" in r or "agressif" in r:
        return "order_flow"
    if "meta-label" in r:
        return "meta_label"
    # LOT 2 (mandat) : raisons du Trade Opportunity Engine — AVANT "seuil"
    # (une raison OPP:EDGE_INSUFFICIENT contient aussi "< seuil" dans le préfixe)
    if "edge_insufficient" in r or "edge net" in r or "edge estimé" in r:
        return "edge_insufficient"
    # PHASE 3 C5 : gate de friction (coût AR attendu > seuil mesuré)
    if "friction" in r or "coût ar" in r:
        return "friction"
    # PHASE 4 P4-A : gate d'exposition factorielle (portefeuille trop exposé)
    if "portfolio_exposure" in r or "exposition nette" in r or "corrélée" in r:
        return "portfolio_exposure"
    if "uncalibrated" in r or "non calibrée" in r:
        return "uncalibrated"
    if "execution_risk" in r or "slippage attendu" in r:
        return "execution_risk"
    if "seuil" in r or "signal|" in r or "conviction" in r:
        return "conviction"
    if "régime" in r or "regime" in r or "moe" in r:
        return "regime_context"
    return "other"


def hedging_decision(symbol: str, positions: list[dict], corr_matrix: dict,
                     max_correlation: float = 0.75) -> dict | None:
    """
    VISION §4c: if the portfolio is over-concentrated in highly-correlated
    positions, suggest a hedge (opposite, smaller position) on the most
    correlated pair. Returns {hedge_symbol, hedge_side, reason} or None.
    """
    if len(positions) < 2 or not corr_matrix:
        return None
    # find the most correlated pair among open positions
    best = None
    best_corr = 0.0
    syms = [p.get("symbol") for p in positions if p.get("qty")]
    for i, a in enumerate(syms):
        for b in syms[i + 1:]:
            c = abs(float(corr_matrix.get(a, {}).get(b, 0.0) or 0.0))
            if c > best_corr:
                best_corr = c
                best = (a, b)
    if best is None or best_corr < max_correlation:
        return None
    # hedge the larger position with a small opposite on the other symbol
    pa = next((p for p in positions if p["symbol"] == best[0]), None)
    pb = next((p for p in positions if p["symbol"] == best[1]), None)
    if pa is None or pb is None:
        return None
    big = pa if abs(pa.get("qty", 0)) * (pa.get("avg_price") or 1) >= abs(pb.get("qty", 0)) * (pb.get("avg_price") or 1) else pb
    other = best[1] if big["symbol"] == best[0] else best[0]
    hedge_side = "SELL" if big.get("qty", 0) > 0 else "BUY"
    hedge_qty = abs(big.get("qty", 0)) * 0.20  # hedge 20% of the big position
    return {
        "hedge_symbol": other, "hedge_side": hedge_side, "hedge_qty": hedge_qty,
        "reason": f"correlation {best_corr:.2f} between {best[0]}/{best[1]} - hedging {big['symbol']}",
    }
