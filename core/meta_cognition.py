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
                                  base_threshold: float = 0.15,
                                  min_threshold: float = 0.08, max_threshold: float = 0.30) -> float:
    """
    VISION §4a: threshold = base scaled by recent directional accuracy.
    If recent signals agreed with realized returns, trust them more (lower bar);
    if not, demand more evidence (higher bar).
    """
    n = min(len(recent_signals), len(recent_returns))
    if n < 10:
        return base_threshold
    correct = sum(1 for i in range(-n, 0) if np.sign(recent_signals[i]) == np.sign(recent_returns[i]))
    accuracy = correct / n
    # accuracy 0.5 -> neutral; >0.55 lowers the bar, <0.45 raises it
    factor = 1.0 - (accuracy - 0.5) * 1.5
    return float(np.clip(base_threshold * factor, min_threshold, max_threshold))


def decide_no_trade(symbol: str, signal: float, threshold: float, reasons: list[str],
                    event_log=None, db=None) -> bool:
    """
    VISION §4b: when the signal is below the conviction bar, log an explicit
    NO_TRADE decision with the reason instead of silently not trading.
    Returns True if the bot should abstain.
    """
    if abs(signal) >= threshold:
        return False
    reason = " | ".join(reasons) if reasons else f"|signal| {abs(signal):.3f} < threshold {threshold:.3f}"
    logger.info(f"⏸️ NO_TRADE {symbol}: {reason}")
    try:
        if db is not None:
            db.add_event(time.time(), "no_trade", f'{{"symbol": "{symbol}", "reason": "{reason}"}}')
        if event_log is not None:
            event_log["no_trades"] = event_log.get("no_trades", 0) + 1
    except Exception:
        pass
    return True


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
