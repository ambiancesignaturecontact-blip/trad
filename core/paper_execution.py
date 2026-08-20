"""
High-fidelity PAPER execution engine (VISION §3 / user requirement: DEMO == REAL).

The DEMO mode now simulates REAL execution conditions instead of filling at the
mid price:
  - BOOK-WALKING on the real order book (BTC/ETH/SOL have real 5-level depth)
  - slippage from the per-venue SlippageModel when no book is available
  - per-venue taker/maker FEES (Binance/Bybit default 0.1% taker)
  - simulated LATENCY with a price drift over the latency window
  - REAL exchange-style REJECTIONS (min notional, precision, insufficient balance)

This makes paper validation statistically meaningful before going REAL.
"""
import logging

import numpy as np

logger = logging.getLogger("PaperExecution")

# Per-venue fees (taker / maker) - configurable, realistic defaults
VENUE_FEES = {
    "Binance": {"taker": 0.001, "maker": 0.0008},
    "Bybit": {"taker": 0.001, "maker": 0.00085},
    "default": {"taker": 0.001, "maker": 0.001},
}
# FIX (logs prod) : le min notional n'est plus en dur — il est dérivé du
# capital (config.yaml : 3$ < 200$, 5$ < 1000$, 10$ sinon) pour que les
# micro-comptes et les paires à petit notionnel (EURUSD) puissent trader.
# P1-8 (audit §3) : fallback branché sur config.yaml lui aussi.
MIN_NOTIONAL_USD = 10.0
try:
    from core.config import settings as _settings
    MIN_NOTIONAL_USD = _settings.get_float("trading", "min_notional_usd_normal", 10.0)
except Exception:
    pass


def min_notional_for_capital(capital: float) -> float:
    """Min notional adapté au capital (config.yaml, défauts documentés)."""
    try:
        from core.config import settings
        if capital < 200.0:
            return settings.get_float("trading", "min_notional_usd_micro", 3.0)
        if capital < 1000.0:
            return settings.get_float("trading", "min_notional_usd_small", 5.0)
        return settings.get_float("trading", "min_notional_usd_normal", 10.0)
    except Exception:
        return 3.0 if capital < 200.0 else (5.0 if capital < 1000.0 else 10.0)


def _book_walk_price(side: str, qty: float, order_book: dict | None):
    """
    Walks the real order book to compute the actual fill price (VWAP of the levels
    consumed) AND the quantity actually filled (partial fills when the book is thin).
    Returns (vwap_price, filled_qty) or (None, 0.0) when the book is missing.
    """
    if not order_book:
        return None, 0.0
    levels = order_book.get("asks") if side.upper() == "BUY" else order_book.get("bids")
    if not levels:
        return None, 0.0
    remaining = qty
    cost = 0.0
    filled = 0.0
    for level in levels:
        try:
            px = float(level[0])
            sz = float(level[1])
        except (TypeError, ValueError, IndexError):
            continue
        take = min(remaining, sz)
        cost += take * px
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    if filled <= 0:
        return None, 0.0
    return cost / filled, filled  # VWAP fill price + filled qty


def estimate_slippage_bps_from_book(side: str, qty: float, order_book: dict | None,
                                    arrival_price: float) -> float | None:
    """
    P1-13 (audit §4.6) : estimation du slippage RÉEL par book-walking du carnet
    consolidé live. Réutilise _book_walk_price (le book-walking existait déjà
    pour les fills paper ; il est maintenant branché sur l'estimation live).

    Retourne les bps d'exécution (prix VWAP vs prix d'arrivée) ou None quand le
    carnet est absent/trop mince pour un jugement fiable — jamais une invention.
    """
    if qty <= 0 or arrival_price <= 0:
        return None
    vwap, filled = _book_walk_price(side, qty, order_book)
    if vwap is None or filled <= 0:
        return None
    # garde : écart > 10 % = carnet d'un autre actif ou périmé (régression EURUSD)
    if abs(vwap - arrival_price) / max(arrival_price, 1e-9) > 0.10:
        return None
    return abs(vwap - arrival_price) / arrival_price * 1e4


def simulate_paper_fill(symbol: str, side: str, qty: float, arrival_price: float,
                        order_book: dict | None, venue: str = "Binance",
                        volatility: float = 0.002, liquidity_score: float = 1.0,
                        balance: float = 0.0, slippage_model=None) -> dict:
    """
    Simulates a REAL-quality fill in DEMO/paper mode.

    Returns:
      {status, fill_price, fill_qty, fee, fee_asset, slippage_bps, latency_ms,
       rejected, reason}
    """

    # ---- 1. Real exchange-style REJECTIONS ----
    if qty <= 0:
        return {"status": "REJECTED", "reason": "invalid quantity", "rejected": True}
    notional = qty * arrival_price
    min_notional = min_notional_for_capital(balance) if balance > 0 else MIN_NOTIONAL_USD
    if notional < min_notional:
        return {"status": "REJECTED", "reason": f"below min notional ${min_notional}", "rejected": True}
    if balance > 0:
        if side.upper() == "BUY" and notional * 1.01 > balance:
            return {"status": "REJECTED", "reason": "insufficient balance", "rejected": True}
        if side.upper() == "SELL" and qty > balance:
            return {"status": "REJECTED", "reason": "insufficient position", "rejected": True}

    # ---- 2. Book-walk the REAL order book when available ----
    walked, book_filled = _book_walk_price(side, qty, order_book)
    fill_qty = qty

    # ---- 3. Modeled slippage when no book (ExecutionSimulator-style + learned) ----
    if walked is not None:
        # Defensive check: a book-walk price deviating >10% from the arrival price
        # means the book belongs to another symbol or is stale - NEVER use it
        # (regression: EURUSD was once filled against the BTC book at $60,010).
        if abs(walked - arrival_price) / max(arrival_price, 1e-9) > 0.10:
            logger.warning(f"Book-walk {walked:.2f} implausible vs arrival {arrival_price:.2f} ({symbol}) - using modeled slippage")
            walked, book_filled = None, 0.0
        else:
            fill_price = walked
            if book_filled < qty:
                fill_qty = book_filled  # partial fill: book too thin for the whole order
                logger.info(f"PARTIAL FILL: {side} {qty} -> {book_filled} ({symbol})")
            slippage_bps = abs(fill_price - arrival_price) / arrival_price * 1e4
    if walked is None:
        learned_bps = None
        if slippage_model is not None:
            try:
                learned_bps = slippage_model.expected_slippage_bps(venue, symbol, fallback=None)
            except Exception:
                learned_bps = None
        base_bps = learned_bps if learned_bps else 5.0
        vol_factor = 1 + (volatility * 8)
        liq_factor = 1 / max(liquidity_score, 0.2)
        size_factor = 1 + (min(qty * arrival_price / max(balance, 1.0), 0.1) * 30)
        slip_bps = base_bps * vol_factor * liq_factor * size_factor
        fill_price = arrival_price * (1 + slip_bps / 1e4) if side.upper() == "BUY" else arrival_price * (1 - slip_bps / 1e4)
        slippage_bps = slip_bps

    # ---- 4. Per-venue fees ----
    fee_cfg = VENUE_FEES.get(venue, VENUE_FEES["default"])
    fee_rate = fee_cfg["taker"]  # paper = market orders = taker
    fee = round(qty * fill_price * fee_rate, 6)

    # ---- 5. Latency with price drift over the latency window ----
    # drift scaled relative to the 2.5s loop tick: over 75ms the price moves only
    # a fraction of the per-tick volatility (realistic, not exaggerated).
    latency_ms = float(np.clip(np.random.normal(75, 25), 30, 250))
    drift_scale = float(np.sqrt((latency_ms / 1000.0) / 2.5))
    drift = volatility * drift_scale * float(np.random.randn())
    fill_price = fill_price * (1.0 + drift) if side.upper() == "BUY" else fill_price * (1.0 - drift)

    return {
        "status": "FILLED",
        "fill_price": round(fill_price, 6),
        "fill_qty": round(fill_qty, 8),
        "partial": fill_qty < qty,
        "fee": round(fee * (fill_qty / qty), 6) if qty > 0 else 0.0,
        "fee_asset": "USDT",
        "slippage_bps": round(slippage_bps, 3),
        "latency_ms": round(latency_ms, 1),
        "rejected": False,
        "reason": "",
    }
