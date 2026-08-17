"""
Position protection manager (audit B7-1 / C1) - STOP-LOSS / TAKE-PROFIT / TRAILING.

Pure, testable decision logic + persistent level storage. Integrated in the
trading loop BEFORE any new signal so a position is always protected first.
"""
import json
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger("PositionManager")


class PositionProtection:
    """
    Per-symbol protection state: entry price, SL/TP levels and trailing high-water.
    Levels are derived from ATR or fixed percentages of the entry price.
    """

    def __init__(self, symbol: str, entry_price: float, qty: float,
                 stop_loss_pct: float = 0.03, take_profit_pct: float = 0.06,
                 trailing_pct: float = 0.0, atr: Optional[float] = None,
                 atr_mult_sl: float = 2.0, atr_mult_tp: float = 3.5):
        self.symbol = symbol
        self.entry_price = float(entry_price)
        self.qty = float(qty)
        self.trailing_pct = float(trailing_pct)

        # ATR-based levels when ATR is available (institutional default)
        if atr and atr > 0:
            self.stop_price = float(entry_price - atr_mult_sl * atr)   # long SL below
            self.take_price = float(entry_price + atr_mult_tp * atr)   # long TP above
        else:
            self.stop_price = float(entry_price * (1.0 - stop_loss_pct))
            self.take_price = float(entry_price * (1.0 + take_profit_pct))

        self.high_water = float(entry_price)   # trailing reference (long side)
        self.updated_ts = time.time()

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "entry_price": self.entry_price, "qty": self.qty,
            "stop_price": round(self.stop_price, 6), "take_price": round(self.take_price, 6),
            "trailing_pct": self.trailing_pct, "high_water": round(self.high_water, 6),
            "updated_ts": self.updated_ts,
        }


def evaluate_protection(prot: PositionProtection, current_price: float, position_qty: float) -> str:
    """
    Returns the protection action for the current tick:
      - "HOLD"      : nothing triggered
      - "STOP_LOSS": current price broke the stop level
      - "TAKE_PROFIT": current price hit the take-profit level
    Also updates the trailing high-water mark (long side) when trailing is active.
    """
    if position_qty == 0:
        return "HOLD"
    if current_price <= 0:
        return "HOLD"

    # Trailing stop: ratchet the high-water up, and pull the stop with it.
    if prot.trailing_pct > 0:
        if current_price > prot.high_water:
            prot.high_water = current_price
            trailing_stop = prot.high_water * (1.0 - prot.trailing_pct)
            if trailing_stop > prot.stop_price:
                prot.stop_price = trailing_stop
        # Short-side mirror (qty negative) handled by symmetry below via price logic

    if position_qty > 0:  # long position
        if current_price <= prot.stop_price:
            return "STOP_LOSS"
        if current_price >= prot.take_price:
            return "TAKE_PROFIT"
    else:  # short position
        if current_price >= prot.stop_price:
            return "STOP_LOSS"
        if current_price <= prot.take_price:
            return "TAKE_PROFIT"
    return "HOLD"


class PositionProtectionStore:
    """Persists protection plans in STATE (JSON-serializable) so they survive ticks."""

    def __init__(self, state: Dict):
        self.state = state
        self.state.setdefault("position_protections", {})  # symbol -> dict

    def get(self, symbol: str) -> Optional[PositionProtection]:
        d = self.state["position_protections"].get(symbol)
        if not d:
            return None
        try:
            p = PositionProtection(symbol, d["entry_price"], d["qty"])
            p.stop_price = float(d["stop_price"])
            p.take_price = float(d["take_price"])
            p.trailing_pct = float(d.get("trailing_pct", 0.0))
            p.high_water = float(d.get("high_water", d["entry_price"]))
            p.updated_ts = float(d.get("updated_ts", 0.0))
            return p
        except Exception as e:
            logger.warning(f"Protection load failed for {symbol}: {e}")
            return None

    def upsert(self, prot: PositionProtection) -> None:
        self.state["position_protections"][prot.symbol] = prot.to_dict()

    def remove(self, symbol: str) -> None:
        self.state["position_protections"].pop(symbol, None)

    def all(self) -> Dict[str, dict]:
        return dict(self.state["position_protections"])
