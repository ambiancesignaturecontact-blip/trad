"""
Position protection manager (audit B7-1 / C1) - STOP-LOSS / TAKE-PROFIT / TRAILING.

Pure, testable decision logic + persistent level storage. Integrated in the
trading loop BEFORE any new signal so a position is always protected first
(atomicité, Pilier G du PROMPT MAÎTRE).

LOT 2 (PDF Faille 3 / Pilier F) : les niveaux SL/TP sont dérivés de la SOURCE
UNIQUE DE VÉRITÉ REWARD_RISK_RATIO (core.risk_pipeline) pour que le RR réel
des stops soit identique au RR utilisé par le sizing Kelly. Plus jamais de
stops à RR 1.75-2.0 pendant que le Kelly utilise 1.5.
"""
import logging
import time

from core.risk_pipeline import ATR_MULT_SL, REWARD_RISK_RATIO, STOP_LOSS_PCT

logger = logging.getLogger("PositionManager")


class PositionProtection:
    """
    Per-symbol protection state: entry price, SL/TP levels and trailing high-water.
    Levels are derived from ATR or fixed percentages of the entry price.
    TP = SL × REWARD_RISK_RATIO (source unique, Pilier F exigence 2).
    """

    def __init__(self, symbol: str, entry_price: float, qty: float,
                 stop_loss_pct: float = None, take_profit_pct: float = None,
                 trailing_pct: float = 0.0, atr: float | None = None,
                 atr_mult_sl: float = None, atr_mult_tp: float = None):
        self.symbol = symbol
        self.entry_price = float(entry_price)
        self.qty = float(qty)
        self.trailing_pct = float(trailing_pct)

        # Défauts alignés sur la config institutionnelle (LOT 2)
        stop_loss_pct = STOP_LOSS_PCT if stop_loss_pct is None else stop_loss_pct
        atr_mult_sl = ATR_MULT_SL if atr_mult_sl is None else atr_mult_sl
        # TP dérivé du RR unifié (source unique de vérité)
        if take_profit_pct is None:
            take_profit_pct = stop_loss_pct * REWARD_RISK_RATIO
        if atr_mult_tp is None:
            atr_mult_tp = atr_mult_sl * REWARD_RISK_RATIO

        # ATR-based levels when ATR is available (institutional default)
        if atr and atr > 0:
            self.stop_price = float(entry_price - atr_mult_sl * atr)   # long SL below
            self.take_price = float(entry_price + atr_mult_tp * atr)   # long TP above
        else:
            self.stop_price = float(entry_price * (1.0 - stop_loss_pct))
            self.take_price = float(entry_price * (1.0 + take_profit_pct))

        self.high_water = float(entry_price)   # trailing reference (long side)
        self.updated_ts = time.time()
        # LOT 6 (PDF Pilier M) : cycle de vie de la position
        self.entry_ts = time.time()            # horodatage d'ouverture (time stop)
        self.breakeven_done = False            # breakeven stop déjà appliqué
        self.tp1_hit = False                   # 1er palier de take-profit touché

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "entry_price": self.entry_price, "qty": self.qty,
            "stop_price": round(self.stop_price, 6), "take_price": round(self.take_price, 6),
            "trailing_pct": self.trailing_pct, "high_water": round(self.high_water, 6),
            "updated_ts": self.updated_ts,
            "entry_ts": self.entry_ts,             # LOT 6 : time stop
            "breakeven_done": self.breakeven_done, # LOT 6 : breakeven
            "tp1_hit": self.tp1_hit,               # LOT 6 : scaling out
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


# --------------------------------------------------------------------------- #
# LOT 6 (PDF Pilier M) : CYCLE DE VIE DES POSITIONS
# --------------------------------------------------------------------------- #
def position_age_hours(prot: PositionProtection, now: float = None) -> float:
    """Âge de la position en heures (depuis l'ouverture)."""
    now = now or time.time()
    return max(0.0, (now - prot.entry_ts) / 3600.0)


def evaluate_time_stop(prot: PositionProtection, current_price: float,
                       position_qty: float, max_age_hours: float = 24.0,
                       min_profit_pct: float = 0.001) -> str:
    """
    TIME STOP (PDF Pilier M, exigence 3) : si l'idée n'a pas produit le
    mouvement attendu après X heures, on sort — le capital immobilisé a un
    coût d'opportunité (mentalité n°17 : on abandonne sans attachement).
    Retourne "TIME_STOP" ou "HOLD".
    """
    if position_qty == 0 or current_price <= 0 or prot.entry_price <= 0:
        return "HOLD"
    if position_age_hours(prot) < max_age_hours:
        return "HOLD"
    # PnL actuel de la position
    direction = 1.0 if position_qty > 0 else -1.0
    pnl_pct = (current_price - prot.entry_price) / prot.entry_price * direction
    if pnl_pct < min_profit_pct:
        return "TIME_STOP"
    return "HOLD"


def apply_breakeven_stop(prot: PositionProtection, current_price: float,
                         position_qty: float, trigger_pct: float = 0.02) -> bool:
    """
    BREAKEVEN STOP (PDF Pilier M, exigence 4) : dès que le trade est en gain
    SIGNIFICATIF (>= trigger_pct, défaut 2 %), on remonte le stop au prix
    d'entrée pour transformer le trade en « risque zéro » (mentalité n°1 :
    survivre d'abord). Retourne True si le stop a été remonté.
    """
    if position_qty == 0 or current_price <= 0 or prot.breakeven_done:
        return False
    direction = 1.0 if position_qty > 0 else -1.0
    pnl_pct = (current_price - prot.entry_price) / prot.entry_price * direction
    if pnl_pct >= trigger_pct:
        prot.stop_price = float(prot.entry_price)
        prot.breakeven_done = True
        return True
    return False


def partial_take_profit(prot: PositionProtection, current_price: float,
                        position_qty: float, tp1_fraction: float = 0.5,
                        exit_fraction: float = 0.5) -> dict:
    """
    SCALING OUT (PDF Pilier M, exigence 1) : on sécurise progressivement.
    Au 1er palier (tp1_fraction de la distance TP, défaut 50 %), on sort
    exit_fraction (défaut 50 %) de la position.

    Retourne {"action": "PARTIAL_TP", "exit_qty": X, "remain_qty": Y} si le
    palier est touché (une seule fois), sinon {"action": "HOLD"}.
    """
    if position_qty == 0 or current_price <= 0 or prot.tp1_hit:
        return {"action": "HOLD"}
    entry = prot.entry_price
    tp1 = entry + (prot.take_price - entry) * tp1_fraction
    tp1_short = entry - (entry - prot.take_price) * tp1_fraction
    hit = False
    if position_qty > 0 and current_price >= tp1:
        hit = True
    elif position_qty < 0 and current_price <= tp1_short:
        hit = True
    if hit:
        prot.tp1_hit = True
        exit_qty = abs(position_qty) * exit_fraction
        remain_qty = abs(position_qty) - exit_qty
        return {"action": "PARTIAL_TP", "exit_qty": exit_qty,
                "remain_qty": remain_qty, "price": current_price}
    return {"action": "HOLD"}


def can_pyramid(prot: PositionProtection, current_price: float,
                position_qty: float, reward_risk: float,
                min_rr: float = 1.5, max_additions: int = 2,
                additions: int = 0) -> tuple:
    """
    PYRAMIDING CONTRÔLÉ (PDF Pilier M, exigence 2) :
      - on ajoute UNIQUEMENT sur les GAGNANTS (position en profit) ;
      - et seulement si le RR reste FAVORABLE (le nouvel ajout conserve un RR
        >= min_rr par rapport au stop actuel) ;
      - jamais plus de max_additions ;
      - INTERDICTION du « moyenne à la baisse » : si la position est PERDANTE,
        retourne False (mentalité n°12 : couper les perdants, laisser courir
        les gagnants).
    Retourne (autorisation, raison).
    """
    if additions >= max_additions:
        return False, f"pyramiding: max {max_additions} ajouts atteint"
    if position_qty == 0 or current_price <= 0:
        return False, "pyramiding: pas de position"
    direction = 1.0 if position_qty > 0 else -1.0
    pnl_pct = (current_price - prot.entry_price) / prot.entry_price * direction
    if pnl_pct <= 0:
        # Moyenne à la baisse INTERDITE (mentalité n°12)
        return False, f"pyramiding: position perdante ({pnl_pct*100:+.2f}%) -> jamais d'ajout"
    # RR du nouvel ajout : distance au TP restant vs distance au stop
    dist_to_tp = abs(prot.take_price - current_price)
    dist_to_stop = abs(current_price - prot.stop_price)
    if dist_to_stop <= 0:
        return False, "pyramiding: stop au prix (risque nul) -> pas d'ajout"
    rr_now = dist_to_tp / dist_to_stop
    if rr_now < min_rr:
        return False, f"pyramiding: RR {rr_now:.2f} < {min_rr} requis"
    return True, f"pyramiding: gagnant ({pnl_pct*100:+.2f}%), RR {rr_now:.2f}"


class PositionProtectionStore:
    """Persists protection plans in STATE (JSON-serializable) so they survive ticks."""

    def __init__(self, state: dict):
        self.state = state
        self.state.setdefault("position_protections", {})  # symbol -> dict

    def get(self, symbol: str) -> PositionProtection | None:
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
            # LOT 6 : restaurer le cycle de vie
            p.entry_ts = float(d.get("entry_ts", p.updated_ts or time.time()))
            p.breakeven_done = bool(d.get("breakeven_done", False))
            p.tp1_hit = bool(d.get("tp1_hit", False))
            return p
        except Exception as e:
            logger.warning(f"Protection load failed for {symbol}: {e}")
            return None

    def upsert(self, prot: PositionProtection) -> None:
        self.state["position_protections"][prot.symbol] = prot.to_dict()

    def remove(self, symbol: str) -> None:
        self.state["position_protections"].pop(symbol, None)

    def all(self) -> dict[str, dict]:
        return dict(self.state["position_protections"])
