"""
COMPTABILITÉ DES COÛTS RÉELS (PROMPT MAÎTRE, Pilier O).

Mentalité n°2 : l'edge est NET des coûts. Le bot doit connaître son coût
RÉEL par trade et le retrancher du PnL affiché :

  coût_total = frais (taker/maker) + slippage réalisé + impact + gas (DEX)
               + coût de financement (portage)

Ce module :
  1. Enregistre chaque trade avec son coût total (par symbole, side, style).
  2. Calcule le coût de PORTAGE (funding) des positions tenues — une position
     longue peut être perdante NET même si le prix monte.
  3. Expose le coût cumulé (télémétrie) pour un PnL NET honnête.
"""
import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger("CostAccounting")

# Défauts réalistes (documentés, ajustables)
DEFAULT_FEE_RATE = 0.001       # 0.1% taker
DEFAULT_SLIPPAGE_BPS = 5.0     # 5 bps moyen


class CostAccounting:
    """Trace les coûts réels par trade et le coût de portage."""

    def __init__(self):
        self.trade_costs: List[dict] = []      # détail par trade
        self.carry_cost: Dict[str, float] = {} # coût de portage par symbole
        self.total_costs_usd: float = 0.0
        self._carry_log: List[dict] = []

    # ------------------------------------------------------------------ #
    # 1. COÛT TOTAL PAR TRADE
    # ------------------------------------------------------------------ #
    def record_trade_cost(self, symbol: str, side: str, qty: float,
                          price: float, fee_rate: float = DEFAULT_FEE_RATE,
                          slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
                          impact_bps: float = 0.0, gas_usd: float = 0.0,
                          funding_cost_usd: float = 0.0, style: str = "market",
                          venue: str = "Binance") -> dict:
        """
        Enregistre le coût total d'un trade :
        frais (qty*price*fee_rate) + slippage (qty*price*slip/1e4) +
        impact + gas + funding. Retourne le détail.
        """
        notional = abs(qty) * price
        fee = notional * fee_rate
        slippage = notional * slippage_bps / 1e4
        impact = notional * impact_bps / 1e4
        total = fee + slippage + impact + gas_usd + funding_cost_usd

        rec = {
            "symbol": symbol, "side": side, "qty": qty, "price": price,
            "notional_usd": round(notional, 2),
            "fee_usd": round(fee, 4),
            "slippage_usd": round(slippage, 4),
            "impact_usd": round(impact, 4),
            "gas_usd": round(gas_usd, 4),
            "funding_usd": round(funding_cost_usd, 4),
            "total_cost_usd": round(total, 4),
            "total_cost_bps": round(total / notional * 1e4, 2) if notional > 0 else 0.0,
            "style": style, "venue": venue, "ts": time.time(),
        }
        self.trade_costs.append(rec)
        if len(self.trade_costs) > 2000:
            self.trade_costs = self.trade_costs[-2000:]
        self.total_costs_usd += total
        return rec

    # ------------------------------------------------------------------ #
    # 2. COÛT DE PORTAGE (FUNDING) DES POSITIONS TENUES
    # ------------------------------------------------------------------ #
    def record_carry_cost(self, symbol: str, position_notional_usd: float,
                          funding_rate_8h: Optional[float],
                          hold_hours: float = 8.0) -> float:
        """
        Coût de portage : pour une position tenue, le funding (perp) est un
        coût (ou un gain) qui s'accumule. funding_rate_8h > 0 = les longs
        PAIENT (coût positif), les shorts reçoivent.
        Retourne le coût en USD (positif = coût).
        """
        if not funding_rate_8h or position_notional_usd <= 0:
            return 0.0
        periods = max(hold_hours / 8.0, 0.0)
        cost = position_notional_usd * abs(funding_rate_8h) * periods
        # Sens : long + funding positif = coût ; short + funding positif = gain
        signed = cost  # simplifié : on trace la magnitude comme coût potentiel
        self.carry_cost[symbol] = self.carry_cost.get(symbol, 0.0) + signed
        self._carry_log.append({
            "symbol": symbol, "notional_usd": round(position_notional_usd, 2),
            "funding_rate_8h": funding_rate_8h, "hold_hours": hold_hours,
            "carry_cost_usd": round(signed, 4), "ts": time.time(),
        })
        if len(self._carry_log) > 1000:
            self._carry_log = self._carry_log[-1000:]
        return signed

    def apply_funding_to_position(self, symbol: str, position_qty: float,
                                  price: float, funding_rate_8h: Optional[float]) -> float:
        """Applique le funding au ledger (position tenue depuis le dernier tick)."""
        notional = abs(position_qty) * price
        return self.record_carry_cost(symbol, notional, funding_rate_8h, hold_hours=8.0)

    # ------------------------------------------------------------------ #
    # 3. EXPOSITION (TÉLÉMÉTRIE + PNL NET)
    # ------------------------------------------------------------------ #
    def net_pnl(self, gross_pnl_usd: float) -> float:
        """PnL NET : le PnL brut moins TOUS les coûts enregistrés."""
        return gross_pnl_usd - self.total_costs_usd

    def to_dict(self) -> dict:
        return {
            "total_costs_usd": round(self.total_costs_usd, 2),
            "n_trades_costed": len(self.trade_costs),
            "last_trade_cost": self.trade_costs[-1] if self.trade_costs else None,
            "carry_costs_by_symbol": {k: round(v, 4) for k, v in self.carry_cost.items()},
            "avg_cost_bps": (sum(t["total_cost_bps"] for t in self.trade_costs)
                             / len(self.trade_costs) if self.trade_costs else 0.0),
        }

    def recent_costs(self, limit: int = 20) -> List[dict]:
        return self.trade_costs[-limit:]
