"""
Mode PAPER TRADING COMPLET (LOT 10)
- Place de vrais ordres sur l'exchange
- Calcule le PnL paper en temps réel
- Met à jour l'equity paper
- Aucune donnée fictive
"""
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger("PaperTrading")

class PaperTradingEngine:
    """
    Mode Paper Trading institutionnel complet.
    - Place de vrais ordres sur l'exchange
    - Calcule le PnL paper en temps réel
    - Met à jour l'equity paper
    - Aucune donnée fictive
    """
    
    def __init__(self, db, ccxt_client_getter):
        self.db = db
        self.get_ccxt = ccxt_client_getter
        self.paper_positions: Dict[str, Dict] = {}
        self.paper_balance = 100000.0
        self.paper_equity_history = [100000.0]
        self.total_paper_pnl = 0.0
        self.trades_count = 0

    def execute_paper_order(self, symbol: str, side: str, qty: float, price: float, mode: str = "PAPER") -> Dict:
        """Exécute un ordre en mode Paper avec calcul PnL complet"""
        client = self.get_ccxt()
        
        if client and mode == "PAPER":
            try:
                # Place un vrai ordre sur l'exchange (pour tester la connectivité)
                order = client.create_order(
                    symbol=symbol.replace("USDT", "/USDT"),
                    type='market',
                    side=side.lower(),
                    amount=qty
                )
                
                # Mise à jour des positions paper + PnL
                self._update_paper_position_and_pnl(symbol, side, qty, price)
                
                logger.info(f"PAPER ORDER PLACED + PNL UPDATED: {side} {qty} {symbol} @ {price}")
                
                return {
                    "success": True,
                    "order_id": order.get("id"),
                    "price": price,
                    "mode": "PAPER",
                    "real_order_placed": True,
                    "paper_equity": self.get_paper_equity({symbol: price})
                }
            except Exception as e:
                logger.error(f"Paper order failed: {e}")
                return {"success": False, "error": str(e)}
        
        # Mode DEMO pur
        self._update_paper_position_and_pnl(symbol, side, qty, price)
        return {
            "success": True,
            "order_id": f"paper_{int(time.time()*1000)}",
            "price": price,
            "mode": "DEMO",
            "paper_equity": self.get_paper_equity({symbol: price})
        }

    def _update_paper_position_and_pnl(self, symbol: str, side: str, qty: float, price: float):
        """Met à jour les positions paper et calcule le PnL réalisé"""
        if symbol not in self.paper_positions:
            self.paper_positions[symbol] = {"qty": 0.0, "avg_price": 0.0}
        
        pos = self.paper_positions[symbol]
        realized_pnl = 0.0
        
        if side == "BUY":
            if pos["qty"] < 0:  # Couverture d'une position courte
                covered = min(abs(pos["qty"]), qty)
                realized_pnl = covered * (pos["avg_price"] - price)
                self.total_paper_pnl += realized_pnl
            
            total_qty = pos["qty"] + qty
            if total_qty > 0:
                pos["avg_price"] = ((pos["qty"] * pos["avg_price"]) + (qty * price)) / total_qty
            pos["qty"] = total_qty
            
        else:  # SELL
            if pos["qty"] > 0:  # Vente d'une position longue
                sold = min(pos["qty"], qty)
                realized_pnl = sold * (price - pos["avg_price"])
                self.total_paper_pnl += realized_pnl
            
            pos["qty"] -= qty
            if pos["qty"] < 0:
                pos["qty"] = 0
        
        self.trades_count += 1
        self.paper_balance += realized_pnl
        
        # Mise à jour de l'equity
        current_equity = self.get_paper_equity({})
        self.paper_equity_history.append(current_equity)
        if len(self.paper_equity_history) > 200:
            self.paper_equity_history.pop(0)

    def get_paper_equity(self, current_prices: Dict[str, float]) -> float:
        """Calcule l'equity paper en temps réel"""
        equity = self.paper_balance
        for sym, pos in self.paper_positions.items():
            price = current_prices.get(sym, 0)
            if price > 0:
                equity += pos["qty"] * price
        return round(equity, 2)

    def get_paper_pnl_summary(self) -> Dict:
        """Retourne un résumé complet du PnL paper"""
        return {
            "paper_balance": round(self.paper_balance, 2),
            "total_paper_pnl": round(self.total_paper_pnl, 2),
            "trades_count": self.trades_count,
            "current_equity": self.get_paper_equity({}),
            "equity_history": self.paper_equity_history[-20:],  # 20 derniers points
            "open_positions": {
                sym: {"qty": pos["qty"], "avg_price": pos["avg_price"]}
                for sym, pos in self.paper_positions.items() if pos["qty"] != 0
            }
        }

    def reset_paper_account(self, new_balance: float = 100000.0):
        """Reset du compte paper"""
        self.paper_positions = {}
        self.paper_balance = new_balance
        self.paper_equity_history = [new_balance]
        self.total_paper_pnl = 0.0
        self.trades_count = 0
        logger.info(f"Paper account reset to ${new_balance:,.2f}")