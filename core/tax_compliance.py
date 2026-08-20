"""
LOT 58: Tax & Compliance Reporting Engine (FIFO, Cost Basis, Realized PnL)
"""

import json
import logging
import os
from collections import deque
from datetime import datetime

import pandas as pd

logger = logging.getLogger("TaxCompliance")

class TaxComplianceEngine:
    """
    LOT 58: Professional tax and compliance reporting.
    Uses strict FIFO method.
    """

    def __init__(self, journal_path: str = "tax_journal.json"):
        self.journal_path = journal_path
        self.positions: dict[str, deque] = {}          # symbol -> FIFO queue
        self.realized_trades: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.journal_path):
            try:
                with open(self.journal_path) as f:
                    data = json.load(f)
                    self.realized_trades = data.get("realized_trades", [])
                    # Rebuild positions from trades if needed (simplified)
            except Exception as e:
                logger.error(f"Failed to load tax journal: {e}")

    def _save(self):
        try:
            with open(self.journal_path, "w") as f:
                json.dump({
                    "realized_trades": self.realized_trades,
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save tax journal: {e}")

    def record_trade(self, symbol: str, side: str, qty: float, price: float, timestamp: datetime = None):
        if timestamp is None:
            timestamp = datetime.now()

        if symbol not in self.positions:
            self.positions[symbol] = deque()

        if side.upper() == "BUY":
            self.positions[symbol].append({
                "qty": qty,
                "price": price,
                "timestamp": timestamp
            })
        else:
            self._process_sell(symbol, qty, price, timestamp)

    def _process_sell(self, symbol: str, sell_qty: float, sell_price: float, sell_time: datetime):
        if symbol not in self.positions or not self.positions[symbol]:
            logger.warning(f"No open position for {symbol}")
            return

        remaining = sell_qty
        total_cost = 0.0
        realized_pnl = 0.0

        while remaining > 0 and self.positions[symbol]:
            lot = self.positions[symbol][0]
            match_qty = min(remaining, lot["qty"])

            cost = lot["price"] * match_qty
            pnl = (sell_price - lot["price"]) * match_qty

            total_cost += cost
            realized_pnl += pnl

            lot["qty"] -= match_qty
            remaining -= match_qty

            if lot["qty"] <= 0:
                self.positions[symbol].popleft()

        trade_record = {
            "id": len(self.realized_trades) + 1,
            "symbol": symbol,
            "side": "SELL",
            "qty": sell_qty,
            "price": sell_price,
            "cost_basis": round(total_cost, 2),
            "realized_pnl": round(realized_pnl, 2),
            "timestamp": sell_time.isoformat()
        }

        self.realized_trades.append(trade_record)
        self._save()

        logger.info(f"[TAX] Realized PnL on {symbol}: ${realized_pnl:.2f}")

    def get_cost_basis(self, symbol: str) -> float:
        if symbol not in self.positions:
            return 0.0
        return sum(lot["qty"] * lot["price"] for lot in self.positions[symbol])

    def get_realized_pnl(self, symbol: str | None = None) -> float:
        if symbol:
            return sum(t["realized_pnl"] for t in self.realized_trades if t["symbol"] == symbol)
        return sum(t["realized_pnl"] for t in self.realized_trades)

    def generate_tax_report(self, year: int | None = None) -> pd.DataFrame:
        if not self.realized_trades:
            return pd.DataFrame()

        df = pd.DataFrame(self.realized_trades)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if year:
            df = df[df["timestamp"].dt.year == year]

        df["tax_category"] = df["timestamp"].apply(
            lambda x: "Long-term" if (datetime.now() - x).days > 365 else "Short-term"
        )

        return df[["id", "symbol", "qty", "price", "cost_basis", "realized_pnl", "tax_category", "timestamp"]]

    def get_summary(self) -> dict:
        if not self.realized_trades:
            return {"total_realized_pnl": 0, "total_trades": 0}

        df = pd.DataFrame(self.realized_trades)
        return {
            "total_realized_pnl": round(df["realized_pnl"].sum(), 2),
            "total_trades": len(df),
            "short_term_pnl": round(df[df["timestamp"].apply(lambda x: (datetime.now() - pd.to_datetime(x)).days <= 365)]["realized_pnl"].sum(), 2),
            "long_term_pnl": round(df[df["timestamp"].apply(lambda x: (datetime.now() - pd.to_datetime(x)).days > 365)]["realized_pnl"].sum(), 2)
        }
