"""
LOT 52: Complete Trade Journal with Notes + Screenshots
"""

import json
import logging
import os
from datetime import datetime

import pandas as pd

logger = logging.getLogger("TradeJournal")

class TradeJournal:
    """
    Professional Trade Journal supporting:
    - Full trade details
    - Notes
    - Screenshot paths
    - Search & filtering
    - Export to CSV / JSON
    """

    def __init__(self, db_path: str = "trade_journal.json"):
        self.db_path = db_path
        self.trades: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path) as f:
                    self.trades = json.load(f)
                logger.info(f"Trade Journal loaded: {len(self.trades)} trades")
            except Exception as e:
                logger.error(f"Failed to load trade journal: {e}")

    def _save(self):
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.trades, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save trade journal: {e}")

    def add_trade(self,
                  symbol: str,
                  side: str,
                  qty: float,
                  price: float,
                  mode: str,
                  strategy: str = "META_MODEL",
                  notes: str = "",
                  screenshot_path: str | None = None,
                  realized_pnl: float | None = None) -> dict:

        trade = {
            "id": len(self.trades) + 1,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": side.upper(),
            "qty": round(qty, 6),
            "price": round(price, 4),
            "mode": mode,
            "strategy": strategy,
            "notes": notes,
            "screenshot_path": screenshot_path,
            "realized_pnl": realized_pnl
        }

        self.trades.append(trade)
        self._save()

        logger.info(f"[JOURNAL] Trade #{trade['id']} logged: {side} {qty} {symbol} @ {price}")
        return trade

    def get_trades(self, symbol: str | None = None, limit: int = 50) -> list[dict]:
        filtered = self.trades
        if symbol:
            filtered = [t for t in filtered if t["symbol"] == symbol]
        return filtered[-limit:][::-1]

    def search(self, keyword: str) -> list[dict]:
        keyword = keyword.lower()
        return [t for t in self.trades if keyword in str(t).lower()]

    def export_to_csv(self, filepath: str = "trade_journal_export.csv"):
        if not self.trades:
            return False
        df = pd.DataFrame(self.trades)
        df.to_csv(filepath, index=False)
        logger.info(f"Trade Journal exported to {filepath}")
        return True

    def get_summary(self) -> dict:
        if not self.trades:
            return {"total_trades": 0}

        df = pd.DataFrame(self.trades)
        total_pnl = df["realized_pnl"].sum() if "realized_pnl" in df.columns else 0

        return {
            "total_trades": len(self.trades),
            "realized_pnl": round(total_pnl, 2),
            "buy_count": len(df[df["side"] == "BUY"]),
            "sell_count": len(df[df["side"] == "SELL"]),
            "avg_trade_size": round(df["qty"].mean(), 4) if len(df) > 0 else 0
        }
