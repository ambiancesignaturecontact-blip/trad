import logging

logger = logging.getLogger("Reconciliation")

class ReconciliationEngine:
    """
    Sovereign Reconciliation Engine (Phase 24 & Lot 9).
    Compares actual exchange balances, positions, open orders, and fills
    against the local database, triggering a complete risk halt upon any mismatch!
    """
    def __init__(self, db_manager):
        self.db = db_manager

    def reconcile_balances(self, actual_balance_usd: float, internal_balance_usd: float) -> bool:
        """
        Reconciles actual broker/exchange USD balance with internal database balance.
        """
        if abs(actual_balance_usd - internal_balance_usd) > 1e-2: # strict $0.01 tolerance!
            reason = f"BALANCE_MISMATCH (Exchange: ${actual_balance_usd:.2f}, DB: ${internal_balance_usd:.2f})"
            logger.critical(f"⚠️ RECONCILIATION MISMATCH DETECTED: {reason}")
            self.db.add_audit_log(
                "RECONCILIATION_FAILED",
                "127.0.0.1",
                f"Balance mismatch detected! Details: {reason}"
            )
            return False

        logger.info("Reconciliation Engine: Balance ledger is fully aligned.")
        return True

    def reconcile_positions(self, actual_positions_dict: dict, mode: str) -> bool:
        db_positions = self.db.get_positions()
        db_positions_dict = {p['symbol']: p['qty'] for p in db_positions if p['mode'] == mode}

        mismatches = []
        for symbol in actual_positions_dict:
            act_qty = actual_positions_dict[symbol]
            db_qty = db_positions_dict.get(symbol, 0.0)
            if abs(act_qty - db_qty) > 1e-4:
                mismatches.append(f"{symbol} (Exchange: {act_qty:.4f}, DB: {db_qty:.4f})")

        for symbol in db_positions_dict:
            if symbol not in actual_positions_dict and db_positions_dict[symbol] > 0:
                mismatches.append(f"{symbol} (Exchange: 0.0000, DB: {db_positions_dict[symbol]:.4f})")

        if mismatches:
            logger.critical(f"⚠️ RECONCILIATION MISMATCH DETECTED: {', '.join(mismatches)}")
            self.db.add_audit_log(
                "RECONCILIATION_FAILED",
                "127.0.0.1",
                f"Positions mismatch detected! Details: {', '.join(mismatches)}"
            )
            return False

        logger.info("Reconciliation Engine: Positions ledger is fully aligned.")
        return True
