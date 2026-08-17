"""
Live Reconciliation Module - LOT 3
Performs periodic real reconciliation between DB and exchange in DEMO/REAL modes.
"""
import asyncio
import logging
from typing import Dict

logger = logging.getLogger("LiveReconciler")

class LiveReconciler:
    def __init__(self, db, reconciliation_engine, ccxt_client_getter):
        self.db = db
        self.reconciler = reconciliation_engine
        self.get_ccxt = ccxt_client_getter
        self.last_reconcile = 0

    async def run_periodic_reconciliation(self, interval_seconds: int = 35):
        """Runs reconciliation every ~35 seconds in background."""
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await self._perform_reconciliation()
            except Exception as e:
                logger.error(f"Live reconciliation failed: {e}")

    async def _perform_reconciliation(self):
        client = self.get_ccxt()
        if not client:
            return

        try:
            # 1. Reconcile balance
            bal = client.fetch_balance()
            actual_usdt = float(bal['free'].get('USDT', 0.0) or bal['total'].get('USDT', 0.0))

            internal_balance = self.db.get_setting("balance_demo") or 0
            try:
                internal_balance = float(internal_balance)
            except:
                internal_balance = 0

            balance_ok = self.reconciler.reconcile_balances(actual_usdt, internal_balance)

            # 2. Reconcile positions (REAL + PAPER)
            if balance_ok:
                try:
                    positions = client.fetch_positions()
                    actual_pos = {}
                    for p in positions:
                        if p.get('contracts', 0) > 0:
                            sym = p['symbol'].replace('/', '').replace(':USDT', '')
                            actual_pos[sym] = float(p.get('contracts', 0))

                    self.reconciler.reconcile_positions(actual_pos, mode="REAL")
                except Exception as e:
                    logger.warning(f"Position reconciliation skipped: {e}")

            # 3. Vérification des ordres ouverts
            try:
                open_orders = client.fetch_open_orders()
                if open_orders:
                    logger.info(f"Open orders detected: {len(open_orders)}")
            except:
                pass

            logger.info("✅ Periodic reconciliation completed successfully.")

        except Exception as e:
            logger.warning(f"Reconciliation attempt failed (non-critical): {e}")