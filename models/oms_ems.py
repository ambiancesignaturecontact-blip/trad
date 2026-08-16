import logging
import time

logger = logging.getLogger("OMS_EMS")

class OrderStatus:
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class Order:
    def __init__(self, symbol, side, qty, price, mode, strategy, client_order_id, order_type="MARKET"):
        self.order_id = None
        self.client_order_id = client_order_id
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.price = price
        self.mode = mode
        self.strategy = strategy
        self.order_type = order_type
        self.status = OrderStatus.PENDING
        self.filled_qty = 0.0
        self.average_filled_price = 0.0
        self.created_at = time.time()


class OrderManagementSystem:
    """
    Institutional Order Management System (OMS).
    Governs order states, tracks executions, and prevents database modifications
    before actual exchange confirmation.
    """
    def __init__(self, db_manager):
        self.db = db_manager
        self.orders_cache = {} # client_order_id -> Order

    def create_order(self, symbol, side, qty, price, mode, strategy, client_order_id) -> Order:
        order = Order(symbol, side, qty, price, mode, strategy, client_order_id)
        self.orders_cache[client_order_id] = order
        return order

    def update_order_status(self, client_order_id: str, status: str, filled_qty: float = 0.0, avg_price: float = 0.0):
        """
        Updates the order status. Only writes the position update to the database
        once a true 'FILLED' status receipt is received from the exchange.
        """
        if client_order_id not in self.orders_cache:
            return
            
        order = self.orders_cache[client_order_id]
        order.status = status
        order.filled_qty = filled_qty
        order.average_filled_price = avg_price
        
        # Persistent write
        order_id = self.db.add_order(
            symbol=order.symbol,
            side=order.side,
            price=avg_price if avg_price > 0 else order.price,
            qty=filled_qty if filled_qty > 0 else order.qty,
            status=status,
            mode=order.mode,
            strategy=order.strategy,
            order_type=order.order_type
        )
        order.order_id = order_id
        
        if status == OrderStatus.FILLED:
            # Only update position in DB upon confirmed fill!
            self.db.update_position(order.symbol, filled_qty, avg_price, order.mode)
            logger.info(f"OMS: Order {client_order_id} FILLED successfully. Database positions synchronized.")


class ReconciliationEngine:
    """
    Reconciliation Engine.
    Periodically compares actual broker/wallet balances and positions 
    against the internal database ledger, freezing trading if a mismatch is detected.
    """
    def __init__(self, db_manager):
        self.db = db_manager

    def reconcile_positions(self, actual_positions_dict: dict, mode: str) -> bool:
        """
        Compares actual exchange positions against internal database positions.
        actual_positions_dict: dict of symbol -> qty
        """
        db_positions = self.db.get_positions()
        db_positions_dict = {p['symbol']: p['qty'] for p in db_positions if p['mode'] == mode}
        
        mismatches = []
        for symbol in actual_positions_dict:
            act_qty = actual_positions_dict[symbol]
            db_qty = db_positions_dict.get(symbol, 0.0)
            
            if abs(act_qty - db_qty) > 1e-4: # Tolerance margin
                mismatches.append(f"{symbol} (Exchange: {act_qty:.4f}, DB: {db_qty:.4f})")
                
        # Check for ghost positions in DB
        for symbol in db_positions_dict:
            if symbol not in actual_positions_dict and db_positions_dict[symbol] > 0:
                mismatches.append(f"{symbol} (Exchange: 0.0000, DB: {db_positions_dict[symbol]:.4f})")
                
        if mismatches:
            logger.critical(f"⚠️ RECONCILIATION MISMATCH DETECTED: {', '.join(mismatches)}")
            self.db.add_audit_log(
                "RECONCILIATION_FAILED",
                "127.0.0.1",
                f"Positions mismatch detected! Freezing trading. Details: {', '.join(mismatches)}"
            )
            return False # Reconciliation failed
            
        logger.info("Reconciliation successful. Exchange and DB ledgers are fully aligned.")
        return True
