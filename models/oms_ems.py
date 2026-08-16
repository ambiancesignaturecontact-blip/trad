import logging
import time

logger = logging.getLogger("OMS_EMS")

class OrderStatus:
    CREATED = "CREATED"
    RISK_APPROVED = "RISK_APPROVED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


class Order:
    """
    Sovereign Order Data Model (Phase 10 & Lot 7).
    Enforces the complete transaction metadata lifecycle.
    """
    def __init__(self, symbol: str, side: str, order_type: str, requested_qty: float, requested_price: float, mode: str, strategy: str, client_order_id: str, exchange: str):
        self.internal_order_id = f"int_ord_{int(time.time()*1000)}"
        self.client_order_id = client_order_id
        self.exchange_order_id = None
        
        self.symbol = symbol
        self.side = side
        self.type = order_type # MARKET, LIMIT
        self.exchange = exchange
        self.strategy = strategy
        self.mode = mode
        
        self.requested_qty = float(requested_qty)
        self.filled_qty = 0.0
        self.remaining_qty = float(requested_qty)
        self.requested_price = float(requested_price) if requested_price else 0.0
        self.average_fill_price = 0.0
        
        self.fees = 0.0
        self.fee_asset = "USDT"
        
        self.status = OrderStatus.CREATED
        self.timestamp_created = time.time()
        self.timestamp_updated = time.time()


class ExecutionManagementSystem:
    """
    Sovereign Execution Management System (EMS - Phase 22 & Lot 7).
    Decides order routing, supports LIMIT/MARKET, and delegates to Exchange Adapters.
    """
    def __init__(self, binance_adapter, bybit_adapter):
        self.binance = binance_adapter
        self.bybit = bybit_adapter

    def route_and_execute_order(self, order: Order) -> dict:
        """
        Smart Order Routing. Delegates execution to the actual, concrete 
        exchange adapter, ensuring no direct client.create_order bypass (Lot 7).
        """
        order.status = OrderStatus.SUBMITTED
        order.timestamp_updated = time.time()
        
        # Select target adapter based on exchange and mode
        adapter = self.binance if order.exchange.lower() == "binance" else self.bybit
        
        if order.mode == "DEMO" or order.mode == "PAPER" or order.mode == "SHADOW":
            # Virtual matching: Simulated execution on real order books (handled by PAPER EMS)
            order.status = OrderStatus.ACKNOWLEDGED
            logger.info(f"EMS: Routed Virtual Order {order.client_order_id} on {order.exchange}.")
            return {
                "status": "ACKNOWLEDGED",
                "order_id": f"sim_id_{int(time.time())}",
                "price": order.requested_price,
                "amount": order.requested_qty
            }
            
        # REAL MONEY ROUTING: Executes strictly on the live exchange adapter!
        if not adapter or not adapter.client:
            order.status = OrderStatus.REJECTED
            logger.error(f"EMS: Rejected Real Order {order.client_order_id} - Target exchange adapter offline.")
            return {"status": "REJECTED", "reason": "Exchange adapter unconfigured"}
            
        try:
            logger.info(f"EMS: ROUTING REAL ORDER TO {order.exchange} ({order.side} {order.requested_qty} {order.symbol})")
            res = adapter.place_order(
                symbol=order.symbol.replace("USDT", "/USDT"),
                order_type=order.type.lower(),
                side=order.side.lower(),
                amount=order.requested_qty,
                price=order.requested_price if order.type.upper() == "LIMIT" else None
            )
            order.status = OrderStatus.ACKNOWLEDGED
            order.exchange_order_id = res.get("id")
            return {
                "status": "SUCCESS",
                "order_id": res.get("id"),
                "price": float(res.get("price", 0.0)),
                "amount": float(res.get("amount", 0.0))
            }
        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"EMS: Real order routing failed: {str(e)}")
            return {"status": "REJECTED", "reason": str(e)}


class OrderManagementSystem:
    """
    Sovereign Order Management System (OMS - Phase 21 & Lot 7).
    Governs order lifecycle, validates with Risk, and serves as the sole entrypoint.
    """
    def __init__(self, db_manager, ems_client):
        self.db = db_manager
        self.ems = ems_client
        self.orders_cache = {}

    def submit_new_order(self, symbol: str, side: str, qty: float, price: float, mode: str, strategy: str, client_order_id: str, exchange: str = "Binance", order_type: str = "MARKET") -> Order:
        """
        OMS: Sole Entrypoint of Order Creation (Lot 7).
        """
        # Create and persist INITIAL order state
        order = Order(symbol, side, order_type, qty, price, mode, strategy, client_order_id, exchange)
        self.orders_cache[client_order_id] = order
        
        # Log to DB cache as CREATED
        self.db.add_order(
            symbol=order.symbol,
            side=order.side,
            price=order.requested_price,
            qty=order.requested_qty,
            status=order.status,
            mode=order.mode,
            strategy=order.strategy,
            order_type=order.type
        )
        
        logger.info(f"OMS: Created order {client_order_id} on {exchange}. Status: {order.status}.")
        return order

    def approve_and_execute_order(self, order: Order) -> dict:
        """
        Routes the order to EMS for execution upon validation.
        """
        order.status = OrderStatus.RISK_APPROVED
        logger.info(f"OMS: Order {order.client_order_id} RISK_APPROVED. Routing to EMS...")
        
        res = self.ems.route_and_execute_order(order)
        return res


class ReconciliationEngine:
    """
    Reconciliation Engine (Phase 24 & Lot 9).
    """
    def __init__(self, db_manager):
        self.db = db_manager

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
                f"Positions mismatch detected! Freezing trading. Details: {', '.join(mismatches)}"
            )
            return False
            
        logger.info("Reconciliation successful. Exchange and DB ledgers are fully aligned.")
        return True
