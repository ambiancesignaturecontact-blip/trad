import logging
import time

logger = logging.getLogger("OMS")

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
    def __init__(self, symbol: str, side: str, order_type: str, requested_qty: float, requested_price: float, mode: str, strategy: str, client_order_id: str, exchange: str):
        self.internal_order_id = f"int_ord_{int(time.time()*1000)}"
        self.client_order_id = client_order_id
        self.exchange_order_id = None
        
        self.symbol = symbol
        self.side = side
        self.type = order_type
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
        self.fills = []

class OrderManagementSystem:
    def __init__(self, db_manager, ems_client):
        self.db = db_manager
        self.ems = ems_client
        self.orders_cache = {}

    def submit_new_order(self, symbol: str, side: str, qty: float, price: float, mode: str, strategy: str, client_order_id: str, exchange: str = "Binance", order_type: str = "MARKET") -> Order:
        order = Order(symbol, side, order_type, qty, price, mode, strategy, client_order_id, exchange)
        self.orders_cache[client_order_id] = order
        
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
        order.status = OrderStatus.RISK_APPROVED
        logger.info(f"OMS: Order {order.client_order_id} RISK_APPROVED. Routing to EMS...")
        res = self.ems.route_and_execute_order(order)
        return res

    def process_exchange_fill_receipt(self, client_order_id: str, fill):
        if client_order_id not in self.orders_cache:
            logger.error(f"OMS: Received fill receipt for untracked client order ID {client_order_id}.")
            return
            
        order = self.orders_cache[client_order_id]
        order.fills.append(fill)
        
        total_filled_qty = sum(f.quantity for f in order.fills)
        total_cost = sum(f.quantity * f.price for f in order.fills)
        avg_price = total_cost / total_filled_qty if total_filled_qty > 0 else 0.0
        
        order.filled_qty = total_filled_qty
        order.remaining_qty = round(max(0.0, order.requested_qty - total_filled_qty), 8)
        order.average_fill_price = avg_price
        order.fees += fill.fee
        order.fee_asset = fill.fee_asset
        
        if order.remaining_qty == 0.0:
            order.status = OrderStatus.FILLED
            logger.info(f"OMS: Order {client_order_id} is fully FILLED.")
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
            logger.info(f"OMS: Order {client_order_id} is PARTIALLY_FILLED ({total_filled_qty}/{order.requested_qty}).")
            
        order.timestamp_updated = time.time()
        
        self.db.add_order(
            symbol=order.symbol,
            side=order.side,
            price=avg_price,
            qty=total_filled_qty,
            status=order.status,
            mode=order.mode,
            strategy=order.strategy,
            order_type=order.type
        )
        
        self.db.save_fill(
            fill_id=fill.fill_id,
            order_id=order.internal_order_id,
            exchange_trade_id=fill.exchange_trade_id,
            price=fill.price,
            quantity=fill.quantity,
            fee=fill.fee,
            fee_asset=fill.fee_asset,
            side=fill.side,
            liquidity=fill.liquidity
        )
        
        self.db.update_position(order.symbol, total_filled_qty, avg_price, order.mode)
        logger.info(f"OMS: Updated DB position for {order.symbol} to {total_filled_qty} at average price {avg_price}.")

    def process_order_rejection(self, client_order_id: str, reason: str):
        if client_order_id in self.orders_cache:
            order = self.orders_cache[client_order_id]
            order.status = OrderStatus.REJECTED
            order.timestamp_updated = time.time()
            self.db.add_order(
                symbol=order.symbol,
                side=order.side,
                price=order.requested_price,
                qty=order.requested_qty,
                status=OrderStatus.REJECTED,
                mode=order.mode,
                strategy=order.strategy,
                order_type=order.type
            )
            logger.warning(f"OMS: Order {client_order_id} REJECTED. Reason: {reason}.")

    def process_order_cancellation(self, client_order_id: str):
        if client_order_id in self.orders_cache:
            order = self.orders_cache[client_order_id]
            order.status = OrderStatus.CANCELLED
            order.timestamp_updated = time.time()
            self.db.add_order(
                symbol=order.symbol,
                side=order.side,
                price=order.requested_price,
                qty=order.requested_qty,
                status=OrderStatus.CANCELLED,
                mode=order.mode,
                strategy=order.strategy,
                order_type=order.type
            )
            logger.info(f"OMS: Order {client_order_id} CANCELLED successfully.")
