import logging
import time

logger = logging.getLogger("EMS")

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

class Fill:
    def __init__(self, fill_id: str, order_id: str, exchange_trade_id: str, price: float, quantity: float, fee: float, fee_asset: str, side: str, liquidity: str = "taker"):
        self.fill_id = fill_id
        self.order_id = order_id
        self.exchange_trade_id = exchange_trade_id
        self.price = float(price)
        self.quantity = float(quantity)
        self.fee = float(fee)
        self.fee_asset = fee_asset
        self.side = side
        self.liquidity = liquidity
        self.timestamp = time.time()

class ExecutionManagementSystem:
    def __init__(self, binance_adapter, bybit_adapter):
        self.binance = binance_adapter
        self.bybit = bybit_adapter

    def route_and_execute_order(self, order) -> dict:
        order.status = OrderStatus.SUBMITTED
        order.timestamp_updated = time.time()
        adapter = self.binance if order.exchange.lower() == "binance" else self.bybit
        
        if order.mode in ["DEMO", "PAPER", "SHADOW"]:
            order.status = OrderStatus.ACKNOWLEDGED
            logger.info(f"EMS: Routed Virtual Order {order.client_order_id} on {order.exchange}.")
            return {
                "status": "ACKNOWLEDGED",
                "order_id": f"sim_id_{int(time.time())}",
                "price": order.requested_price,
                "amount": order.requested_qty
            }
            
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
