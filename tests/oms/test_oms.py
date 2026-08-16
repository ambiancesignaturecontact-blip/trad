import pytest
from models.oms_ems import OrderManagementSystem, ExecutionManagementSystem, OrderStatus
from adapters.exchange_adapter import BinanceExchangeAdapter, BybitExchangeAdapter
from db_manager import DBManager

def test_oms_and_ems_lifecycle_flow():
    db = DBManager()
    
    # Instantiate adapters
    binance = BinanceExchangeAdapter(None)
    bybit = BybitExchangeAdapter(None)
    
    # Instantiate EMS & OMS
    ems = ExecutionManagementSystem(binance, bybit)
    oms = OrderManagementSystem(db, ems)
    
    # Create order
    order = oms.submit_new_order(
        symbol="BTCUSDT",
        side="BUY",
        qty=0.01,
        price=60000.0,
        mode="DEMO",
        strategy="META_MODEL",
        client_order_id="unique_client_id_77"
    )
    
    assert order.status == OrderStatus.CREATED
    assert order.requested_qty == 0.01
    
    # Approve and execute order through OMS -> EMS!
    res = oms.approve_and_execute_order(order)
    
    assert res["status"] == "ACKNOWLEDGED"
    assert order.status == OrderStatus.ACKNOWLEDGED
