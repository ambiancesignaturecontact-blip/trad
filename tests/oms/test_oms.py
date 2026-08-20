from adapters.exchange_adapter import BinanceExchangeAdapter, BybitExchangeAdapter
from database.db_manager import DBManager
from models.oms_ems import ExecutionManagementSystem, Fill, OrderManagementSystem, OrderStatus


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

def test_confirmed_fills_flow():
    db = DBManager()
    ems = ExecutionManagementSystem(None, None)
    oms = OrderManagementSystem(db, ems)

    order = oms.submit_new_order(
        symbol="BTCUSDT",
        side="BUY",
        qty=0.10,
        price=60000.0,
        mode="DEMO",
        strategy="META_MODEL",
        client_order_id="client_id_88"
    )

    # Enforce that order status is NOT filled on submission
    assert order.status == OrderStatus.CREATED

    # Process a PARTIAL fill
    f1 = Fill(
        fill_id="f_01",
        order_id=order.internal_order_id,
        exchange_trade_id="trade_01",
        price=60100.0,
        quantity=0.04,
        fee=2.4,
        fee_asset="USDT",
        side="BUY"
    )
    oms.process_exchange_fill_receipt("client_id_88", f1)

    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert order.filled_qty == 0.04
    assert order.remaining_qty == 0.06

    # Process the final finishing fill (fully filled!)
    f2 = Fill(
        fill_id="f_02",
        order_id=order.internal_order_id,
        exchange_trade_id="trade_02",
        price=60200.0,
        quantity=0.06,
        fee=3.6,
        fee_asset="USDT",
        side="BUY"
    )
    oms.process_exchange_fill_receipt("client_id_88", f2)

    assert order.status == OrderStatus.FILLED
    assert order.filled_qty == 0.10
    assert order.remaining_qty == 0.0
    assert order.average_fill_price == 60160.0 # weighted average price!

def test_order_rejection_flow():
    db = DBManager()
    ems = ExecutionManagementSystem(None, None)
    oms = OrderManagementSystem(db, ems)

    order = oms.submit_new_order(
        symbol="BTCUSDT",
        side="BUY",
        qty=0.05,
        price=60000.0,
        mode="DEMO",
        strategy="META_MODEL",
        client_order_id="client_id_rejection"
    )

    oms.process_order_rejection("client_id_rejection", "Risk limit exceeded")
    assert order.status == OrderStatus.REJECTED

def test_order_cancellation_flow():
    db = DBManager()
    ems = ExecutionManagementSystem(None, None)
    oms = OrderManagementSystem(db, ems)

    order = oms.submit_new_order(
        symbol="BTCUSDT",
        side="BUY",
        qty=0.05,
        price=60000.0,
        mode="DEMO",
        strategy="META_MODEL",
        client_order_id="client_id_cancellation"
    )

    oms.process_order_cancellation("client_id_cancellation")
    assert order.status == OrderStatus.CANCELLED
