import pytest
from models.oms_ems import OrderManagementSystem, ReconciliationEngine, OrderStatus
from db_manager import DBManager

def test_oms_order_lifecycle():
    db = DBManager()
    oms = OrderManagementSystem(db)
    
    order = oms.create_order(
        symbol="BTCUSDT",
        side="BUY",
        qty=0.05,
        price=60000.0,
        mode="DEMO",
        strategy="META_MODEL",
        client_order_id="client_id_99"
    )
    
    assert order.status == OrderStatus.PENDING
    
    # Update to Filled
    oms.update_order_status("client_id_99", OrderStatus.FILLED, filled_qty=0.05, avg_price=60100.0)
    
    assert order.status == OrderStatus.FILLED
    assert order.filled_qty == 0.05
    assert order.average_filled_price == 60100.0

def test_reconciliation_flow():
    db = DBManager()
    reconciler = ReconciliationEngine(db)
    
    # Query current positions from DB to align mock exchange
    db_positions = db.get_positions()
    mock_exchange_positions = {p['symbol']: p['qty'] for p in db_positions if p['mode'] == "DEMO"}
    
    # Reconciling with identical positions should succeed perfectly!
    res = reconciler.reconcile_positions(mock_exchange_positions, "DEMO")
    assert res is True
