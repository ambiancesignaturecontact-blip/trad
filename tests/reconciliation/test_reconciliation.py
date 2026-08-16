import pytest
from models.oms_ems import ReconciliationEngine
from db_manager import DBManager

def test_reconciliation_balance_success():
    db = DBManager()
    reconciler = ReconciliationEngine(db)
    
    # Matching balances should pass
    assert reconciler.reconcile_balances(1000.0, 1000.0) is True

def test_reconciliation_balance_mismatch():
    db = DBManager()
    reconciler = ReconciliationEngine(db)
    
    # Mismatch of $5 should fail
    assert reconciler.reconcile_balances(1000.0, 995.0) is False

def test_reconciliation_positions_mismatch():
    db = DBManager()
    reconciler = ReconciliationEngine(db)
    
    # Real mismatch should fail
    actual_on_exchange = {"BTCUSDT": 0.0} # Flat on exchange, but DB has positions
    assert reconciler.reconcile_positions(actual_on_exchange, "DEMO") is False
