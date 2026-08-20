import pytest
from database.db_manager import DBManager

def test_capital_persistence_flow():
    db = DBManager()
    
    # 1. Test 10,000 capital persistence
    db.save_setting("balance_demo", "10000.0")
    db.save_setting("initial_capital_demo", "10000.0")
    assert float(db.get_setting("balance_demo")) == 10000.0
    assert float(db.get_setting("initial_capital_demo")) == 10000.0
    
    # 2. Test 50,000 capital persistence
    db.save_setting("balance_demo", "50000.0")
    db.save_setting("initial_capital_demo", "50000.0")
    assert float(db.get_setting("balance_demo")) == 50000.0
    assert float(db.get_setting("initial_capital_demo")) == 50000.0
    
    # 3. Test 100,000 capital persistence
    db.save_setting("balance_demo", "100000.0")
    db.save_setting("initial_capital_demo", "100000.0")
    assert float(db.get_setting("balance_demo")) == 100000.0
    assert float(db.get_setting("initial_capital_demo")) == 100000.0
