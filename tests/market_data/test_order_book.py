import pytest
import time
from market_data.order_book import LiveOrderBookManager

def test_order_book_initialization():
    manager = LiveOrderBookManager("BTCUSDT", "Binance")
    
    bids = [["60000.0", "1.5"], ["59900.0", "2.0"]]
    asks = [["60100.0", "1.0"], ["60200.0", "3.4"]]
    
    manager.initialize_with_snapshot(bids, asks, sequence=100)
    
    assert manager.is_synced is True
    assert manager.active_book.sequence == 100
    assert len(manager.active_book.bids) == 2
    assert manager.active_book.bids[0][0] == 60000.0

def test_apply_update_success():
    manager = LiveOrderBookManager("BTCUSDT", "Binance")
    manager.initialize_with_snapshot(
        bids=[["60000.0", "1.5"]],
        asks=[["60100.0", "1.0"]],
        sequence=100
    )
    
    # Apply a valid sequential update (first_update_id = 101, final_update_id = 101)
    res = manager.apply_websocket_update(
        bids_update=[["60000.0", "2.5"]], # overwrite quantity
        asks_update=[["60200.0", "0.5"]], # insert new level
        first_update_id=101,
        final_update_id=101
    )
    
    assert res is True
    assert manager.active_book.sequence == 101
    assert manager.active_book.bids[0][1] == 2.5
    assert len(manager.active_book.asks) == 2

def test_gap_detection():
    manager = LiveOrderBookManager("BTCUSDT", "Binance")
    manager.initialize_with_snapshot(
        bids=[["60000.0", "1.5"]],
        asks=[["60100.0", "1.0"]],
        sequence=100
    )
    
    # Apply a gapped update (first_update_id = 105, skipping 101-104)
    res = manager.apply_websocket_update(
        bids_update=[],
        asks_update=[],
        first_update_id=105,
        final_update_id=105
    )
    
    assert res is False
    assert manager.is_synced is False # Synced flag is broken, triggering resync!

def test_stale_detection():
    manager = LiveOrderBookManager("BTCUSDT", "Binance", stale_timeout_seconds=1.0)
    manager.initialize_with_snapshot(
        bids=[["60000.0", "1.5"]],
        asks=[["60100.0", "1.0"]],
        sequence=100
    )
    
    assert manager.check_is_valid_and_fresh() is True
    
    # Wait for stale timeout
    time.sleep(1.2)
    
    # Should evaluate as stale and invalid
    assert manager.check_is_valid_and_fresh() is False
