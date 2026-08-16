import time
import logging

logger = logging.getLogger("OrderBook")

class OrderBookSnapshot:
    """
    Standard normalized Order Book Snapshot structure (Phase 9).
    """
    def __init__(self, symbol: str, exchange: str, bids: list, asks: list, sequence: int = 0):
        self.symbol = symbol
        self.exchange = exchange
        self.bids = bids # List of [price, qty]
        self.asks = asks # List of [price, qty]
        self.sequence = sequence
        self.timestamp = time.time()


class LiveOrderBookManager:
    """
    Sovereign Live Order Book Manager (Phase 5 & Lot 3).
    Tracks real-world WebSocket order book streams, enforces sequence integrity,
    detects sequence gaps, and triggers automatic resynchronization.
    """
    def __init__(self, symbol: str, exchange: str, stale_timeout_seconds=15.0):
        self.symbol = symbol
        self.exchange = exchange
        self.stale_timeout_seconds = stale_timeout_seconds
        
        self.active_book = None # Active OrderBookSnapshot
        self.last_update_epoch = 0.0
        self.is_synced = False

    def initialize_with_snapshot(self, bids: list, asks: list, sequence: int):
        """
        Loads the initial order book snapshot from the exchange REST API.
        """
        # Parse floats
        parsed_bids = [[float(b[0]), float(b[1])] for b in bids]
        parsed_asks = [[float(a[0]), float(a[1])] for a in asks]
        
        self.active_book = OrderBookSnapshot(
            symbol=self.symbol,
            exchange=self.exchange,
            bids=parsed_bids,
            asks=parsed_asks,
            sequence=sequence
        )
        self.last_update_epoch = time.time()
        self.is_synced = True
        logger.info(f"OrderBook: Initialized {self.symbol} ({self.exchange}) snapshot at sequence {sequence}.")

    def apply_websocket_update(self, bids_update: list, asks_update: list, first_update_id: int, final_update_id: int) -> bool:
        """
        Applies live incremental WebSocket updates to the active book.
        Enforces strict sequence and gap detection checks!
        """
        if not self.active_book:
            logger.warning(f"OrderBook: Cannot apply update. No active snapshot loaded for {self.symbol}.")
            return False
            
        # 1. Enforce Sequence Validation & Gap Detection
        # The update's first update ID must align precisely with the previous sequence
        # Binance protocol: first_update_id <= last_update_id + 1 AND final_update_id >= last_update_id + 1
        last_seq = self.active_book.sequence
        if first_update_id > last_seq + 1:
            logger.error(
                f"🚨 ORDER BOOK GAP DETECTED on {self.symbol}! "
                f"Expected first update ID <= {last_seq + 1}, got {first_update_id}. Triggering Resynchronization!"
            )
            self.is_synced = False
            return False
            
        # Ignore old updates
        if final_update_id <= last_seq:
            return True
            
        # 2. Update Bids and Asks
        # Helper to merge updates: if quantity is 0, delete the level; otherwise overwrite/insert
        self._update_levels(self.active_book.bids, bids_update, reverse=True)
        self._update_levels(self.active_book.asks, asks_update, reverse=False)
        
        # Update sequence tracking
        self.active_book.sequence = final_update_id
        self.last_update_epoch = time.time()
        return True

    def _update_levels(self, current_levels: list, updates: list, reverse: bool):
        for upd in updates:
            price = float(upd[0])
            qty = float(upd[1])
            
            # Find if price level already exists
            match_idx = -1
            for idx, lvl in enumerate(current_levels):
                if abs(lvl[0] - price) < 1e-8:
                    match_idx = idx
                    break
                    
            if qty == 0.0:
                if match_idx != -1:
                    current_levels.pop(match_idx)
            else:
                if match_idx != -1:
                    current_levels[match_idx][1] = qty
                else:
                    current_levels.append([price, qty])
                    
        # Re-sort order book (bids descending, asks ascending)
        current_levels.sort(key=lambda x: x[0], reverse=reverse)
        # Keep maximum of 20 depth levels for memory efficiency
        del current_levels[20:]

    def check_is_valid_and_fresh(self) -> bool:
        """
        Checks if the order book is synchronized and fresh.
        If stale, flags as invalid to block trading!
        """
        if not self.is_synced or not self.active_book:
            return False
            
        elapsed = time.time() - self.last_update_epoch
        if elapsed > self.stale_timeout_seconds:
            logger.error(f"OrderBook: {self.symbol} is STALE (Last update {elapsed:.1f}s ago). Halting trading!")
            self.is_synced = False
            return False
            
        return True

    def get_bids_asks(self) -> tuple:
        """
        Returns the active 5-level bids and asks.
        Returns (None, None) if the book is invalid or stale.
        """
        if not self.check_is_valid_and_fresh():
            return None, None
        return self.active_book.bids[:5], self.active_book.asks[:5]
