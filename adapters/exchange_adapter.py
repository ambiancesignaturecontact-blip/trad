import logging

logger = logging.getLogger("ExchangeAdapter")

class ExchangeAdapter:
    """
    Abstract Base Class representing the unified Exchange Adapter Interface (Phase 23).
    """
    def get_balance(self) -> dict:
        raise NotImplementedError

    def get_positions(self) -> list:
        raise NotImplementedError

    def get_open_orders(self, symbol: str = None) -> list:
        raise NotImplementedError

    def get_order(self, order_id: str, symbol: str = None) -> dict:
        raise NotImplementedError

    def place_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None) -> dict:
        raise NotImplementedError

    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        raise NotImplementedError

    def get_trades(self, symbol: str = None) -> list:
        raise NotImplementedError

    def get_orderbook(self, symbol: str) -> dict:
        raise NotImplementedError


class BinanceExchangeAdapter(ExchangeAdapter):
    """
    Binance perpetual futures Exchange Adapter utilizing the CCXT client.
    """
    def __init__(self, ccxt_client):
        self.client = ccxt_client

    def get_balance(self) -> dict:
        if not self.client:
            return {}
        try:
            bal = self.client.fetch_balance()
            return bal
        except Exception as e:
            logger.error(f"BinanceAdapter: get_balance failed: {str(e)}")
            return {}

    def get_positions(self) -> list:
        if not self.client:
            return []
        try:
            positions = self.client.fetch_positions()
            return positions
        except Exception as e:
            logger.error(f"BinanceAdapter: get_positions failed: {str(e)}")
            return []

    def get_open_orders(self, symbol: str = None) -> list:
        if not self.client:
            return []
        try:
            orders = self.client.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            logger.error(f"BinanceAdapter: get_open_orders failed: {str(e)}")
            return []

    def get_order(self, order_id: str, symbol: str = None) -> dict:
        if not self.client:
            return {}
        try:
            order = self.client.fetch_order(order_id, symbol)
            return order
        except Exception as e:
            logger.error(f"BinanceAdapter: get_order failed: {str(e)}")
            return {}

    def place_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None) -> dict:
        if not self.client:
            return {}
        try:
            order = self.client.create_order(symbol, order_type, side, amount, price)
            return order
        except Exception as e:
            logger.error(f"BinanceAdapter: place_order failed: {str(e)}")
            raise e

    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        if not self.client:
            return False
        try:
            self.client.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"BinanceAdapter: cancel_order failed: {str(e)}")
            return False

    def get_trades(self, symbol: str = None) -> list:
        if not self.client:
            return []
        try:
            trades = self.client.fetch_my_trades(symbol)
            return trades
        except Exception as e:
            logger.error(f"BinanceAdapter: get_trades failed: {str(e)}")
            return []

    def get_orderbook(self, symbol: str) -> dict:
        if not self.client:
            return {}
        try:
            ob = self.client.fetch_order_book(symbol)
            return ob
        except Exception as e:
            logger.error(f"BinanceAdapter: get_orderbook failed: {str(e)}")
            return {}


class BybitExchangeAdapter(ExchangeAdapter):
    """
    Bybit perpetual futures Exchange Adapter utilizing the CCXT client.
    """
    def __init__(self, ccxt_client):
        self.client = ccxt_client

    def get_balance(self) -> dict:
        if not self.client:
            return {}
        try:
            return self.client.fetch_balance()
        except Exception as e:
            logger.error(f"BybitAdapter: get_balance failed: {str(e)}")
            return {}

    def get_positions(self) -> list:
        if not self.client:
            return []
        try:
            return self.client.fetch_positions()
        except Exception as e:
            logger.error(f"BybitAdapter: get_positions failed: {str(e)}")
            return []

    def get_open_orders(self, symbol: str = None) -> list:
        if not self.client:
            return []
        try:
            return self.client.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"BybitAdapter: get_open_orders failed: {str(e)}")
            return []

    def get_order(self, order_id: str, symbol: str = None) -> dict:
        if not self.client:
            return {}
        try:
            return self.client.fetch_order(order_id, symbol)
        except Exception as e:
            logger.error(f"BybitAdapter: get_order failed: {str(e)}")
            return {}

    def place_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None) -> dict:
        if not self.client:
            return {}
        try:
            return self.client.create_order(symbol, order_type, side, amount, price)
        except Exception as e:
            logger.error(f"BybitAdapter: place_order failed: {str(e)}")
            raise e

    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        if not self.client:
            return False
        try:
            self.client.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"BybitAdapter: cancel_order failed: {str(e)}")
            return False

    def get_trades(self, symbol: str = None) -> list:
        if not self.client:
            return []
        try:
            return self.client.fetch_my_trades(symbol)
        except Exception as e:
            logger.error(f"BybitAdapter: get_trades failed: {str(e)}")
            return []

    def get_orderbook(self, symbol: str) -> dict:
        if not self.client:
            return {}
        try:
            return self.client.fetch_order_book(symbol)
        except Exception as e:
            logger.error(f"BybitAdapter: get_orderbook failed: {str(e)}")
            return {}
