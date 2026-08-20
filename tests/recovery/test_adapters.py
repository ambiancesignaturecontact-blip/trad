from adapters.exchange_adapter import BinanceExchangeAdapter, BybitExchangeAdapter


def test_adapters_initialization_with_none():
    # Adapters should be fully safe and return empty collections if initialized with None (no real CCXT client)
    binance = BinanceExchangeAdapter(None)
    bybit = BybitExchangeAdapter(None)

    assert binance.get_balance() == {}
    assert binance.get_positions() == []
    assert binance.get_open_orders() == []
    assert binance.get_trades() == []

    assert bybit.get_balance() == {}
    assert bybit.get_positions() == []
    assert bybit.get_open_orders() == []
    assert bybit.get_trades() == []
