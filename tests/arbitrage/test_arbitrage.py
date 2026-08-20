from models.dex_cex_arbitrage import DexCexArbitrageEngine


def test_arbitrage_book_walking_success():
    engine = DexCexArbitrageEngine(min_profit_spread_pct=0.005) # 0.5% profit threshold

    # Exchange A is cheaper than Exchange B (profitable arbitrage)
    # We set tiny volumes on the first level to force book walking slippage!
    book_a = {
        "bids": [[59900.0, 1.0]],
        "asks": [[60000.0, 0.01], [60100.0, 2.0]]
    }

    book_b = {
        "bids": [[61000.0, 0.01], [60900.0, 2.0]],
        "asks": [[61200.0, 1.0]]
    }

    res = engine.calculate_executable_arbitrage(
        symbol="BTCUSDT",
        book_a=book_a,
        book_b=book_b,
        fee_a=0.001,  # 0.1% fee
        fee_b=0.0015, # 0.15% fee
        max_order_usd=3000.0
    )

    assert res["action"] == "EXECUTE_ARBITRAGE"
    assert res["route"] == "BUY_A_SELL_B"
    assert res["buy_price"] > 60000.0 # slipped price walked up the book!
    assert res["sell_price"] < 61000.0 # slipped price walked down the book!
    assert res["net_spread_pct"] > 0.005

def test_arbitrage_insufficient_depth():
    engine = DexCexArbitrageEngine()

    # Empty book levels should trigger fail-safe hold
    res = engine.calculate_executable_arbitrage("BTCUSDT", {}, {}, 0.001, 0.001)
    assert res["action"] == "HOLD"
    assert "missing" in res["reason"]
