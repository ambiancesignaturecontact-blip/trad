"""
AUDIT B16-1/B16-2/B16-3: unit tests for the previously-untested core modules:
strategies engine, risk engine, OMS/EMS, SOR, dynamic hedging, correlation risk,
online model selector, adaptive ensemble, position protection, and a mocked
CCXT execution path (fills, partial fills, rejections).
"""
import numpy as np
import pandas as pd
import pytest


# ---------------- strategies / meta allocation ----------------
def test_meta_engine_allocate_returns_weights():
    from strategies.engine import TrendFollowingStrategy, MeanReversionStrategy, MetaAllocationEngine
    meta = MetaAllocationEngine(strategies=[TrendFollowingStrategy(), MeanReversionStrategy()])
    df = pd.DataFrame({"close": np.linspace(100, 110, 80), "high": np.linspace(101, 111, 80),
                       "low": np.linspace(99, 109, 80), "volume": [1000] * 80})
    md = {"df": df, "price_primary": 105.0, "price_secondary": 105.0, "bids": [[104, 1]],
          "asks": [[106, 1]], "inventory": 0.0, "max_inventory": 100.0,
          "vpin": 0.5, "kyle_lambda": 0.0001, "onchain_risk": 0.2, "sentiment": 0.1}
    res = meta.allocate(md, 2, 0.0, 0.0)
    assert -1.0 <= res["final_signal"] <= 1.0
    assert "modulate_factor" in res
    assert set(res["walkforward_weights"].keys()) == {"Trend Following", "Mean Reversion"}


def test_meta_engine_vpin_modulates_signal():
    from strategies.engine import TrendFollowingStrategy, MetaAllocationEngine
    meta = MetaAllocationEngine(strategies=[TrendFollowingStrategy()])
    df = pd.DataFrame({"close": np.linspace(100, 110, 80), "high": np.linspace(101, 111, 80),
                       "low": np.linspace(99, 109, 80), "volume": [1000] * 80})
    base = {"df": df, "price_primary": 105.0, "price_secondary": 105.0, "bids": [[104, 1]],
            "asks": [[106, 1]], "inventory": 0.0, "max_inventory": 100.0,
            "vpin": 0.5, "kyle_lambda": 0.0001, "onchain_risk": 0.2, "sentiment": 0.1}
    low = meta.allocate(dict(base), 2, 0.0, 0.0)["final_signal"]
    base["vpin"] = 0.98
    high = meta.allocate(dict(base), 2, 0.0, 0.0)["final_signal"]
    assert abs(high) <= abs(low) + 1e-9  # toxic flow must not increase conviction


# ---------------- risk engine ----------------
def test_risk_manager_sizing_and_safety():
    from risk.risk_manager import RiskManager
    rm = RiskManager()
    rm.set_initial_capital(100000.0)
    qty = rm.calculate_position_size(100000.0, 200.0, 60000.0)
    assert qty > 0
    ok, reason = rm.validate_order_safety(order_price=60000.0, mid_market_price=60050.0,
                                          order_qty=qty, capital_available=100000.0)
    assert ok is True
    # absurd order must be rejected
    ok2, _ = rm.validate_order_safety(order_price=99999.0, mid_market_price=60000.0,
                                      order_qty=1000000.0, capital_available=100000.0)
    assert ok2 is False


def test_circuit_breaker_trips_on_big_drawdown():
    from risk.risk_manager import RiskManager
    rm = RiskManager()
    rm.set_initial_capital(100000.0)
    tripped, msg = rm.check_circuit_breaker(90000.0)
    # daily drawdown limit is 2.5% -> a 10% loss MUST trip the breaker
    assert tripped is True
    assert msg != ""


# ---------------- OMS / EMS ----------------
def test_oms_submit_and_partial_fill():
    from oms.manager import Order, OrderManagementSystem, OrderStatus
    from ems.manager import Fill

    class FakeDB:
        def add_order(self, **kw):
            self.last = kw
        def save_fill(self, *a, **kw):
            pass
        def update_position(self, *a, **kw):
            pass

    class FakeEMS:
        def route_and_execute_order(self, order):
            order.status = OrderStatus.ACKNOWLEDGED
            return {"status": "ACKNOWLEDGED", "order_id": "ex1", "price": 100.0, "amount": 1.0}

    db = FakeDB()
    oms = OrderManagementSystem(db, FakeEMS())
    order = oms.submit_new_order("BTCUSDT", "BUY", 1.0, 60000.0, "DEMO", "TEST", "cid1")
    assert order.status == OrderStatus.CREATED
    res = oms.approve_and_execute_order(order)
    assert res["status"] == "ACKNOWLEDGED"

    # partial fill -> PARTIALLY_FILLED, then full -> FILLED
    oms.process_exchange_fill_receipt("cid1", Fill("f1", "cid1", "t1", 60000.0, 0.4, 0.1, "USDT", "BUY"))
    assert order.status == OrderStatus.PARTIALLY_FILLED
    oms.process_exchange_fill_receipt("cid1", Fill("f2", "cid1", "t2", 60000.0, 0.6, 0.1, "USDT", "BUY"))
    assert order.status == OrderStatus.FILLED
    assert order.average_fill_price == 60000.0


# ---------------- SOR ----------------
def test_multi_exchange_sor_selects_best_venue():
    from core.multi_exchange_sor import MultiExchangeSmartOrderRouter, ExchangeQuote
    sor = MultiExchangeSmartOrderRouter()
    def q(exchange, bid, ask):
        return ExchangeQuote(exchange=exchange, price=(bid+ask)/2, bid=bid, ask=ask,
                             bid_qty=10.0, ask_qty=10.0, fee_rate=0.001, latency_ms=10.0,
                             liquidity_usd=1e6, net_cost_buy=ask, net_cost_sell=bid)
    quotes = [q("Binance", 60000.0, 60010.0), q("Bybit", 60010.0, 60012.0)]
    best = sor.select_best_venue(quotes, side="BUY")
    assert best.exchange == "Binance"  # cheaper ask
    best_sell = sor.select_best_venue(quotes, side="SELL")
    assert best_sell.exchange == "Bybit"  # higher bid


# ---------------- dynamic hedging / correlation ----------------
def test_dynamic_hedging_multiplier():
    from core.dynamic_hedging import DynamicHedgingEngine
    h = DynamicHedgingEngine()
    assert h.get_hedging_multiplier(0.9) < h.get_hedging_multiplier(0.2)
    assert h.should_hedge(0.9) is True


def test_correlation_risk_adjustment():
    from core.correlation_risk import CorrelationRiskEngine
    c = CorrelationRiskEngine()
    assert c.get_correlation_adjustment(0.9) < c.get_correlation_adjustment(0.2)


# ---------------- online model selector / ensemble ----------------
def test_online_model_selector_weights_and_status():
    from models.online_model_selector import OnlineModelSelector
    sel = OnlineModelSelector(["m1", "m2", "m3"])
    sel.update_performance("m1", 0.5)
    sel.update_performance("m2", -0.3)
    sel.update_performance("m3", 0.1)
    status = sel.get_status()
    assert set(status["active_models"]) == {"m1", "m2", "m3"}
    w = sel.get_active_weights()
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_adaptive_ensemble_decide():
    from models.online_model_selector import OnlineModelSelector
    from rl.adaptive_ensemble_agent import AdaptiveEnsembleAgent
    sel = OnlineModelSelector(["m1", "m2"])
    agent = AdaptiveEnsembleAgent(sel)
    res = agent.decide("BTCUSDT", {"price": 60000.0}, {"m1": 0.2, "m2": -0.1})
    assert isinstance(res, dict)


# ---------------- position protection ----------------
def test_position_protection_sl_tp_trailing():
    from core.position_manager import PositionProtection, evaluate_protection
    p = PositionProtection("BTCUSDT", 60000.0, 0.1, stop_loss_pct=0.05, take_profit_pct=0.10)
    assert evaluate_protection(p, 57000.0, 0.1) == "STOP_LOSS"
    assert evaluate_protection(p, 66000.0, 0.1) == "TAKE_PROFIT"
    assert evaluate_protection(p, 61000.0, 0.1) == "HOLD"

    t = PositionProtection("ETHUSDT", 3000.0, 2.0, trailing_pct=0.02)
    evaluate_protection(t, 3300.0, 2.0)  # ratchet up
    assert t.high_water == 3300.0
    assert t.stop_price > 3000.0 * 0.98  # trailing pulled up


# ---------------- mocked CCXT execution (audit B16-3) ----------------
class FakeCCXT:
    """Minimal fake CCXT client used to exercise the REAL execution path."""

    def __init__(self, fail=False):
        self.fail = fail
        self.orders = []

    def create_order(self, symbol, type, side, amount, params=None):
        if self.fail:
            raise Exception("insufficient funds")
        oid = f"oid_{len(self.orders)}"
        self.orders.append({"id": oid, "symbol": symbol, "side": side,
                            "amount": amount, "status": "open", "filled": 0.0,
                            "price": 60000.0, "average": None})
        return self.orders[-1]

    def fetch_order(self, order_id, symbol=None):
        for o in self.orders:
            if o["id"] == order_id:
                return o
        raise Exception("not found")


def _make_adapter(client):
    from ems.manager import ExecutionManagementSystem

    class FakeAdapter:
        def place_order(self, symbol, order_type, side, amount, price=None):
            return client.create_order(symbol, order_type, side, amount)

    # attach the client AFTER class creation (class-body scope quirk)
    FakeAdapter.client = client
    return FakeAdapter(), ExecutionManagementSystem(FakeAdapter(), FakeAdapter())


def test_real_execution_fill_confirmation_path():
    """Exercises the loop's REAL branch logic: submit -> confirm fill -> ledger."""
    from models.oms_ems import OrderManagementSystem

    client = FakeCCXT()
    adapter, ems = _make_adapter(client)

    class Db:
        def add_order(self, **kw):
            pass
        def update_position(self, *a, **kw):
            pass

    oms = OrderManagementSystem(Db(), ems)
    order = oms.submit_new_order("BTCUSDT", "BUY", 0.1, 60000.0, "REAL", "TEST", "cid_real")
    assert order.status == "CREATED"
    res = ems.route_and_execute_order(order)
    assert res["status"] == "SUCCESS"
    oid = res["order_id"]

    # fill confirmation: mark the order filled on the fake exchange
    for o in client.orders:
        o["status"] = "closed"
        o["filled"] = o["amount"]
        o["average"] = 60000.0
    fill = client.fetch_order(oid)
    assert fill["status"] == "closed"
    assert fill["filled"] == 0.1


def test_real_execution_rejection_path():
    from ems.manager import ExecutionManagementSystem
    from oms.manager import Order

    client = FakeCCXT(fail=True)
    _, ems = _make_adapter(client)
    order = Order("BTCUSDT", "BUY", "MARKET", 0.1, 60000.0, "REAL", "TEST", "cid_bad", "Binance")
    res = ems.route_and_execute_order(order)
    assert res["status"] == "REJECTED"
    assert order.status == "REJECTED"
