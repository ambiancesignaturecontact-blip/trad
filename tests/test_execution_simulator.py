import pytest
from core.execution_simulator import ExecutionSimulator

def test_slippage_calculation():
    sim = ExecutionSimulator(base_slippage_bps=5.0)
    
    price = 60000
    executed = sim.simulate_slippage(price, "BUY", volatility=0.02, liquidity_score=0.8, order_size_pct=0.03)
    
    assert executed > price  # Slippage should increase buy price

def test_latency_simulation():
    sim = ExecutionSimulator()
    latency = sim.simulate_latency(network_jitter=True)
    assert 50 < latency < 200
