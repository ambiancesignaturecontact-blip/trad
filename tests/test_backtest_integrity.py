"""
Backtest integrity tests (roadmap precision #2):
1. The EventDrivenBacktester runs end-to-end on synthetic bars without errors.
2. Structural guard: the engine must NEVER peek at future bars
   (look-ahead bias). It may only consume `df_bars.iloc[:i]` at step i.
"""
import numpy as np
import pandas as pd

from backtester.engine import EventDrivenBacktester
from models.regime_detector import MarketRegimeDetector
from models.price_predictor import LSTMLikePredictor, PPOTRAgent
from strategies.engine import TrendFollowingStrategy, MetaAllocationEngine
from risk.risk_manager import RiskManager


def _make_bars(n=300, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high = close * 1.005
    low = close * 0.995
    open_ = close * 0.999
    volume = rng.uniform(500, 2000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_backtester_runs_end_to_end():
    df = _make_bars()
    engine = EventDrivenBacktester(initial_capital=100000.0)
    strategy = TrendFollowingStrategy()
    meta = MetaAllocationEngine(strategies=[strategy])
    risk = RiskManager()
    detector = MarketRegimeDetector()
    predictor = LSTMLikePredictor(input_dim=5, hidden_dim=8)
    ppo = PPOTRAgent(state_dim=4, action_dim=1)

    metrics = engine.run(df, meta, risk, detector, predictor, ppo)
    assert metrics is not None
    assert metrics["initial_capital"] == 100000.0
    assert isinstance(metrics["final_equity"], (int, float))
    # Fees + slippage must never manufacture capital out of thin air
    assert metrics["final_equity"] > 0


def test_no_lookahead_bias_in_engine():
    """Structural guard: engine may only consume strictly past bars at step i."""
    src = open("backtester/engine.py").read()
    assert "iloc[:i]" in src, "engine must slice past bars with iloc[:i]"
    # Forbidden: any slice including the current/future bar
    for forbidden in ["iloc[:i+1]", "iloc[:i + 1]", "iloc[:i+2]", "iloc[i:]", "iloc[i+1:]"]:
        assert forbidden not in src, f"look-ahead pattern {forbidden} detected!"


def test_walk_forward_validation_runs():
    df = _make_bars(n=400)
    engine = EventDrivenBacktester(initial_capital=100000.0)
    strategy = TrendFollowingStrategy()
    meta = MetaAllocationEngine(strategies=[strategy])
    risk = RiskManager()
    detector = MarketRegimeDetector()
    predictor = LSTMLikePredictor(input_dim=5, hidden_dim=8)
    ppo = PPOTRAgent(state_dim=4, action_dim=1)

    from backtester.engine import WalkForwardValidator
    wf = WalkForwardValidator(train_ratio=0.7)
    res = wf.run_validation(df, engine, meta, risk, detector, predictor, ppo)
    assert res is not None
    assert "in_sample_metrics" in res and "out_of_sample_metrics" in res
