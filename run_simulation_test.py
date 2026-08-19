import numpy as np
import pandas as pd
import sys

# Import our quant engines
from models.regime_detector import MarketRegimeDetector
from models.price_predictor import LSTMLikePredictor, PPOTRAgent
from strategies.engine import MetaAllocationEngine, TrendFollowingStrategy, MeanReversionStrategy, GridTradingStrategy
from risk.risk_manager import RiskManager
from backtester.engine import EventDrivenBacktester
from backtester.bias_audit import audit_backtest

def run_quant_test():
    print("=========================================================================")
    print("📈 INSTITUTIONAL AI TRADING PLATFORM - SIMULATION PERFORMANCE TEST")
    print("=========================================================================")
    
    # Generate high-fidelity price history representing a real market cycle
    # (1000 bars: Uptrend, then crash, then range consolidation)
    np.random.seed(123)
    prices = [60000.0]
    
    # Phase 1: Bull Trend (300 bars)
    for _ in range(300):
        prices.append(prices[-1] * (1.0 + np.random.normal(0.0003, 0.004)))
        
    # Phase 2: Bear Crash (200 bars)
    for _ in range(200):
        prices.append(prices[-1] * (1.0 + np.random.normal(-0.0008, 0.008)))
        
    # Phase 3: Range Consolidation (500 bars)
    for _ in range(500):
        prices.append(prices[-1] * (1.0 + np.random.normal(0.0, 0.003)))
        
    timestamps = pd.date_range(start="2026-01-01", periods=1001, freq="H")
    df = pd.DataFrame({
        "close": prices,
        "high": [p * 1.002 for p in prices],
        "low": [p * 0.998 for p in prices],
        "open": [p * np.random.uniform(0.999, 1.001) for p in prices],
        "volume": [np.random.uniform(10.0, 100.0) for _ in prices]
    }, index=timestamps)
    
    # Initialize Models
    detector = MarketRegimeDetector()
    predictor = LSTMLikePredictor(5, 24)  # P0-5 : même archi que le live (audit §4.9)
    ppo = PPOTRAgent(4, 1)
    risk = RiskManager()
    
    # Fit models on the first 100 bars
    train_df = df.iloc[:100]
    returns = train_df['close'].pct_change().dropna().values
    vols = train_df['close'].pct_change().rolling(5).std().dropna().values
    min_len = min(len(returns), len(vols))
    detector.fit(np.column_stack((returns[-min_len:], vols[-min_len:])))
    
    # Fit price predictor
    feats = []
    labs = []
    pct_df = train_df[['close', 'volume', 'high', 'low', 'open']].pct_change().fillna(0)
    for i in range(5, len(pct_df) - 1):
        feats.append(pct_df.iloc[i-5:i].values)
        labs.append(pct_df['close'].iloc[i])
    predictor.fit(feats, np.array(labs))
    
    # Setup Strategy Engine
    from strategies.engine import (
        TrendFollowingStrategy, MeanReversionStrategy, MarketMakingStrategy,
        StatisticalArbitrageStrategy, ArbitrageInterExchangeStrategy, GridTradingStrategy,
        ScalpingStrategy
    )
    strategies = [
        TrendFollowingStrategy(),
        MeanReversionStrategy(),
        MarketMakingStrategy(),
        StatisticalArbitrageStrategy(),
        ArbitrageInterExchangeStrategy(),
        GridTradingStrategy(),
        ScalpingStrategy()
    ]
    meta_engine = MetaAllocationEngine(strategies=strategies)
    
    # Run backtest on the out-of-sample dataset (900 bars of test data)
    test_df = df.iloc[100:]
    backtester = EventDrivenBacktester(initial_capital=100000.0, commission_pct=0.0005, slippage_pct=0.0002)

    # P0-5 (audit §4.9) : garde-fou anti-biais identique au live — REJET si échec.
    _bias = audit_backtest(
        test_df,
        assets_universe=["BTCUSDT"],
        assets_tested=["BTCUSDT"],
        slippage_bps=2.0,          # 0.0002 * 10000 (coûts réalistes, jamais 0)
        commission_pct=0.0005,
    )
    if _bias["status"] == "REJECTED":
        print(f"❌ BACKTEST REJETÉ par l'audit des biais : {_bias['issues']}")
        return
    print(f"✅ Audit des biais passé (score {_bias['score']})")

    results = backtester.run(test_df, meta_engine, risk, detector, predictor, ppo)
    
    print(f"Initial Capital : ${results['initial_capital']:.2f}")
    print(f"Final Equity    : ${results['final_equity']:.2f}")
    print(f"Total Return    : {results['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio    : {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown    : {results['max_drawdown_pct']:.2f}%")
    print(f"Total Trades    : {results['total_trades']}")
    print(f"Win Rate        : {results['win_rate_pct']:.2f}%")
    print(f"Profit Factor   : {results['profit_factor']:.2f}")
    print("=========================================================================")
    
    if results['final_equity'] > results['initial_capital']:
        print("✅ SUCCESS: The bot generated positive net returns under realistic slippages & fees!")
    else:
        print("❌ FAILURE: The bot ended with a net loss.")

if __name__ == "__main__":
    run_quant_test()
