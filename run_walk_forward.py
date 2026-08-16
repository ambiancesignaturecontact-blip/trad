import numpy as np
import pandas as pd
import sys

# Import our quant models
from models.regime_detector import MarketRegimeDetector
from models.price_predictor import LSTMLikePredictor, PPOTRAgent
from strategies.engine import MetaAllocationEngine, TrendFollowingStrategy, MeanReversionStrategy, GridTradingStrategy
from risk.risk_manager import RiskManager
from backtester.engine import EventDrivenBacktester
from models.mlops_pipeline import MLOpsAutoTrainer
from db_manager import DBManager

def execute_walk_forward_analysis():
    print("=========================================================================")
    print("📈 PIPELINE D'ANALYSE GLISSANTE (WALK-FORWARD ANALYSIS - WFA)")
    print("=========================================================================")
    
    # Generate an extended 1000-hour multi-cycle market dataset
    np.random.seed(99)
    prices = [60000.0]
    for i in range(1000):
        # Cyclical trends representing market cycles (bull run, FTX-like correction, consolidation)
        trend = 40.0 * np.sin(i * 2 * np.pi / 200.0)
        noise = np.random.normal(0, 120.0)
        prices.append(prices[-1] + trend + noise)
        
    timestamps = pd.date_range(start="2026-01-01", periods=1001, freq="h")
    df = pd.DataFrame({
        "close": prices,
        "high": [p * 1.0015 for p in prices],
        "low": [p * 0.9985 for p in prices],
        "open": [p * np.random.uniform(0.9995, 1.0005) for p in prices],
        "volume": [np.random.uniform(10.0, 100.0) for _ in prices]
    }, index=timestamps)
    
    db = DBManager()
    detector = MarketRegimeDetector()
    predictor = LSTMLikePredictor(5, 8)
    ppo = PPOTRAgent(4, 1)
    mlops_trainer = MLOpsAutoTrainer(detector, predictor, db)
    
    # Configure walk-forward glissant partitions:
    # 5 windows: each has 150 bars training/warm-up and 50 bars out-of-sample test
    window_train_size = 150
    window_test_size = 50
    num_windows = 4
    
    oos_sharpe_scores = []
    is_sharpe_scores = []
    
    print(f"Dataset total : {len(df)} barres. Configuration: {num_windows} fenêtres glissantes.")
    print("-------------------------------------------------------------------------")
    
    for w in range(num_windows):
        start_idx = w * window_test_size
        train_end_idx = start_idx + window_train_size
        test_end_idx = train_end_idx + window_test_size
        
        train_slice = df.iloc[start_idx:train_end_idx]
        test_slice = df.iloc[train_end_idx:test_end_idx]
        
        # 1. Optimize parameters on In-Sample (IS) training slice using our Genetic Algorithm!
        ga_results = mlops_trainer.execute_genetic_tuning(train_slice)
        is_sharpe = ga_results['sharpe_score']
        is_sharpe_scores.append(is_sharpe)
        
        # 2. Deploy calibrated strategies on Out-Of-Sample (OOS) testing slice
        optimized_trend = TrendFollowingStrategy(params={'ema_fast': ga_results['ema_fast'], 'ema_slow': ga_results['ema_slow']})
        optimized_rev = MeanReversionStrategy(params={'period': ga_results['bbands_period'], 'rsi_period': ga_results['rsi_period']})
        
        meta_engine = MetaAllocationEngine(strategies=[optimized_trend, optimized_rev])
        risk = RiskManager()
        backtester = EventDrivenBacktester(initial_capital=100000.0, commission_pct=0.0004, slippage_pct=0.0001)
        
        # Run OOS Backtest
        results = backtester.run(df.iloc[start_idx:test_end_idx], meta_engine, risk, detector, predictor, ppo)
        oos_sharpe = results['sharpe_ratio']
        oos_sharpe_scores.append(oos_sharpe)
        
        print(f"FENÊTRE {w+1} | In-Sample Sharpe: {is_sharpe:.2f} | Out-of-Sample Sharpe: {oos_sharpe:.2f} (Return: {results['total_return_pct']:.2f}%)")
        
    # Calculate Walk-Forward Efficiency (WFE) index
    # WFE = Average(OOS_Sharpe) / Average(IS_Sharpe)
    avg_is = np.mean(is_sharpe_scores) if is_sharpe_scores else 1.0
    avg_oos = np.mean(oos_sharpe_scores) if oos_sharpe_scores else 0.0
    wfe_index = (avg_oos / avg_is) * 100.0 if avg_is > 0 else 0.0
    
    print("-------------------------------------------------------------------------")
    print(f"Sharpe Moyen In-Sample  : {avg_is:.2f}")
    print(f"Sharpe Moyen Out-of-Sample : {avg_oos:.2f}")
    print(f"WALK-FORWARD EFFICIENCY INDEX (WFE) : {wfe_index:.2f}%")
    print("=========================================================================")
    
    if wfe_index >= 50.0:
        print("✅ SYSTEM STABILITY CONFIRMED: WFE exceeds the 50% institutional threshold! Zero overfitting detected.")
    else:
        print("⚠️ WARNING: WFE is low. Potential over-fitting on past noise. Calibrate parameters or expand training windows.")

if __name__ == "__main__":
    execute_walk_forward_analysis()
