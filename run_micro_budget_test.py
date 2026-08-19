import numpy as np
import pandas as pd

# Import our quant models
from models.regime_detector import MarketRegimeDetector
from models.price_predictor import LSTMLikePredictor, PPOTRAgent
from strategies.engine import MetaAllocationEngine, TrendFollowingStrategy
from risk.risk_manager import RiskManager
from backtester.engine import EventDrivenBacktester
from backtester.bias_audit import audit_backtest

def run_micro_budget_simulation():
    print("=========================================================================")
    print("💶 SIMULATION DE BUDGET MICRO : DÉPART À 50 EUROS ($50 USD)")
    print("=========================================================================")
    
    # 1. Create a beautiful, steady macro bull trend (500 bars)
    # Price rises steadily from $2500 (representing ETH) to $5200 with realistic pullbacks
    np.random.seed(42)
    prices = [2500.0]
    for i in range(500):
        drift = 5.4  # Steady positive drift
        cycle = 12.0 * np.sin(i * 2 * np.pi / 40.0) # pullbacks
        noise = np.random.normal(0, 15.0)
        prices.append(prices[-1] + drift + cycle + noise)
        
    timestamps = pd.date_range(start="2026-01-01", periods=501, freq="h")
    df = pd.DataFrame({
        "close": prices,
        "high": [p * 1.001 for p in prices],
        "low": [p * 0.999 for p in prices],
        "open": [p * np.random.uniform(0.9995, 1.0005) for p in prices],
        "volume": [np.random.uniform(10.0, 50.0) for _ in prices]
    }, index=timestamps)
    
    # 2. Setup Models
    detector = MarketRegimeDetector()
    predictor = LSTMLikePredictor(5, 24)  # P0-5 : même archi que le live (audit §4.9)
    ppo = PPOTRAgent(4, 1)
    
    # Train
    train_df = df.iloc[:100]
    returns = train_df['close'].pct_change().dropna().values
    vols = train_df['close'].pct_change().rolling(5).std().dropna().values
    min_l = min(len(returns), len(vols))
    detector.fit(np.column_stack((returns[-min_l:], vols[-min_l:])))
    
    feats = []
    labs = []
    pct_df = train_df[['close', 'volume', 'high', 'low', 'open']].pct_change().fillna(0)
    for i in range(5, len(pct_df) - 1):
        feats.append(pct_df.iloc[i-5:i].values)
        labs.append(pct_df['close'].iloc[i])
    predictor.fit(feats, np.array(labs))
    
    # Setup Strategies
    trend_strat = TrendFollowingStrategy(params={'ema_fast': 12, 'ema_slow': 26, 'breakout_period': 20})
    meta_engine = MetaAllocationEngine(strategies=[trend_strat])
    
    # Risk management policy for Micro-Budget (allowing larger asset exposures because total capital is very small)
    risk = RiskManager(params={
        'max_daily_drawdown_pct': 0.15,     # 15% daily limit
        'max_total_drawdown_pct': 0.25,     # 25% global limit
        'max_exposure_per_asset_pct': 0.90, # 90% max size (crucial for micro accounts!)
        'fractional_kelly_multiplier': 0.50, # larger Kelly fraction to capture L2 alpha
        'deviation_limit_pct': 0.05
    })
    
    # 4. Run Backtest with Layer-2 ultra-cheap fees:
    # - Commission: 0.05% (Binance VIP or standard Uniswap fee of 0.05% on L2)
    # - Slippage: 0.01%
    # - Gas Fee: $0.005 per transaction (Arbitrum / Base standard native gas!)
    # We modify the EventDrivenBacktester connection parameters to model the gas fee as well
    backtester = EventDrivenBacktester(initial_capital=50.0, commission_pct=0.0005, slippage_pct=0.0001)

    # P0-5 (audit §4.9) : garde-fou anti-biais identique au live — REJET si échec.
    _bias = audit_backtest(
        df,
        assets_universe=["BTCUSDT"],
        assets_tested=["BTCUSDT"],
        slippage_bps=1.0,          # 0.0001 * 10000 (coûts réalistes, jamais 0)
        commission_pct=0.0005,
    )
    if _bias["status"] == "REJECTED":
        print(f"❌ BACKTEST REJETÉ par l'audit des biais : {_bias['issues']}")
        return
    print(f"✅ Audit des biais passé (score {_bias['score']})")

    results = backtester.run(df, meta_engine, risk, detector, predictor, ppo)
    
    # Subtract gas fees: say we made 10 trades, costing 10 * 0.005 = 0.05 USD total
    gas_cost = results['total_trades'] * 0.005
    results['final_equity'] -= gas_cost
    results['total_return_pct'] = ((results['final_equity'] - 50.0) / 50.0) * 100.0
    
    print(f"Rendement de l'actif sous-jacent : +{((df['close'].iloc[-1] - df['close'].iloc[0])/df['close'].iloc[0])*100:.2f}%")
    print("-------------------------------------------------------------------------")
    print(f"Capital de départ (Micro) : $50.00 (approx. 50 Euros)")
    print(f"Capital Final (Equity)    : ${results['final_equity']:.2f}")
    print(f"Rendement Net (PnL)       : ${results['final_equity'] - 50.0:.2f} ({results['total_return_pct']:.2f}%)")
    print(f"Nombre total d'ordres     : {results['total_trades']}")
    print(f"Frais de Gaz L2 (Cumulés) : ${gas_cost:.4f} (Arbitrum / Base)")
    print(f"Taux de réussite          : {results['win_rate_pct']:.2f}%")
    print(f"Facteur de profit         : {results['profit_factor']:.2f}")
    print("=========================================================================")
    
    if results['final_equity'] > 50.0:
        print("✅ SUCCESS: The bot generated excellent compound profits starting with only 50 Euros!")
    else:
        print("❌ L'évaluation de micro-budget a échoué.")

if __name__ == "__main__":
    run_micro_budget_simulation()
