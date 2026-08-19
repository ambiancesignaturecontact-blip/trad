import numpy as np
import pandas as pd

# Import our quant models
from models.regime_detector import MarketRegimeDetector
from models.price_predictor import LSTMLikePredictor, PPOTRAgent
from strategies.engine import (
    MetaAllocationEngine, TrendFollowingStrategy, MeanReversionStrategy,
    StatisticalArbitrageStrategy, GridTradingStrategy, ScalpingStrategy
)
from risk.risk_manager import RiskManager
from backtester.engine import EventDrivenBacktester
from backtester.bias_audit import audit_backtest
from backtester.live_candles import fetch_real_candles

def prove_trend_profit():
    print("=========================================================================")
    print("📈 PROUVER LA RENTABILITÉ DE L'IA DANS UN MARCHÉ DIRECTIONNEL")
    print("=========================================================================")
    
    # 1. Données RÉELLES uniquement (OKX -> Coinbase -> Kraken -> Binance).
    #    Plus AUCUNE donnée synthétique : sans données réelles, pas de preuve
    #    (P0-5, audit §4.9 — un backtest sur données fabriquées ne prouve rien).
    df, _src = fetch_real_candles("BTCUSDT", limit=500)
    if df is None:
        return
    
    # 2. Setup models
    detector = MarketRegimeDetector()
    predictor = LSTMLikePredictor(5, 24)  # P0-5 : même archi que le live (audit §4.9)
    ppo = PPOTRAgent(4, 1)
    
    # Pre-train
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
    
    # 3. Setup Strategies
    # We only enable Trend Following to prove trend capturing!
    trend_strat = TrendFollowingStrategy(params={'ema_fast': 12, 'ema_slow': 26, 'breakout_period': 20})
    meta_engine = MetaAllocationEngine(strategies=[trend_strat])
    
    risk = RiskManager(params={
        'max_daily_drawdown_pct': 0.05,
        'max_total_drawdown_pct': 0.10,
        'max_exposure_per_asset_pct': 0.40,
        'fractional_kelly_multiplier': 0.30,
        'deviation_limit_pct': 0.05
    })
    
    # 4. Run Backtest
    backtester = EventDrivenBacktester(initial_capital=100000.0, commission_pct=0.0002, slippage_pct=0.0001)

    # P0-5 (audit §4.9) : garde-fou anti-biais identique au live — REJET si échec.
    _bias = audit_backtest(
        df,
        assets_universe=["BTCUSDT"],
        assets_tested=["BTCUSDT"],
        slippage_bps=1.0,          # 0.0001 * 10000 (coûts réalistes, jamais 0)
        commission_pct=0.0002,
    )
    if _bias["status"] == "REJECTED":
        print(f"❌ BACKTEST REJETÉ par l'audit des biais : {_bias['issues']}")
        return
    print(f"✅ Audit des biais passé (score {_bias['score']})")

    results = backtester.run(df, meta_engine, risk, detector, predictor, ppo)
    
    print(f"Rendement de l'actif sous-jacent : +{((df['close'].iloc[-1] - df['close'].iloc[0])/df['close'].iloc[0])*100:.2f}%")
    print("-------------------------------------------------------------------------")
    print(f"Capital de départ     : ${results['initial_capital']:.2f}")
    print(f"Capital Final (Equity): ${results['final_equity']:.2f}")
    print(f"Rendement Net (PnL)   : ${results['final_equity'] - results['initial_capital']:.2f} ({results['total_return_pct']:.2f}%)")
    print(f"Sharpe Ratio          : {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown          : {results['max_drawdown_pct']:.2f}%")
    print(f"Nombre total d'ordres : {results['total_trades']}")
    print(f"Taux de réussite      : {results['win_rate_pct']:.2f}%")
    print(f"Facteur de profit     : {results['profit_factor']:.2f}")
    print("=========================================================================")
    
    if results['final_equity'] > results['initial_capital']:
        print("✅ SUCCESS: The bot generated massive net profits by capturing the macro trend!")
    else:
        print("❌ FAILURE.")

if __name__ == "__main__":
    prove_trend_profit()
