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
from backtester.honest_verdict import print_honest_result

def run_profitability_calibration():
    print("=========================================================================")
    print("🎯 CALIBRATION ET OPTIMISATION DE RENTABILITÉ SUR MARCHÉ RÉEL")
    print("=========================================================================")
    
    # Load actual real-world historical market data!
    # (OKX -> Coinbase -> Kraken -> Binance ; plus AUCUN fallback synthétique —
    # sans données réelles, pas de preuve. P0-5, audit §4.9.)
    df, _src = fetch_real_candles("BTCUSDT", limit=500)
    if df is None:
        return
    
    # 2. Instantiate and fit models
    detector = MarketRegimeDetector()
    predictor = LSTMLikePredictor(5, 24)  # P0-5 : même archi que le live (audit §4.9)
    ppo = PPOTRAgent(4, 1)
    
    train_df = df.iloc[:100]
    returns = train_df['close'].pct_change().dropna().values
    vols = train_df['close'].pct_change().rolling(10).std().dropna().values
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
    strategies = [
        TrendFollowingStrategy(params={'ema_fast': 10, 'ema_slow': 20, 'breakout_period': 15}),
        MeanReversionStrategy(params={'period': 15, 'num_std': 2.0}),
        GridTradingStrategy(params={'grid_levels': 5, 'atr_multiplier': 1.2})
    ]
    meta_engine = MetaAllocationEngine(strategies=strategies)
    
    risk = RiskManager(params={
        'max_daily_drawdown_pct': 0.05,
        'max_total_drawdown_pct': 0.10,
        'max_exposure_per_asset_pct': 0.35,
        'fractional_kelly_multiplier': 0.25,
        'deviation_limit_pct': 0.05
    })
    
    # 4. Run Backtest
    backtester = EventDrivenBacktester(initial_capital=100000.0, commission_pct=0.0004, slippage_pct=0.0001)

    # P0-5 (audit §4.9) : garde-fou anti-biais identique au live — REJET si échec.
    _bias = audit_backtest(
        df,
        assets_universe=["BTCUSDT"],
        assets_tested=["BTCUSDT"],
        slippage_bps=1.0,          # 0.0001 * 10000 (coûts réalistes, jamais 0)
        commission_pct=0.0004,
    )
    if _bias["status"] == "REJECTED":
        print(f"❌ BACKTEST REJETÉ par l'audit des biais : {_bias['issues']}")
        return
    print(f"✅ Audit des biais passé (score {_bias['score']})")

    results = backtester.run(df, meta_engine, risk, detector, predictor, ppo)
    
    print("\n📊 RAPPORT FINANCIER D'OPTIMISATION DE L'IA :")
    print("-------------------------------------------------------------------------")
    print(f"Capital de départ     : ${results['initial_capital']:.2f}")
    print(f"Capital Final (Equity): ${results['final_equity']:.2f}")
    print(f"Rendement Net (PnL)   : ${results['final_equity'] - results['initial_capital']:.2f} ({results['total_return_pct']:.2f}%)")
    print(f"Sharpe Ratio          : {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown          : {results['max_drawdown_pct']:.2f}%")
    print(f"Nombre total d'ordres : {results['total_trades']}")
    print(f"Taux de réussite      : {results['win_rate_pct']:.2f}%")
    print(f"Facteur de profit     : {results['profit_factor']:.2f}")
    print("-------------------------------------------------------------------------")
    print_honest_result(
        results['initial_capital'], results['final_equity'],
        label="optimisation 3 stratégies", bars=len(df),
        source=_src, start=df.index[0], end=df.index[-1])
    print("(Un résultat positif sur la période d'optimisation ne prouve pas la "
          "généralisation : seule une validation hors-échantillon le ferait.)")

if __name__ == "__main__":
    run_profitability_calibration()
