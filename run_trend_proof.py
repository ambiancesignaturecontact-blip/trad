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


def prove_trend_profit():
    print("=========================================================================")
    print("📈 TESTER LA CAPTURE DE TENDANCE SUR MARCHÉ RÉEL (MULTI-ACTIFS)")
    print("=========================================================================")
    print("Données RÉELLES uniquement (OKX -> Coinbase -> Kraken -> Binance).")
    print("Aucune donnée synthétique : les conclusions sont celles du marché réel.")
    print("=========================================================================")

    # 3 actifs réels : la conclusion ne dépend plus d'un seul marché.
    assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    results_by_asset = {}

    for symbol in assets:
        print(f"\n{'=' * 70}\n=== ACTIF : {symbol} ===\n{'=' * 70}")
        df, src = fetch_real_candles(symbol, limit=600)
        if df is None:
            continue

        detector = MarketRegimeDetector()
        predictor = LSTMLikePredictor(5, 24)  # P0-5 : même archi que le live
        ppo = PPOTRAgent(4, 1)

        # Pre-train sur les 100 premières barres réelles
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

        # Stratégie trend uniquement (test de capture de tendance)
        trend_strat = TrendFollowingStrategy(params={'ema_fast': 12, 'ema_slow': 26, 'breakout_period': 20})
        meta_engine = MetaAllocationEngine(strategies=[trend_strat])

        risk = RiskManager(params={
            'max_daily_drawdown_pct': 0.05,
            'max_total_drawdown_pct': 0.10,
            'max_exposure_per_asset_pct': 0.40,
            'fractional_kelly_multiplier': 0.30,
            'deviation_limit_pct': 0.05
        })

        backtester = EventDrivenBacktester(initial_capital=100000.0, commission_pct=0.0002, slippage_pct=0.0001)

        # P0-5 (audit §4.9) : garde-fou anti-biais identique au live — REJET si échec.
        _bias = audit_backtest(
            df,
            assets_universe=[symbol],
            assets_tested=[symbol],
            slippage_bps=1.0,          # 0.0001 * 10000 (coûts réalistes, jamais 0)
            commission_pct=0.0002,
        )
        if _bias["status"] == "REJECTED":
            print(f"❌ BACKTEST REJETÉ par l'audit des biais : {_bias['issues']}")
            continue
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
        print("-------------------------------------------------------------------------")
        cat = print_honest_result(
            results['initial_capital'], results['final_equity'],
            label=f"TrendFollowing {symbol}", bars=len(df),
            source=src, start=df.index[0], end=df.index[-1])
        results_by_asset[symbol] = cat

    print("\n=== SYNTHÈSE MULTI-ACTIFS (données réelles) ===")
    for symbol, cat in results_by_asset.items():
        print(f"  {symbol}: {cat}")
    if results_by_asset and all(c == "profit" for c in results_by_asset.values()):
        print("✅ La stratégie trend a été rentable (>= 1%) sur TOUS les actifs testés.")
    elif results_by_asset:
        print("⚠️ Pas de preuve de rentabilité généralisable sur cette période réelle "
              "(le marché ne donne pas de tendance uniforme).")


if __name__ == "__main__":
    prove_trend_profit()
