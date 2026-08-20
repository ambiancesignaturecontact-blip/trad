
import numpy as np

from backtester.bias_audit import audit_backtest
from backtester.engine import EventDrivenBacktester
from backtester.live_candles import fetch_real_candles
from database.db_manager import DBManager
from models.mlops_pipeline import MLOpsAutoTrainer
from models.price_predictor import LSTMLikePredictor, PPOTRAgent

# Import our quant models
from models.regime_detector import MarketRegimeDetector
from risk.risk_manager import RiskManager
from strategies.engine import MeanReversionStrategy, MetaAllocationEngine, TrendFollowingStrategy


def execute_walk_forward_analysis():
    print("=========================================================================")
    print("📈 PIPELINE D'ANALYSE GLISSANTE (WALK-FORWARD ANALYSIS - WFA)")
    print("=========================================================================")

    # Données RÉELLES uniquement (OKX -> Coinbase -> Kraken -> Binance).
    # Plus AUCUNE donnée synthétique : sans données réelles, pas de preuve
    # (P0-5, audit §4.9 — un backtest sur données fabriquées ne prouve rien).
    df, _src = fetch_real_candles("BTCUSDT", limit=500)
    if df is None:
        return

    db = DBManager()
    detector = MarketRegimeDetector()
    predictor = LSTMLikePredictor(5, 24)  # P0-5 : même archi que le live (audit §4.9)
    ppo = PPOTRAgent(4, 1)
    mlops_trainer = MLOpsAutoTrainer(detector, predictor, db)

    # P0-5 (audit §4.9) : garde-fou anti-biais identique au live — un backtest
    # qui échoue à l'audit (look-ahead/survivorship/slippage) est REJETÉ :
    # aucune preuve de rentabilité valide pour le système réellement déployé.
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

    # Configure walk-forward glissant partitions:
    # fenêtres de 150 barres d'entraînement + 50 barres OOS, adapté au nombre
    # de barres réelles disponibles (OKX plafonne à 300 barres par requête).
    window_train_size = 150
    window_test_size = 50
    num_windows = max(1, min(4, (len(df) - window_train_size) // window_test_size - 1))

    oos_sharpe_scores = []
    is_sharpe_scores = []

    print(f"Dataset total : {len(df)} barres. Configuration: {num_windows} fenêtres glissantes.")
    print("-------------------------------------------------------------------------")

    for w in range(num_windows):
        start_idx = w * window_test_size
        train_end_idx = start_idx + window_train_size
        test_end_idx = train_end_idx + window_test_size

        train_slice = df.iloc[start_idx:train_end_idx]

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
        print("✅ WFE >= 50% : le modèle généralise sur données réelles hors-échantillon.")
    elif wfe_index >= 0.0:
        print("⚠️ WFE faible : risque d'overfitting sur le bruit passé. Calibrer ou élargir les fenêtres.")
    else:
        print("❌ WFE NÉGATIF : overfitting hors-échantillon — AUCUNE preuve de "
              "généralisation sur données réelles.")

if __name__ == "__main__":
    execute_walk_forward_analysis()
