
import numpy as np

from backtester.bias_audit import audit_backtest
from backtester.engine import EventDrivenBacktester
from backtester.honest_verdict import print_honest_result
from backtester.live_candles import fetch_real_candles
from models.price_predictor import LSTMLikePredictor, PPOTRAgent

# Import our quant models
from models.regime_detector import MarketRegimeDetector
from risk.risk_manager import RiskManager
from strategies.engine import MeanReversionStrategy, MetaAllocationEngine, TrendFollowingStrategy


def run_extended_cycles_backtest():
    print("=========================================================================")
    print("🌋 SIMULATION DE BACKTESTING SUR CYCLES DE MARCHÉ REELS (MULTI-SOURCES)")
    print("=========================================================================")

    # Load actual real-world historical market data!
    # (OKX -> Coinbase -> Kraken -> Binance ; Binance seul est géobloqué en
    # France. Plus AUCUN fallback synthétique : sans données réelles, pas de
    # preuve — P0-5, audit §4.9.)
    df, _src = fetch_real_candles("BTCUSDT", limit=500)
    if df is None:
        return

    detector = MarketRegimeDetector()
    predictor = LSTMLikePredictor(5, 24)  # P0-5 : même archi que le live (audit §4.9)
    ppo = PPOTRAgent(4, 1)

    # Pre-train HMM on early bars
    train_df = df.iloc[:100]
    returns = train_df['close'].pct_change().dropna().values
    vols = train_df['close'].pct_change().rolling(10).std().dropna().values
    min_l = min(len(returns), len(vols))
    detector.fit(np.column_stack((returns[-min_l:], vols[-min_l:])))

    # Setup strategies
    trend = TrendFollowingStrategy()
    rev = MeanReversionStrategy()
    meta_engine = MetaAllocationEngine(strategies=[trend, rev])
    risk = RiskManager()

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

    print(f"Rendement global de l'actif : {((df['close'].iloc[-1] - df['close'].iloc[0])/df['close'].iloc[0])*100:.2f}%")
    print("-------------------------------------------------------------------------")
    print(f"Capital Initial   : ${results['initial_capital']:.2f}")
    print(f"Capital Final     : ${results['final_equity']:.2f}")
    print(f"Rendement Net     : ${results['final_equity'] - results['initial_capital']:.2f} ({results['total_return_pct']:.2f}%)")
    print(f"Sharpe Ratio      : {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown      : {results['max_drawdown_pct']:.2f}%")
    print(f"Total des ordres  : {results['total_trades']}")
    print(f"Taux de réussite  : {results['win_rate_pct']:.2f}%")
    print(f"Profit Factor     : {results['profit_factor']:.2f}")
    print("=========================================================================")
    print_honest_result(
        results['initial_capital'], results['final_equity'],
        label="cycles réels multi-sources", bars=len(df),
        source=_src, start=df.index[0], end=df.index[-1])

if __name__ == "__main__":
    run_extended_cycles_backtest()
