"""
Walk-Forward Optimization Automatique - LOT 9 (100% réel)
"""
import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger("WalkForward")

class WalkForwardOptimizer:
    """
    Walk-Forward Optimization 100% basé sur des données réelles.
    Aucune simulation, aucune donnée fictive.
    """
    
    def __init__(self, n_folds: int = 5, train_ratio: float = 0.7):
        self.n_folds = n_folds
        self.train_ratio = train_ratio

    def run_walk_forward(self, df_bars: pd.DataFrame, strategy_engine, risk_manager,
                         regime_detector, price_predictor, ppo_agent) -> Dict:
        """
        Exécute un walk-forward réel sur les données historiques.
        Retourne les métriques agrégées.
        """
        if len(df_bars) < 300:
            return {"status": "INSUFFICIENT_DATA", "message": "Need at least 300 bars"}

        results = []
        fold_size = len(df_bars) // self.n_folds

        for fold in range(self.n_folds - 1):
            train_start = fold * fold_size
            train_end = (fold + 2) * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, len(df_bars))

            if test_end - test_start < 80:
                continue

            train_df = df_bars.iloc[train_start:train_end]
            test_df = df_bars.iloc[test_start:test_end]

            # Entraînement sur la fenêtre d'entraînement
            from backtester.enhanced_engine import EnhancedEventDrivenBacktester
            backtester = EnhancedEventDrivenBacktester(initial_capital=10000)

            train_metrics = backtester.run(
                train_df, strategy_engine, risk_manager,
                regime_detector, price_predictor, ppo_agent
            )

            # Test sur la fenêtre suivante (out-of-sample)
            test_metrics = backtester.run(
                test_df, strategy_engine, risk_manager,
                regime_detector, price_predictor, ppo_agent
            )

            results.append({
                "fold": fold,
                "train_sharpe": train_metrics.get("sharpe_ratio", 0),
                "test_sharpe": test_metrics.get("sharpe_ratio", 0),
                "train_return": train_metrics.get("total_return_pct", 0),
                "test_return": test_metrics.get("total_return_pct", 0),
                "test_drawdown": test_metrics.get("max_drawdown_pct", 0)
            })

        if not results:
            return {"status": "NO_VALID_FOLDS"}

        avg_test_sharpe = np.mean([r["test_sharpe"] for r in results])
        avg_test_return = np.mean([r["test_return"] for r in results])

        logger.info(f"Walk-Forward completed: Avg Test Sharpe = {avg_test_sharpe:.3f}")

        return {
            "status": "SUCCESS",
            "n_folds": len(results),
            "avg_test_sharpe": round(avg_test_sharpe, 3),
            "avg_test_return": round(avg_test_return, 2),
            "folds": results,
            "recommendation": "AGGRESSIVE" if avg_test_sharpe > 1.5 else "CONSERVATIVE"
        }