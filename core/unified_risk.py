"""
Unified Dynamic Risk Management Layer (LOT 35)
Central engine that applies ALL risk constraints in one place.
This is the "brain" of institutional risk management.
"""
import logging

import pandas as pd

logger = logging.getLogger("UnifiedRisk")

class UnifiedRiskManager:
    """
    Unified Risk Management Layer.
    Applies in sequence:
    1. Regime-Switching
    2. Multi-Timeframe Consensus
    3. Meta-Labeling Quality Filter
    4. Kelly Fraction (per strategy + regime)
    5. CVaR Portfolio Constraint
    6. Strategy Correlation Hedging
    7. On-Chain Alpha Filter
    """

    def __init__(self,
                 regime_switcher,
                 mtf_consensus,
                 meta_labeling,
                 kelly_sizer,
                 cvar_optimizer,
                 correlation_risk,
                 onchain_tracker):

        self.regime_switcher = regime_switcher
        self.mtf_consensus = mtf_consensus
        self.meta_labeling = meta_labeling
        self.kelly_sizer = kelly_sizer
        self.cvar_optimizer = cvar_optimizer
        self.correlation_risk = correlation_risk
        self.onchain_tracker = onchain_tracker

    def apply_all_risk_filters(self,
                               symbol: str,
                               base_signal: float,
                               current_price: float,
                               regime_id: int,
                               strategy_name: str,
                               recent_scores: list,
                               positions: list,
                               returns_dict: dict,
                               corr_matrix: pd.DataFrame,
                               onchain_risk: float,
                               db) -> tuple[float, dict]:
        """
        Applies all risk layers sequentially.
        Returns (final_signal, risk_report)
        """
        signal = base_signal
        report = {"symbol": symbol, "layers": []}

        # 1. Regime-Switching
        signal = self.regime_switcher.apply_regime_switching(signal, regime_id, strategy_name)
        report["layers"].append({"layer": "Regime-Switching", "signal": round(signal, 3)})

        # 2. Multi-Timeframe Consensus
        mtf = self.mtf_consensus.check_consensus(symbol, current_price, signal, db)
        signal = mtf["adjusted_signal"]
        report["layers"].append({
            "layer": "Multi-Timeframe",
            "signal": round(signal, 3),
            "consensus": mtf["consensus_score"]
        })

        # 3. Meta-Labeling Quality Filter
        if hasattr(self.meta_labeling, 'is_fitted') and self.meta_labeling.is_fitted:
            meta_features = {
                'signal_strength': abs(signal),
                'volatility_5': 0.01,
                'momentum_10': 0.0,
                'volume_ratio': 1.0
            }
            quality = self.meta_labeling.predict_signal_quality(meta_features)
            if quality < 0.35:
                signal *= 0.3
            report["layers"].append({
                "layer": "Meta-Labeling",
                "signal": round(signal, 3),
                "quality": round(quality, 2)
            })

        # 4. Kelly Fraction (per strategy + regime)
        kelly_mult = self.kelly_sizer.get_position_size_multiplier(strategy_name, recent_scores, regime_id)
        signal *= kelly_mult
        report["layers"].append({
            "layer": "Kelly Sizing",
            "signal": round(signal, 3),
            "kelly_mult": round(kelly_mult, 2)
        })

        # 5. CVaR Portfolio Constraint
        try:
            current_cvar = self.cvar_optimizer.calculate_current_cvar(positions, returns_dict, corr_matrix)
            cvar_mult = self.cvar_optimizer.get_cvar_constrained_size(current_cvar, 1.0, symbol)
            signal *= cvar_mult
            report["layers"].append({
                "layer": "CVaR Constraint",
                "signal": round(signal, 3),
                "cvar": round(current_cvar, 4)
            })
        except Exception:
            pass

        # 6. Strategy Correlation Hedging
        try:
            avg_corr = self.correlation_risk.calculate_strategy_correlation_matrix()
            corr_mult = self.correlation_risk.get_hedging_multiplier(avg_corr)
            signal *= corr_mult
            report["layers"].append({
                "layer": "Correlation Hedging",
                "signal": round(signal, 3),
                "avg_corr": round(avg_corr, 2)
            })
        except Exception:
            pass

        # 7. On-Chain Alpha Filter
        if onchain_risk > 0.78:
            signal *= 0.55
        elif onchain_risk < 0.32:
            signal *= 1.15
        report["layers"].append({
            "layer": "On-Chain Alpha",
            "signal": round(signal, 3),
            "onchain_risk": round(onchain_risk, 2)
        })

        report["final_signal"] = round(signal, 3)
        return signal, report
