"""
Dynamic Risk Scaling (LOT 14)
Ajuste automatiquement les paramètres de risque en fonction :
- Des performances Walk-Forward récentes
- Du régime de marché actuel
"""
import logging
import numpy as np

logger = logging.getLogger("DynamicRisk")

class DynamicRiskScaler:
    """
    Ajuste dynamiquement les paramètres de risque.
    Très institutionnel.
    """
    
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        self.base_params = risk_manager.params.copy()
        self.last_adjustment = 0

    def adjust_risk_parameters(self, walkforward_weights: dict, regime_id: int, 
                               recent_strategy_scores: dict) -> dict:
        """
        Ajuste les paramètres de risque en fonction des données réelles.
        Retourne les nouveaux paramètres appliqués.
        """
        now = __import__('time').time()
        if now - self.last_adjustment < 1800:  # 30 minutes minimum entre ajustements
            return {"adjusted": False, "reason": "Too soon"}
        
        adjustment_factor = 1.0
        reasons = []
        
        # 1. Ajustement selon Walk-Forward
        avg_weight = np.mean(list(walkforward_weights.values())) if walkforward_weights else 0.12
        if avg_weight < 0.09:
            adjustment_factor *= 0.75  # Réduire le risque
            reasons.append("Low strategy performance")
        elif avg_weight > 0.18:
            adjustment_factor *= 1.15  # Augmenter légèrement
            reasons.append("Strong strategy performance")
        
        # 2. Ajustement selon le régime
        if regime_id == 3:  # Haute volatilité
            adjustment_factor *= 0.65
            reasons.append("High volatility regime")
        elif regime_id == 0:  # Bull Trend calme
            adjustment_factor *= 1.10
            reasons.append("Bullish low-vol regime")
        
        # 3. Application des nouveaux paramètres
        new_params = self.base_params.copy()
        
        # Ajustement du max exposure
        new_params['max_exposure_per_asset_pct'] = min(
            0.35, 
            self.base_params['max_exposure_per_asset_pct'] * adjustment_factor
        )
        
        # Ajustement du daily drawdown
        new_params['max_daily_drawdown_pct'] = max(
            0.015,
            self.base_params['max_daily_drawdown_pct'] * (1 / adjustment_factor)
        )
        
        # Mise à jour du RiskManager
        self.risk_manager.params.update(new_params)
        self.last_adjustment = now
        
        logger.info(f"Dynamic Risk Adjusted: {adjustment_factor:.2f}x | Reasons: {', '.join(reasons)}")
        
        return {
            "adjusted": True,
            "adjustment_factor": round(adjustment_factor, 2),
            "new_params": new_params,
            "reasons": reasons
        }