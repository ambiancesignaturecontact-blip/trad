"""
Adversarial Validation Engine (LOT 29)
Detects distribution shift between training and live data.
If the model can distinguish "historical" vs "recent" data too well,
it means concept drift is occurring.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import logging
from typing import Tuple

logger = logging.getLogger("AdversarialValidation")

class AdversarialValidator:
    """
    Adversarial Validation.
    Trains a classifier to distinguish between old and new data.
    High AUC = distribution shift detected → model may be stale.
    """
    
    def __init__(self, n_estimators: int = 100):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            min_samples_leaf=30,
            random_state=42
        )
        self.last_auc = 0.5

    def run_validation(self, historical_df: pd.DataFrame, 
                       recent_df: pd.DataFrame, 
                       features: list = None) -> Tuple[float, bool]:
        """
        Runs adversarial validation.
        Returns (AUC, is_shift_detected)
        """
        if historical_df.empty or recent_df.empty:
            return 0.5, False

        try:
            # Prepare data
            hist = historical_df.copy()
            recent = recent_df.copy()
            
            hist['is_recent'] = 0
            recent['is_recent'] = 1
            
            combined = pd.concat([hist, recent], ignore_index=True)
            
            if features is None:
                # Use price-based features
                features = ['close', 'volume', 'high', 'low']
                for col in features:
                    if col not in combined.columns:
                        combined[col] = combined['close'] if 'close' in combined.columns else 0
            
            X = combined[features].fillna(0)
            y = combined['is_recent']
            
            # Cross-validation AUC
            scores = cross_val_score(self.model, X, y, cv=3, scoring='roc_auc')
            auc = np.mean(scores)
            
            self.last_auc = auc
            is_shift = auc > 0.68  # Threshold for shift detection
            
            if is_shift:
                logger.warning(f"Adversarial Validation: DISTRIBUTION SHIFT DETECTED (AUC={auc:.3f})")
            else:
                logger.info(f"Adversarial Validation: Data distribution stable (AUC={auc:.3f})")
            
            return float(auc), is_shift
            
        except Exception as e:
            logger.error(f"Adversarial Validation failed: {e}")
            return 0.5, False

    def get_last_auc(self) -> float:
        return self.last_auc