"""
Meta-Labeling Engine (LOT 21)
Implémentation inspirée de Marcos Lopez de Prado.
Prédit si un signal primaire a de fortes chances d'être rentable.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging
from typing import Dict

logger = logging.getLogger("MetaLabeling")

class MetaLabelingEngine:
    """
    Meta-Labeling.
    - Primary model : génère des signaux (déjà fait avec nos 9 stratégies)
    - Secondary model : prédit si le signal est "bon" (rentable) ou non
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            class_weight="balanced"
        )
        self.is_fitted = False
        self.feature_importance = {}

    def create_meta_labels(self, df: pd.DataFrame, signals: pd.Series, 
                          forward_returns: pd.Series, threshold: float = 0.002) -> pd.Series:
        """
        Crée les labels pour le modèle secondaire.
        Label = 1 si le signal a généré un retour > threshold dans la bonne direction.
        """
        labels = []
        
        for i in range(len(df) - 5):
            signal = signals.iloc[i]
            future_ret = forward_returns.iloc[i:i+5].mean()
            
            if signal > 0.1 and future_ret > threshold:
                labels.append(1)
            elif signal < -0.1 and future_ret < -threshold:
                labels.append(1)
            else:
                labels.append(0)
        
        # Padding
        labels = labels + [0] * (len(df) - len(labels))
        return pd.Series(labels, index=df.index)

    def fit(self, df: pd.DataFrame, primary_signals: pd.Series, 
            forward_returns: pd.Series):
        """
        Entraîne le modèle de Meta-Labeling sur données réelles.
        """
        try:
            labels = self.create_meta_labels(df, primary_signals, forward_returns)
            
            # Features simples
            features = pd.DataFrame({
                'signal_strength': primary_signals.abs(),
                'volatility_5': df['close'].pct_change().rolling(5).std(),
                'momentum_10': df['close'].pct_change(10),
                'volume_ratio': df['volume'] / df['volume'].rolling(20).mean()
            }).dropna()
            
            # Alignement
            common_idx = features.index.intersection(labels.index)
            X = features.loc[common_idx]
            y = labels.loc[common_idx]
            
            if len(X) < 200:
                logger.warning("Not enough data for Meta-Labeling")
                return False
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, shuffle=False
            )
            
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            
            # Feature importance
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            
            acc = self.model.score(X_test, y_test)
            logger.info(f"Meta-Labeling trained. Test accuracy: {acc:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Meta-Labeling training failed: {e}")
            return False

    def predict_signal_quality(self, current_features: Dict) -> float:
        """
        Retourne la probabilité que le signal actuel soit rentable (0 à 1).
        """
        if not self.is_fitted:
            return 0.5  # Neutre par défaut
        
        try:
            X = np.array([[
                current_features.get('signal_strength', 0),
                current_features.get('volatility_5', 0.01),
                current_features.get('momentum_10', 0),
                current_features.get('volume_ratio', 1.0)
            ]])
            
            prob = self.model.predict_proba(X)[0][1]
            return float(prob)
            
        except Exception as e:
            logger.warning(f"Meta-Labeling prediction error: {e}")
            return 0.5

    def get_feature_importance(self) -> Dict:
        return self.feature_importance.copy()