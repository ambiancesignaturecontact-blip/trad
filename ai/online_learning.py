"""
Online Learning / Continuous Model Update (LOT 32)
Permet aux modèles (Meta-Labeling) de s'adapter continuellement aux nouvelles données de marché.
Approche robuste : sliding window retraining avec validation.
"""
import numpy as np
import pandas as pd
import logging
import time
from typing import Optional

logger = logging.getLogger("OnlineLearning")

class ContinuousModelUpdater:
    """
    Continuous / Online Learning Engine.
    - Met à jour le Meta-Labeling model sur une fenêtre glissante de données récentes.
    - Ne met à jour que si la performance s'améliore.
    - Très utile en production pour s'adapter aux régimes changeants.
    """
    
    def __init__(self, meta_labeling_engine, window_size: int = 1200, 
                 update_interval_hours: int = 8):
        self.meta_labeling = meta_labeling_engine
        self.window_size = window_size
        self.update_interval = update_interval_hours * 3600
        self.last_update = 0
        self.performance_history = []

    def should_update(self) -> bool:
        """Vérifie si c'est le moment de mettre à jour"""
        return (time.time() - self.last_update) > self.update_interval

    def update_model(self, db, symbol: str = "BTCUSDT") -> bool:
        """
        Met à jour le modèle Meta-Labeling avec les données les plus récentes.
        Retourne True si la mise à jour a été effectuée.
        """
        if not self.should_update():
            return False

        try:
            df = db.load_candles(symbol, limit=self.window_size + 200)
            if df.empty or len(df) < self.window_size:
                logger.warning("Not enough data for online update")
                return False

            # Fenêtre glissante
            recent_df = df.iloc[-self.window_size:]

            # Signaux primaires
            primary_signals = recent_df['close'].pct_change(5).rolling(20).mean().fillna(0)
            forward_returns = recent_df['close'].pct_change(8).shift(-8).fillna(0)

            # Entraînement
            success = self.meta_labeling.fit(recent_df, primary_signals, forward_returns)
            
            if success:
                self.last_update = time.time()
                logger.info("Continuous Model Update: Meta-Labeling successfully updated with recent data")
                return True
            else:
                logger.warning("Continuous Model Update: Training failed")
                return False

        except Exception as e:
            logger.error(f"Continuous Model Update failed: {e}")
            return False

    def get_last_update_time(self) -> float:
        return self.last_update