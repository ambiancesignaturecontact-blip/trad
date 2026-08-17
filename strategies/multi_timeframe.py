"""
Multi-Timeframe Consensus Engine (LOT 30)
Version complète : 1H + 4H + Daily + Weekly
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List

logger = logging.getLogger("MultiTimeframe")

class MultiTimeframeConsensus:
    """
    Full Multi-Timeframe Consensus (LOT 30)
    Supporte 4 timeframes : 1H, 4H, Daily, Weekly
    Exige un consensus fort (minimum 3/4) pour valider le signal.
    """
    
    def __init__(self, timeframes: List[str] = ["1h", "4h", "1d", "1w"]):
        self.timeframes = timeframes
        self.consensus_threshold = 3  # Minimum 3 timeframes d'accord sur 4

    def check_consensus(self, symbol: str, current_price: float, 
                       strategy_signal: float, db) -> Dict:
        """
        Vérifie le consensus multi-timeframe complet.
        Retourne le signal ajusté + score de consensus détaillé.
        """
        agreements = 0
        total_signals = []
        timeframe_results = {}
        
        for tf in self.timeframes:
            try:
                cache_key = f"{symbol}_{tf}"
                df = db.load_candles(cache_key, limit=150)
                
                if df.empty or len(df) < 40:
                    timeframe_results[tf] = "NO_DATA"
                    continue
                
                # Signal directionnel sur ce timeframe
                recent_returns = df['close'].pct_change().dropna().values[-12:]
                ret_mean = np.mean(recent_returns) if len(recent_returns) > 0 else 0.0
                
                tf_signal = 0.0
                if ret_mean > 0.0008:
                    tf_signal = 1.0
                elif ret_mean < -0.0008:
                    tf_signal = -1.0
                
                total_signals.append(tf_signal)
                timeframe_results[tf] = "AGREE" if np.sign(tf_signal) == np.sign(strategy_signal) and abs(tf_signal) > 0.5 else "DISAGREE"
                
                if timeframe_results[tf] == "AGREE":
                    agreements += 1
                    
            except Exception as e:
                logger.warning(f"MTF error on {tf}: {e}")
                timeframe_results[tf] = "ERROR"
                continue
        
        if len(total_signals) == 0:
            return {
                "adjusted_signal": strategy_signal * 0.5,
                "consensus_score": 0.0,
                "timeframes_agreed": 0,
                "total_timeframes": 0,
                "details": timeframe_results
            }
        
        consensus_score = agreements / len(total_signals)
        
        # Ajustement progressif du signal
        if agreements >= self.consensus_threshold:
            adjusted_signal = strategy_signal  # Signal validé
            multiplier = 1.0
        elif agreements == 2:
            adjusted_signal = strategy_signal * 0.65
            multiplier = 0.65
        else:
            adjusted_signal = strategy_signal * 0.30
            multiplier = 0.30
        
        return {
            "adjusted_signal": adjusted_signal,
            "consensus_score": round(consensus_score, 2),
            "timeframes_agreed": agreements,
            "total_timeframes": len(total_signals),
            "multiplier": multiplier,
            "details": timeframe_results
        }