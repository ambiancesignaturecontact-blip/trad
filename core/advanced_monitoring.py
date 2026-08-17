"""
LOT 60: Advanced Monitoring & Auto-Scaling System
Real-time performance monitoring + automatic risk adjustment + anomaly detection
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger("AdvancedMonitoring")

class AdvancedMonitoringSystem:
    """
    LOT 60: Advanced monitoring with auto-scaling capabilities.
    """

    def __init__(self, 
                 max_drawdown_threshold: float = 0.08,
                 pnl_drop_threshold: float = 0.15,
                 model_performance_threshold: float = 0.05):
        
        self.max_drawdown_threshold = max_drawdown_threshold
        self.pnl_drop_threshold = pnl_drop_threshold
        self.model_performance_threshold = model_performance_threshold

        # Historical data
        self.equity_history = deque(maxlen=500)
        self.drawdown_history = deque(maxlen=200)
        self.model_performance = {}  # model_name -> deque of scores

        self.alerts: List[Dict] = []
        self.auto_scaling_actions: List[Dict] = []

    def update_equity(self, current_equity: float, timestamp: Optional[datetime] = None):
        if timestamp is None:
            timestamp = datetime.now()

        self.equity_history.append({
            "timestamp": timestamp,
            "equity": current_equity
        })

        # Calculate drawdown
        if len(self.equity_history) > 1:
            peak = max([e["equity"] for e in self.equity_history])
            drawdown = (current_equity - peak) / peak if peak > 0 else 0
            self.drawdown_history.append(drawdown)

            if drawdown < -self.max_drawdown_threshold:
                self._trigger_alert("HIGH_DRAWDOWN", f"Drawdown reached {drawdown*100:.1f}%")

    def update_model_performance(self, model_name: str, score: float):
        if model_name not in self.model_performance:
            self.model_performance[model_name] = deque(maxlen=50)
        self.model_performance[model_name].append(score)

        # Check for underperformance
        if len(self.model_performance[model_name]) >= 10:
            recent_avg = np.mean(list(self.model_performance[model_name])[-10:])
            if recent_avg < self.model_performance_threshold:
                self._trigger_alert("MODEL_UNDERPERFORMING", f"{model_name} avg score: {recent_avg:.3f}")

    def check_anomalies(self, current_pnl: float, current_drawdown: float) -> List[str]:
        """Detect performance anomalies"""
        anomalies = []

        if current_drawdown < -self.max_drawdown_threshold:
            anomalies.append("Excessive drawdown")

        # Check for sudden PnL drop
        if len(self.equity_history) > 5:
            recent_pnl = [e["equity"] for e in list(self.equity_history)[-5:]]
            if len(recent_pnl) >= 2:
                pnl_change = (recent_pnl[-1] - recent_pnl[0]) / recent_pnl[0]
                if pnl_change < -self.pnl_drop_threshold:
                    anomalies.append("Sudden PnL drop detected")

        return anomalies

    def auto_scale_exposure(self, current_exposure: float, anomalies: List[str]) -> float:
        """Automatically adjust capital exposure based on risk"""
        new_exposure = current_exposure

        if "Excessive drawdown" in anomalies:
            new_exposure = max(0.25, current_exposure * 0.6)
            self._log_scaling_action("REDUCE_EXPOSURE", current_exposure, new_exposure, "High drawdown")

        elif "Sudden PnL drop detected" in anomalies:
            new_exposure = max(0.30, current_exposure * 0.75)
            self._log_scaling_action("REDUCE_EXPOSURE", current_exposure, new_exposure, "PnL drop")

        # Gradual recovery
        elif len(anomalies) == 0 and current_exposure < 0.65:
            new_exposure = min(0.85, current_exposure * 1.1)
            self._log_scaling_action("INCREASE_EXPOSURE", current_exposure, new_exposure, "Conditions improved")

        return round(new_exposure, 3)

    def _trigger_alert(self, alert_type: str, message: str):
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message
        }
        self.alerts.append(alert)
        logger.warning(f"[MONITORING] {alert_type}: {message}")

    def _log_scaling_action(self, action: str, old_value: float, new_value: float, reason: str):
        log = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason
        }
        self.auto_scaling_actions.append(log)
        logger.info(f"[AUTO-SCALING] {action}: {old_value:.2f} → {new_value:.2f} ({reason})")

    def get_status(self) -> Dict:
        return {
            "current_drawdown": list(self.drawdown_history)[-1] if self.drawdown_history else 0,
            "active_alerts": len(self.alerts),
            "auto_scaling_actions": len(self.auto_scaling_actions),
            "monitored_models": list(self.model_performance.keys())
        }
