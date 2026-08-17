"""
Advanced Alerting Engine (LOT 17)
Envoie des alertes proactives et intelligentes via Telegram.
"""
import asyncio
import logging
import time
from typing import Dict

logger = logging.getLogger("AlertingEngine")

class AdvancedAlertingEngine:
    def __init__(self, telegram_bot, state):
        self.telegram = telegram_bot
        self.state = state
        self.last_alerts = {}  # Pour éviter le spam

    async def send_alert(self, alert_type: str, message: str, cooldown_minutes: int = 60):
        """Envoie une alerte avec cooldown anti-spam"""
        now = time.time()
        last_sent = self.last_alerts.get(alert_type, 0)
        
        if now - last_sent < cooldown_minutes * 60:
            return  # Trop tôt, on skip
        
        if self.telegram:
            try:
                await self.telegram.send_push_notification(
                    f"🔔 *ALERTE INSTITUTIONNELLE*\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{message}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )
                self.last_alerts[alert_type] = now
                logger.info(f"Alert sent: {alert_type}")
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")

    async def check_risk_adjustment(self, adjustment: dict):
        """Alerte quand le Dynamic Risk Scaling modifie les paramètres"""
        if adjustment.get("adjusted"):
            factor = adjustment.get("adjustment_factor", 1.0)
            reasons = ", ".join(adjustment.get("reasons", []))
            
            msg = (
                f"⚖️ *AJUSTEMENT DE RISQUE*\n"
                f"Facteur appliqué : *{factor:.2f}x*\n"
                f"Raisons : `{reasons}`\n"
                f"Exposition max mise à jour."
            )
            await self.send_alert("risk_adjustment", msg, cooldown_minutes=120)

    async def check_rejection_rate(self, stats: dict):
        """Alerte si le taux de rejet des ordres est trop élevé"""
        rejected = stats.get("rejected", 0)
        total = stats.get("total_attempts", 1)
        rate = (rejected / total) * 100 if total > 0 else 0
        
        if rate > 25:
            msg = (
                f"⚠️ *TAUX DE REJET ÉLEVÉ*\n"
                f"Taux actuel : *{rate:.1f}%*\n"
                f"Ordres rejetés : {rejected}/{total}\n"
                f"Vérifiez les soldes et les tailles d'ordre."
            )
            await self.send_alert("high_rejection", msg, cooldown_minutes=90)

    async def check_strategy_performance(self, recent_performance: dict):
        """Alerte si une stratégie est en sous-performance prolongée"""
        for name, scores in recent_performance.items():
            if len(scores) >= 30:
                avg_score = sum(scores[-30:]) / 30
                if avg_score < -0.3:
                    msg = (
                        f"📉 *STRATÉGIE EN SOUS-PERFORMANCE*\n"
                        f"`{name}`\n"
                        f"Score moyen (30 trades) : *{avg_score:.3f}*\n"
                        f"Action : Walk-Forward a réduit son poids."
                    )
                    await self.send_alert(f"strategy_{name}", msg, cooldown_minutes=240)

    async def check_drawdown(self, current_equity: float, initial_capital: float):
        """Alerte drawdown"""
        if initial_capital <= 0:
            return
        dd = ((initial_capital - current_equity) / initial_capital) * 100
        if dd > 4.0:
            msg = (
                f"🚨 *DRAWDOWN ÉLEVÉ*\n"
                f"Drawdown actuel : *{dd:.2f}%*\n"
                f"Capital : ${current_equity:,.2f}\n"
                f"Surveillez les positions."
            )
            await self.send_alert("drawdown", msg, cooldown_minutes=180)