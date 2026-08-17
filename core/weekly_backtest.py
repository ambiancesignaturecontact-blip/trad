"""
Weekly Backtest + Telegram Report (LOT 16)
Exécute automatiquement un backtest complet toutes les semaines
sur des données réelles et envoie un rapport Telegram.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta

from backtester.enhanced_engine import EnhancedEventDrivenBacktester

logger = logging.getLogger("WeeklyBacktest")

class WeeklyBacktestReporter:
    def __init__(self, db, meta_engine, risk_manager, regime_detector, 
                 price_predictor, ppo_agent, telegram_bot):
        self.db = db
        self.meta_engine = meta_engine
        self.risk_manager = risk_manager
        self.regime_detector = regime_detector
        self.price_predictor = price_predictor
        self.ppo_agent = ppo_agent
        self.telegram = telegram_bot

    async def run_weekly_backtest(self):
        """Exécute le backtest hebdomadaire sur données réelles"""
        try:
            logger.info("Starting Weekly Backtest (real data)...")
            
            # Récupération des données réelles (dernières 2000 barres)
            df = self.db.load_candles("BTCUSDT", limit=2000)
            
            if df.empty or len(df) < 800:
                logger.warning("Not enough real data for weekly backtest")
                return
            
            # Split : 70% train / 30% test (out-of-sample)
            split = int(len(df) * 0.7)
            train_df = df.iloc[:split]
            test_df = df.iloc[split:]
            
            backtester = EnhancedEventDrivenBacktester(
                initial_capital=50000,
                commission_maker=0.0002,
                commission_taker=0.0005,
                slippage_bps=1.8
            )
            
            # Backtest sur la période de test (out-of-sample)
            metrics = backtester.run(
                test_df, 
                self.meta_engine, 
                self.risk_manager,
                self.regime_detector, 
                self.price_predictor, 
                self.ppo_agent,
                symbol="BTCUSDT",
                is_perpetual=True
            )
            
            # Rapport Telegram
            report = self._format_report(metrics, len(test_df))
            
            if self.telegram:
                await self.telegram.send_push_notification(report)
            
            logger.info(f"Weekly Backtest completed. Sharpe: {metrics.get('sharpe_ratio', 0):.3f}")
            
        except Exception as e:
            logger.error(f"Weekly backtest failed: {e}")

    def _format_report(self, metrics: dict, bars: int) -> str:
        """Formate un beau rapport Telegram"""
        sharpe = metrics.get('sharpe_ratio', 0)
        winrate = metrics.get('win_rate_pct', 0)
        dd = metrics.get('max_drawdown_pct', 0)
        ret = metrics.get('total_return_pct', 0)
        trades = metrics.get('total_trades', 0)
        
        # Interprétation
        if sharpe > 1.8:
            quality = "🟢 EXCELLENT"
        elif sharpe > 1.2:
            quality = "🟡 BON"
        else:
            quality = "🔴 À AMÉLIORER"
        
        report = (
            f"📊 *RAPPORT BACKTEST HEBDOMADAIRE*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 Période testée : {bars} barres (≈ 12 semaines)\n\n"
            f"📈 *Performance*\n"
            f"• Rendement : *{ret:.2f}%*\n"
            f"• Sharpe Ratio : *{sharpe:.3f}*\n"
            f"• Win Rate : *{winrate:.1f}%*\n"
            f"• Max Drawdown : *{dd:.2f}%*\n"
            f"• Nombre de trades : *{trades}*\n\n"
            f"🏆 *Qualité globale* : {quality}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"_Backtest exécuté sur données réelles uniquement._"
        )
        return report

    async def start_scheduler(self):
        """Lance le scheduler hebdomadaire (tous les lundis à 08:00)"""
        while True:
            now = datetime.now()
            # Prochain lundi à 08:00
            days_ahead = 0 - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(hour=8, minute=0, second=0, microsecond=0)
            
            wait_seconds = (next_run - now).total_seconds()
            logger.info(f"Next weekly backtest scheduled in {wait_seconds/3600:.1f} hours")
            
            await asyncio.sleep(wait_seconds)
            await self.run_weekly_backtest()