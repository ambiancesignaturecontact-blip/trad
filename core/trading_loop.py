"""
Refactored Trading Loop - LOT 5
Version modulaire et maintenable de live_trading_loop
"""
import asyncio
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict

from core.market_data import MarketDataFetcher
from core.indicators import ExternalIndicators
from core.risk_engine import RiskEngine

# LOT 42 + LOT 43 + LOT 44 + LOT 45
try:
    from models.transformer_forecaster import LiveTransformerEngine
    from rl.transformer_rl import TransformerRLAgent
    from models.gnn_dependency import LiveGNNEngine
    from rl.gnn_rl_agent import GNNRLHybridAgent
    from models.ensemble_controller import AdaptiveEnsembleController
    from rl.ensemble_hybrid_agent import EnsembleHybridAgent
    from models.lifelong_learner import LifelongLearningEngine
    from rl.lifelong_rl_agent import LifelongRLAgent
except ImportError:
    LiveTransformerEngine = None
    TransformerRLAgent = None
    LiveGNNEngine = None
    GNNRLHybridAgent = None
    AdaptiveEnsembleController = None
    EnsembleHybridAgent = None
    LifelongLearningEngine = None
    LifelongRLAgent = None

logger = logging.getLogger("TradingLoop")

class InstitutionalTradingLoop:
    def __init__(self, state: dict, db, risk_manager, meta_engine, 
                 price_predictor, regime_detector, ppo_agent, mlops_trainer,
                 news_analyzer, onchain_tracker, macro_calendar, funding_arb_engine,
                 covariance_engine, microstructure_engine, telegram_bot, 
                 risk_manager_instance, get_ccxt_client, db_save_candles,
                 transformer_engine=None, transformer_rl_agent=None,   # LOT 42
                 gnn_engine=None, gnn_rl_agent=None,                   # LOT 43
                 ensemble_controller=None, ensemble_hybrid_agent=None, # LOT 44
                 lifelong_engine=None, lifelong_rl_agent=None):        # LOT 45
        
        self.state = state
        self.db = db
        self.meta_engine = meta_engine
        self.price_predictor = price_predictor
        self.regime_detector = regime_detector
        self.ppo_agent = ppo_agent
        self.mlops_trainer = mlops_trainer
        self.telegram_bot = telegram_bot
        self.risk_manager = risk_manager_instance
        self.get_ccxt = get_ccxt_client
        
        # LOT 42
        self.transformer_engine = transformer_engine
        self.transformer_rl_agent = transformer_rl_agent
        
        # LOT 43
        self.gnn_engine = gnn_engine
        self.gnn_rl_agent = gnn_rl_agent
        
        # LOT 44
        self.ensemble_controller = ensemble_controller
        self.ensemble_hybrid_agent = ensemble_hybrid_agent
        
        # LOT 45
        self.lifelong_engine = lifelong_engine
        self.lifelong_rl_agent = lifelong_rl_agent
        
        # Modules refactorisés
        self.market_data = MarketDataFetcher(state)
        self.indicators = ExternalIndicators(
            state, news_analyzer, onchain_tracker, macro_calendar, funding_arb_engine
        )
        self.risk_engine = RiskEngine(risk_manager, covariance_engine)
        
        self.loop_count = 0

    async def run(self):
        """Boucle principale refactorisée"""
        logger.info("Institutional Trading Loop (refactored) started")
        
        while True:
            if not self.state["is_running"] or self.state["kill_switch_active"]:
                await asyncio.sleep(1)
                continue

            self.loop_count += 1
            start_time = time.time()
            
            try:
                await self._process_trading_cycle()
            except Exception as e:
                logger.error(f"Trading cycle error: {e}")
            
            # Contrôle de fréquence
            elapsed = time.time() - start_time
            sleep_time = max(2.5 - elapsed, 0.5)
            await asyncio.sleep(sleep_time)

    async def _process_trading_cycle(self):
        """Un cycle complet de trading"""
        # 1. Mise à jour des indicateurs externes
        news_shock = await self.indicators.update_sentiment(self.loop_count)
        await self.indicators.update_onchain(self.loop_count)
        macro_scale = await self.indicators.update_macro()
        
        # 2. Mise à jour des prix
        await self.market_data.update_all_prices()
        
        # 3. Calcul de la matrice de corrélation
        returns_dict = self._get_returns_dict()
        corr_df = self.risk_engine.covariance.calculate_correlation_matrix(returns_dict)
        
        # 4. Boucle sur les actifs
        active_balance_key = "balance_demo" if self.state["mode"] == "DEMO" else "balance_real"
        capital = self.state[active_balance_key]
        
        for symbol in list(self.state["assets"].keys()):
            current_price = self.state["assets"][symbol]["price"]
            if current_price is None:
                continue
                
            await self._process_symbol(symbol, current_price, capital, active_balance_key, 
                                       news_shock, macro_scale, corr_df, returns_dict)

    def _get_returns_dict(self) -> Dict:
        """Récupère les rendements historiques pour le calcul de covariance"""
        returns_dict = {}
        for asset in self.state["assets"]:
            df = self.db.load_candles(asset, limit=30)
            if not df.empty and len(df) >= 5:
                returns_dict[asset] = df['close'].pct_change().dropna().values
            else:
                returns_dict[asset] = np.zeros(30)
        return returns_dict

    async def _process_symbol(self, symbol, current_price, capital, active_balance_key,
                              news_shock, macro_scale, corr_df, returns_dict):
        """Traite un symbole individuel"""
        # Récupération des données historiques
        df = self.db.load_candles(symbol, limit=120)
        if df.empty or len(df) < 10:
            df = self.state.get("historical_bars")
            if df is None:
                return

        # Mise à jour du DataFrame avec le dernier prix
        vol_val = self.state.get("last_tick_volume", 15.0)
        new_row = pd.DataFrame([{
            "open": current_price * 0.9995,
            "high": current_price * 1.0005,
            "low": current_price * 0.9990,
            "close": current_price,
            "volume": vol_val
        }], index=[pd.Timestamp.now()])
        
        df = pd.concat([df.iloc[1:], new_row])
        self.db.save_candles(symbol, new_row)

        # === Prédictions IA ===
        recent_returns = df['close'].pct_change().dropna().values[-10:]
        ret_mean = np.mean(recent_returns) if len(recent_returns) > 0 else 0.0
        vol_mean = np.std(recent_returns) if len(recent_returns) > 0 else 0.01
        
        regime_id = int(self.regime_detector.predict(np.array([[ret_mean, vol_mean]]))[0])
        self.state["regime_id"] = regime_id
        
        seq = df[['close', 'volume', 'high', 'low', 'open']].pct_change().fillna(0).values[-5:]
        ml_pred = float(self.price_predictor.predict(seq))
        self.state["ml_prediction_pct"] = ml_pred

        # Position actuelle
        positions = self.db.get_positions()
        asset_pos = next((p for p in positions if p['symbol'] == symbol), None)
        pos_qty = asset_pos['qty'] if asset_pos else 0.0

        # === Génération du signal ===
        market_data = {
            'df': df,
            'price_primary': current_price,
            'price_secondary': current_price,
            'bids': self.state["order_book"].get("bids"),
            'asks': self.state["order_book"].get("asks"),
            'inventory': pos_qty,
            'max_inventory': capital / current_price if capital > 0 else 0.0
        }
        
        consensus = self.meta_engine.allocate(market_data, regime_id, ml_pred, 0.0)
        final_signal = consensus["final_signal"]
        final_signal = (0.80 * final_signal) + (0.20 * self.state.get("sentiment_index", 0.0))
        
        # === LOT 42: Transformer + Hybrid RL Decision ===
        try:
            if hasattr(self, 'transformer_engine') and self.transformer_engine is not None:
                transformer_pred = self.transformer_engine.predict(df)
                if hasattr(self, 'transformer_rl_agent') and self.transformer_rl_agent is not None:
                    hybrid_decision = self.transformer_rl_agent.decide(
                        symbol=symbol,
                        market_data=market_data,
                        transformer_pred=transformer_pred,
                        regime_id=regime_id
                    )
                    # Blend with existing signal
                    final_signal = 0.65 * final_signal + 0.35 * hybrid_decision["final_signal"]
                    if self.loop_count % 12 == 0:
                        logger.info(f"LOT 42 [{symbol}] | Transformer signal: {hybrid_decision['explanation']}")
        except Exception as e:
            pass  # Fail-safe: keep original signal if LOT 42 fails

        # === LOT 43: Dynamic GNN Cross-Asset Dependency Adjustment ===
        try:
            if hasattr(self, 'gnn_engine') and self.gnn_engine is not None:
                # Build returns dict for GNN
                returns_dict = {}
                for asset in self.state["assets"]:
                    df_asset = self.db.load_candles(asset, limit=40)
                    if not df_asset.empty:
                        returns_dict[asset] = df_asset['close'].pct_change().dropna().values[-30:]
                
                prices = {a: self.state["assets"][a]["price"] for a in self.state["assets"]}
                gnn_output = self.gnn_engine.update_graph_and_predict(returns_dict, prices)
                
                if hasattr(self, 'gnn_rl_agent') and self.gnn_rl_agent is not None:
                    gnn_decision = self.gnn_rl_agent.decide(
                        symbol=symbol,
                        market_data=market_data,
                        gnn_output=gnn_output,
                        base_signal=final_signal,
                        regime_id=regime_id
                    )
                    final_signal = gnn_decision["final_signal"]
                    if self.loop_count % 15 == 0:
                        logger.info(f"LOT 43 [{symbol}] | GNN signal: {gnn_decision['explanation']}")
        except Exception as e:
            pass  # Fail-safe

        # === LOT 44: Adaptive Ensemble Final Blending ===
        try:
            if hasattr(self, 'ensemble_controller') and self.ensemble_controller is not None:
                model_signals = {
                    "transformer": self.state.get("ml_prediction_pct", 0.0) * 2,
                    "gnn": final_signal * 0.9,
                    "meta_labeling": getattr(meta_labeling_engine, 'last_quality', 0.6) if 'meta_labeling_engine' in globals() else 0.5,
                    "multi_agent_rl": 0.0,
                    "bayesian_risk": 0.0,
                    "causal_filter": 0.0,
                    "regime_switcher": 0.0
                }
                
                # Try to get RL signal
                if hasattr(self, 'multi_agent_rl') and self.multi_agent_rl:
                    try:
                        model_signals["multi_agent_rl"] = self.multi_agent_rl.get_action(symbol, market_data)
                    except:
                        pass
                
                if hasattr(self, 'ensemble_hybrid_agent') and self.ensemble_hybrid_agent is not None:
                    ensemble_decision = self.ensemble_hybrid_agent.decide(
                        symbol=symbol,
                        market_data=market_data,
                        model_signals=model_signals,
                        model_confidences={"transformer": 0.82, "gnn": 0.78, "meta_labeling": 0.75},
                        regime_id=regime_id
                    )
                    final_signal = ensemble_decision["final_signal"]
                    
                    if self.loop_count % 10 == 0:
                        logger.info(f"LOT 44 [{symbol}] | Ensemble: {ensemble_decision['explanation']['final_signal']}")
        except Exception as e:
            pass  # Fail-safe
        
        final_signal = max(-1.0, min(1.0, final_signal))

        # === Risk Sizing ===
        atr = df['high'].values[-1] - df['low'].values[-1] or current_price * 0.008
        target_qty = self.risk_manager.calculate_position_size(capital, atr, current_price)
        
        # Application des facteurs de risque
        news_scale = 0.20 if news_shock else 1.0
        onchain_risk = self.state.get("onchain_risk_score", 0.5)
        corr_reduction = 1.0
        
        if corr_df is not None and not corr_df.empty:
            corr_reduction = self.risk_engine.covariance.evaluate_portfolio_concentration_risk(
                symbol, positions, corr_df
            )
        
        target_qty = self.risk_engine.apply_risk_scaling(
            target_qty, news_scale, macro_scale, onchain_risk, corr_reduction
        )
        
        target_qty *= abs(final_signal)
        target_direction = np.sign(final_signal) if abs(final_signal) > 0.08 else 0.0
        desired_qty = target_direction * target_qty
        trade_qty = desired_qty - pos_qty

        # === Exécution ===
        trade_notional = abs(trade_qty) * current_price
        min_notional = 3.0 if capital < 200 else 5.0 if capital < 1000 else 10.0
        
        if trade_notional > min_notional:
            logger.info(f"TRADE SIGNAL [{symbol}] | Signal: {final_signal:.3f} | Notional: ${trade_notional:.2f}")
            # L'exécution réelle reste dans main.py pour l'instant (complexité des ordres)
            # TODO: Déplacer l'exécution dans core/execution.py dans un prochain lot

        # Mise à jour de l'equity (simplifié)
        net_equity = capital + sum(
            p['qty'] * self.state["assets"].get(p['symbol'], {}).get("price", current_price) 
            for p in positions
        )
        self.state["current_equity"] = net_equity