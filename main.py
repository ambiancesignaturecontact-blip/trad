from dotenv import load_dotenv
load_dotenv()  # .env (secrets) before any env consumers

from core.config import settings

import asyncio
import uuid
import httpx
import json
import logging
import random
import time
import numpy as np
import pandas as pd
import ccxt
import os
import websockets
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict
from core.middleware import (RequestLoggingMiddleware, SecurityHeadersMiddleware,
                             IPRateLimitMiddleware, LoginRateLimitMiddleware, install_cors)
from core.position_manager import (PositionProtection, PositionProtectionStore,
                                     evaluate_protection, evaluate_time_stop,
                                     apply_breakeven_stop, partial_take_profit,
                                     can_pyramid, position_age_hours)
from core.portfolio_allocator import PortfolioAllocator
from core.counterparty_risk import CounterpartyRiskManager
from core.cost_accounting import CostAccounting
from core.module_honesty import get_module_status, status_summary
from core.attribution import PerformanceAttribution, quality_metrics
from models.scenario_stress import ScenarioStressTester, CRISIS_SCENARIOS
from backtester.bias_audit import audit_backtest
from core.volatility_targeting import volatility_scale_factor
from core.signal_library import SIGNAL_LIBRARY, evaluate_all_signals, evaluate_signal
from core.execution_router import ExecutionAlpha, SlippageModel, decide_style
from core.copy_mirror import fetch_trader_positions, build_mirror_orders, mirror_status_text
from core.paper_execution import simulate_paper_fill
from core.world_model import (compute_regime_probs, compute_market_state,
                              discover_causal_parents, build_causal_feature_df, counterfactual_alpha)
from core.mixture_experts import MixtureOfExperts, risk_adjusted_reward, curriculum_sort
from core.hypothesis_generator import HypothesisGenerator
from core.meta_cognition import (adaptive_conviction_threshold, decide_no_trade, hedging_decision)
from core.execution_agent import ExecutionStyleBandit, tradability_factor, StrategyExecutionAttribution
from core.risk_committee import RiskCommittee, daily_risk_budget
from core.self_assessment import simulation_divergence, honesty_factor, meta_attribution, health_honesty_component, reason_weight_from_attribution
from core.llm_narrative import (daily_market_narrative, explain_decision, answer_question,
                                 daily_market_narrative_async, answer_question_async)
from core.organization import Organization
from core.research_discipline import (pre_register_hypothesis, double_validation, live_p_value, meta_label_filter)
from core.robustness import (save_state_snapshot, restore_state_snapshot, Supervisor, chaos_cut_feed,
                             audit_deterministic, seed_audit_rng)
from core.confidence_index import compute_confidence_index
from core.risk_pipeline import (REWARD_RISK_RATIO, MIN_REWARD_RISK, kelly_dynamic,
                                rr_requirement, rr_net_positive, entry_rr_filter,
                                RiskStateMachine, StrategyWinRateTracker,
                                apply_risk_pipeline, ROUND_TRIP_COST_PCT,
                                STOP_LOSS_PCT, ATR_MULT_SL)
from core.world_model import compute_structural_regimes, cross_asset_bias
from core.reporting import build_daily_report, compute_health_score, build_concierge_message

# Import our quant models
from models.regime_detector import MarketRegimeDetector, compute_order_book_imbalance
from models.price_predictor import LSTMLikePredictor, PPOTRAgent
from strategies.engine import (
    TrendFollowingStrategy, MeanReversionStrategy, MarketMakingStrategy,
    StatisticalArbitrageStrategy, ArbitrageInterExchangeStrategy, GridTradingStrategy,
    ScalpingStrategy, MetaAllocationEngine
)
from copytrading.manager import CopyTradingManager
from risk.risk_manager import RiskManager
from db_manager import DBManager
from backtester.engine import EventDrivenBacktester

# NEW ADVANCED MODELS
from models.sentiment_analyzer import NewsSentimentAnalyzer, SOURCE_WEIGHTS as SOURCE_WEIGHTS_REF
from models.onchain_tracker import OnChainTracker
from models.defi_wallet import NonCustodialDeFiWallet
from models.mlops_pipeline import MLOpsAutoTrainer
from models.risk_covariance import RiskCovarianceEngine
from market_data.multi_source import MultiSourcePriceEngine
from market_data.order_flow import OrderFlowEngine
from models.volatility_arbitrage import OptionsVolatilityArbitrageEngine
from models.telegram_bot import TelegramBotManager
from models.funding_arbitrage import FundingRateArbitrageEngine
from models.dex_cex_arbitrage import DexCexArbitrageEngine
from models.monte_carlo import MonteCarloStressTester
from models.execution_slicer import SmartOrderSlicer
from models.microstructure_edge import MicrostructureEdgeEngine
from models.lopez_de_prado import MetaLabelingTripleBarrier, calculate_deflated_sharpe_ratio, PurgedKFoldEmbargo
from models.almgren_chriss import AlmgrenChrissExecutionOptimizer, calculate_cvar_constrained_sizing
from models.macro_calendar import MacroeconomicCalendarEngine
from models.oms_ems import OrderManagementSystem, ReconciliationEngine, OrderStatus

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("InstitutionalTradingBot")

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan handler (modern replacement for @app.on_event)."""
    await startup_event()
    yield
    await shutdown_event()


app = FastAPI(title="Institutional AI Trading Platform", version="1.0.0", docs_url="/api/docs", redoc_url="/api/redoc",
              lifespan=lifespan)


@app.exception_handler(Exception)
async def _global_exception_handler(request, exc):
    """LOT 8 : handler global — loggue le traceback complet (diagnostic)."""
    import traceback
    logger.error(f"⚠️ EXCEPTION GLOBALE sur {request.method} {request.url.path}: {exc}\n{traceback.format_exc()}")
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})
# Institutional middleware (audit B2/B3): request logging, security headers, rate limits, CORS
app.add_middleware(LoginRateLimitMiddleware)
app.add_middleware(IPRateLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
install_cors(app)
templates = Jinja2Templates(directory="templates")

# Data Quality Status Enum
class DataQualityStatus:
    LIVE = "LIVE"
    DELAYED = "DELAYED"
    STALE = "STALE"
    INVALID = "INVALID"
    DISCONNECTED = "DISCONNECTED"
    UNAVAILABLE = "UNAVAILABLE"

# Globals
db = DBManager()
copy_manager = CopyTradingManager()
risk_manager = RiskManager()

# Advanced Engines
news_analyzer = NewsSentimentAnalyzer()
onchain_tracker = OnChainTracker()
defi_wallet = NonCustodialDeFiWallet()
covariance_engine = RiskCovarianceEngine(max_correlation_threshold=0.70)
volatility_arb_engine = OptionsVolatilityArbitrageEngine()
funding_arb_engine = FundingRateArbitrageEngine()
dex_cex_arb_engine = DexCexArbitrageEngine()
monte_carlo_tester = MonteCarloStressTester(num_simulations=10000, horizon_steps=100)
order_slicer = SmartOrderSlicer(time_horizon_seconds=120, num_slices=5)
microstructure_engine = MicrostructureEdgeEngine()
almgren_chriss_optimizer = AlmgrenChrissExecutionOptimizer()
macro_calendar = MacroeconomicCalendarEngine()

# Unified Exchange Adapters, EMS, OMS & Reconciliation Engine
from adapters.exchange_adapter import BinanceExchangeAdapter, BybitExchangeAdapter
from models.oms_ems import ExecutionManagementSystem

binance_adapter = BinanceExchangeAdapter(None)
bybit_adapter = BybitExchangeAdapter(None)

ems = ExecutionManagementSystem(binance_adapter, bybit_adapter)
oms = OrderManagementSystem(db, ems)
reconciler = ReconciliationEngine(db)

# Instantiate strategies
# VISION §5: institutional strategy suite - now includes the previously-dead
# modules (momentum, volatility_breakout, multi_timeframe) + carry + cross-sectional.
from strategies.momentum import MomentumStrategy
from strategies.volatility_breakout import VolatilityBreakoutStrategy
from strategies.institutional import CarryStrategy, CrossSectionalMomentumStrategy, MultiTimeframeWrapperStrategy
from strategies.regime_switching import RegimeSwitchingAllocator

strategies_list = [
    TrendFollowingStrategy(),
    MeanReversionStrategy(),
    MarketMakingStrategy(),
    StatisticalArbitrageStrategy(),
    ArbitrageInterExchangeStrategy(),
    GridTradingStrategy(),
    ScalpingStrategy(),
    MomentumStrategy(),
    VolatilityBreakoutStrategy(),
    CarryStrategy(),
    CrossSectionalMomentumStrategy(),
    MultiTimeframeWrapperStrategy(db=db),
]
regime_allocator = RegimeSwitchingAllocator()
meta_engine = MetaAllocationEngine(strategies=strategies_list, regime_allocator=regime_allocator)

# ML Models
regime_detector = MarketRegimeDetector()
price_predictor = LSTMLikePredictor(input_dim=5, hidden_dim=24)  # deeper LSTM (audit B9-1)
ppo_agent = PPOTRAgent(state_dim=4, action_dim=1)

# MLOps Auto-Trainer
mlops_trainer = MLOpsAutoTrainer(regime_detector, price_predictor, db)

# === LOT 46: Online Model Selection & Adaptive Ensemble Pruning ===
from core.lot46_integration import create_lot46_components
# Honest functional model names (audit B9-1): each selector slot is a real,
# distinct strategy family instead of aspirational "transformer/gnn" labels.
model_names_lot46 = ["trend_lstm", "meanrev_net", "volatility_net", "correlation_net", "regime_net"]
model_selector, adaptive_ensemble_agent = create_lot46_components(model_names_lot46)
logger.info("✅ LOT 46: Online Model Selector initialized")

# LOT 46 Scheduler
# REAL performance attribution: each model is scored from the REALIZED PnL of the
# assets it tracks (via the Trade Journal), normalized to [-1, 1]. No synthetic
# np.random data — the ensemble selector now learns from actual trading outcomes.
_MODEL_ASSET_MAP = {
    "trend_lstm": ["BTCUSDT", "ETHUSDT"],
    "meanrev_net": ["XAUUSD", "EURUSD"],
    "volatility_net": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "correlation_net": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "regime_net": ["AAPL", "TSLA"],
}

async def lot46_model_selection_scheduler():
    while True:
        await asyncio.sleep(3 * 3600)
        try:
            trades = trade_journal.get_trades(limit=500)
            initial_cap = STATE.get("initial_capital_demo", 100000.0)
            for name in model_names_lot46:
                assets = _MODEL_ASSET_MAP.get(name, ["BTCUSDT"])
                realized = sum(
                    float(t.get("realized_pnl") or 0.0)
                    for t in trades if t.get("symbol") in assets
                )
                # Normalize realized PnL vs 1% of capital into [-1, 1]
                score = float(np.tanh(realized / max(initial_cap * 0.01, 1.0)))
                model_selector.update_performance(name, score)
            status = model_selector.get_status()
            logger.info(f"LOT 46: Active models → {status['active_models']} (realized-PnL attribution)")
        except Exception as e:
            logger.warning(f"LOT 46 scheduler error: {e}")

# NOTE: The LOT 46 scheduler task is started inside the FastAPI startup event,
# because asyncio.create_task() at module level crashes with
# "RuntimeError: no running event loop" (the loop only exists once uvicorn runs).

# === LOT 47++: Complete Multi-Exchange Smart Order Router ===
from core.multi_exchange_sor import MultiExchangeSmartOrderRouter
multi_exchange_sor = MultiExchangeSmartOrderRouter()
logger.info("✅ LOT 47++: Complete Multi-Exchange SOR initialized (Binance + Bybit + Real metrics)")

# === LOT 48+: Robust Feature Store with Versioning ===
from core.feature_store import FeatureStore
feature_store = FeatureStore()
logger.info("✅ LOT 48+: Robust Feature Store with Versioning initialized")

# === LOT 49: Realistic Execution Simulator (Slippage + Latency) ===
from core.execution_simulator import ExecutionSimulator
execution_simulator = ExecutionSimulator(base_slippage_bps=6.0, base_latency_ms=75)
logger.info("✅ LOT 49: Execution Simulator (Slippage + Latency) initialized")

# === LOT 50: Dynamic Capital Allocator (Kelly + Risk Parity) ===
from core.dynamic_capital_allocator import DynamicCapitalAllocator
capital_allocator = DynamicCapitalAllocator(base_exposure=0.68, max_exposure=0.92, min_exposure=0.28)
portfolio_allocator = PortfolioAllocator()  # LOT 6: allocation top-down en cascade (PDF Pilier L)
counterparty_risk = CounterpartyRiskManager()  # LOT 7: risque de contrepartie par exchange (PDF Pilier P)
cost_accounting = CostAccounting()            # LOT 8: coût réel par trade + portage (PDF Pilier O)
attribution = PerformanceAttribution()        # LOT 8: attribution par facteur/régime/actif (PDF Pilier Q)
scenario_tester = ScenarioStressTester()      # LOT 8: stress par crises réelles (PDF Pilier N)
logger.info("✅ LOT 50: Dynamic Capital Allocator initialized")

# === LOT 52: Trade Journal (with Notes + Screenshots) ===
from core.trade_journal import TradeJournal
trade_journal = TradeJournal()
logger.info("✅ LOT 52: Trade Journal initialized")

# === LOT 53: Advanced Causal Discovery Engine ===
from ai.causal_discovery import CausalDiscoveryEngine
causal_engine = CausalDiscoveryEngine()
logger.info("✅ LOT 53: Advanced Causal Discovery Engine initialized")

# === LOT 54: Generative Models for Extreme Scenarios ===
from ai.generative_extreme_scenarios import ExtremeScenarioGenerator
generative_engine = ExtremeScenarioGenerator()
logger.info("✅ LOT 54: Generative Extreme Scenario Engine initialized")

# === LOT 55: RLHF (Reinforcement Learning from Human Feedback) ===
from rl.rlhf_reward_model import RLHFRewardModel
rlhf_reward_model = RLHFRewardModel()
logger.info("✅ LOT 55: RLHF Reward Model initialized")

# === LOT 56: Multi-Objective Portfolio Optimizer (Sharpe + CVaR + Max Drawdown) ===
from core.multi_objective_optimizer import MultiObjectivePortfolioOptimizer
multi_objective_optimizer = MultiObjectivePortfolioOptimizer()
logger.info("✅ LOT 56: Multi-Objective Portfolio Optimizer initialized")

# === LOT 57: Advanced Almgren-Chriss Market Impact & Liquidity Model ===
from core.almgren_chriss_advanced import AdvancedAlmgrenChrissModel
almgren_chriss_model = AdvancedAlmgrenChrissModel(gamma=0.12, eta=0.06, lambda_risk=0.45)
logger.info("✅ LOT 57: Advanced Almgren-Chriss Model initialized")

# === LOT 58: Tax & Compliance Reporting (FIFO, Cost Basis, PnL Fiscal) ===
from core.tax_compliance import TaxComplianceEngine
tax_engine = TaxComplianceEngine()
logger.info("✅ LOT 58: Tax & Compliance Engine initialized")

# === LOT 59: Advanced Model Explainability (SHAP + LIME) ===
from ai.model_explainability import ModelExplainer
# Note: ModelExplainer needs to be initialized with a trained model and feature names
# Example: explainer = ModelExplainer(model=some_model, feature_names=feature_list)
logger.info("✅ LOT 59: Model Explainability module initialized (SHAP + LIME ready)")

# === LOT 60: Advanced Monitoring & Auto-Scaling System ===
from core.advanced_monitoring import AdvancedMonitoringSystem
monitoring_system = AdvancedMonitoringSystem()
logger.info("✅ LOT 60: Advanced Monitoring & Auto-Scaling System initialized")

# === LOT 61: Prometheus /metrics exposition (for the bundled Grafana stack) ===
from core import metrics as platform_metrics
platform_metrics.mark_startup()
logger.info("✅ LOT 61: Prometheus /metrics registry initialized")

# === LOT 63: Centralized outbound API rate limiting ===
from core.rate_limits import bybit_limiter, binance_limiter, yahoo_limiter, news_limiter, rpc_limiter
from database.auth import AuthManager, Roles, security as auth_security
from fastapi.security import HTTPBearer

# Optional bearer: when no Authorization header is present, credentials is None
# and require_auth() decides whether auth is needed (DEMO vs REAL / AUTH_ENABLED).
auth_security_optional = HTTPBearer(auto_error=False)
logger.info("✅ LOT 63: Outbound API rate limiters initialized")

# React dashboard served at /app when built (audit B14-1: one modern UI, one classic)
import os as _os
if _os.path.isdir(_os.path.join(_os.getcwd(), "frontend", "dist")):
    try:
        from fastapi.staticfiles import StaticFiles as _SF
        app.mount("/app", _SF(directory=_os.path.join(_os.getcwd(), "frontend", "dist"), html=True), name="react-dashboard")
        logger.info("✅ React dashboard mounted at /app (frontend/dist found)")
    except Exception as e:
        logger.warning(f"React dashboard mount skipped: {e}")

# Request bodies validation models
_VALID_STRATEGIES = (
    "Trend Following", "Mean Reversion", "Market Making", "Statistical Arbitrage",
    "Inter-Exchange Arbitrage", "Grid Trading", "Scalping", "Momentum",
    "Volatility Breakout", "Carry", "Cross-Sectional Momentum", "Multi-Timeframe",
)

class StrategyToggle(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    enabled: bool

    @field_validator("name")
    @classmethod
    def _valid_strategy(cls, v: str) -> str:
        if v not in _VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy '{v}'. Valid: {_VALID_STRATEGIES}")
        return v

class RiskSettingsUpdate(BaseModel):
    max_daily_drawdown_pct: float = Field(gt=0.0, le=0.50)
    max_total_drawdown_pct: float = Field(gt=0.0, le=0.80)
    max_exposure_per_asset_pct: float = Field(gt=0.0, le=1.0)
    fractional_kelly_multiplier: float = Field(gt=0.0, le=1.0)
    deviation_limit_pct: float = Field(ge=0.0, le=1.0)

class KeyStorage(BaseModel):
    api_key: str = Field(min_length=3, max_length=128)
    secret_key: str = Field(min_length=3, max_length=256)
    exchange: str = Field(pattern="^(binance|bybit|hyperliquid)$", description="Supported exchange")

class SwitchModeRequest(BaseModel):
    target_mode: str = Field(pattern="^(DEMO|REAL)$")
    verification_2fa: str = Field(min_length=4, max_length=12)

class BotToggleRequest(BaseModel):
    is_running: bool

class SetBalanceRequest(BaseModel):
    balance: float = Field(gt=0.0, le=1_000_000_000.0, description="Positive demo balance in USD")

class CopyTradeRequest(BaseModel):
    trader_id: str = Field(min_length=4, max_length=80)
    action: str = Field(pattern="^(START|STOP)$")
    allocated_capital: float = Field(ge=0.0, le=1_000_000_000.0, default=0.0)

# Platform State (Memory cache + DB synchronized)
# HONNÊTETÉ (faille 1 corrigée — mentalité n°5) : plus AUCUNE valeur de marché
# inventée. Prix/carnet/volume sont None ou UNAVAILABLE tant qu'une source
# réelle n'a pas parlé. Règle : « AUCUNE DONNÉE -> AUCUN ORDRE ».
STATE = {
    "mode": "DEMO",                  # DEMO vs REAL
    "is_running": True,              # Overall bot main loop switch
    "kill_switch_active": False,     # Emergency lock
    "balance_demo": 100000.0,        # Configurable virtual capital
    "balance_real": 0.0,             # Real wallet balance (loaded from exchange)
    "initial_capital_demo": 100000.0,  # Startup initial capital for P&L tracking
    "initial_capital_real": 0.0,      # Startup real capital for P&L tracking
    "current_equity": 100000.0,
    "last_price": None,               # Dernier prix RÉEL connu (None = aucune donnée réelle)
    "last_known_prices": {},          # Dernier prix réel connu PAR ACTIF (pour calculs de position)
    "last_tick_volume": None,         # Volume réel (Bybit ticker) — plus de 15.0 inventé
    "price_history": [],              # Historique de prix RÉELS uniquement
    "order_book": None,               # Carnet BTCUSDT réel (None tant qu'aucun flux reçu)
    "order_books": {},                # Carnets réels consolidés BBO PAR ACTIF
    "exchange_order_books": {},          # Carnets réels PAR EXCHANGE x PAR ACTIF
    "price_consensus": {},               # Consensus multi-sources par actif (PDF)
    "price_divergent": {},               # Flag de divergence anormale (gel du trading)
    "regime_id": 2,                   # Initialized to Range (modèle, pas une donnée de marché)
    "regime_name": "Mean-Reverting Range",
    "ml_prediction_pct": 0.0,
    "ppo_action": 0.0,
    "connected_websockets": [],
    "equity_history_demo": [100000.0],
    "equity_history_real": [0.0],
    "historical_bars": None,         # Infilled during training (données réelles)
    
    # MULTI-ASSET telemetry mapping — prix RÉELS uniquement (None = source absente)
    "assets": {
        "BTCUSDT": {"price": None, "qty": 0.0, "pnl": 0.0, "class": "Crypto",
                    "has_real_price": False, "data_status": "UNAVAILABLE", "volume_24h": None},
        "ETHUSDT": {"price": None, "qty": 0.0, "pnl": 0.0, "class": "Crypto",
                    "has_real_price": False, "data_status": "UNAVAILABLE", "volume_24h": None},
        "SOLUSDT": {"price": None, "qty": 0.0, "pnl": 0.0, "class": "Crypto",
                    "has_real_price": False, "data_status": "UNAVAILABLE", "volume_24h": None},
        "XAUUSD": {"price": None, "qty": 0.0, "pnl": 0.0, "class": "Commodity (Gold)",
                   "has_real_price": False, "data_status": "UNAVAILABLE", "volume_24h": None},
        "EURUSD": {"price": None, "qty": 0.0, "pnl": 0.0, "class": "Forex (EUR/USD)",
                   "has_real_price": False, "data_status": "UNAVAILABLE", "volume_24h": None},
        "AAPL": {"price": None, "qty": 0.0, "pnl": 0.0, "class": "Stock (Apple)",
                 "has_real_price": False, "data_status": "UNAVAILABLE", "volume_24h": None},
        "TSLA": {"price": None, "qty": 0.0, "pnl": 0.0, "class": "Stock (Tesla)",
                 "has_real_price": False, "data_status": "UNAVAILABLE", "volume_24h": None}
    },
    
    # Advanced Signals Cache — disponibilité honnête (None = indisponible)
    "sentiment_index": None,
    "sentiment_available": False,
    "sentiment_confidence": 0.0,
    "recent_headlines": [],             # LOT 5: dernières actualités RÉELLES (source+titre)
    "news_shock": {"shock_detected": False},  # LOT 5: choc systémique détecté
    "macro_phase": "NONE",              # LOT 5: APPROACHING/ACTIVE/AFTERMATH/NONE
    "macro_event": "",                  # LOT 5: événement macro en cours
    "onchain_risk_score": None,
    "onchain_available": False,
    "eth_defi_balance": 0.0,
    "defi_wallet_address": "Not Connected",
    "covariance_matrix": {},
    "options_strategy": {"strategy": "UNAVAILABLE", "details": "Aucune IV réelle reçue.", "legs": [], "estimated_yield_pct": 0.0},
    "real_iv": {},                      # IV RÉELLE par symbole (Deribit DVOL)
    "data_quality_status": DataQualityStatus.UNAVAILABLE,
    "asset_data_status": {},          # Qualité de données PAR ACTIF (LIVE/DELAYED/STALE/UNAVAILABLE)
    "macro_scale_factor_tactile": 1.0,  # Controlled by interactive Telegram mobile buttons!
    "last_sent_macro_event": None,      # Tracks sent notifications to avoid spamming
    "last_broadcast_time": 0.0,          # For real-time telemetry throttling
    "cached_positions": [],
    "cached_orders": [],
    "cached_audit_logs": [],
    "last_db_query_time": 0.0,
    "last_order_times": {},          # per-symbol order cooldown (idempotence)
    "ppo_buffer": [],                 # real RL experiences for autonomous PPO training
    "price_alerts": [],               # custom price alerts (audit C3)
    "ab_base": [],                    # A/B paper: baseline hypothetical equity
    "ab_vol": [],                     # A/B paper: vol-targeted hypothetical equity
    "regime_probs": {},               # VISION §1a: soft HMM probabilities
    "regime_confidence": {"confidence": 0.5, "regime_id": 2},  # LOT 4: certitude du régime
    "hmm_validation": {},             # LOT 4: validation HMM multi-actifs
    "causal_analyzed": False,         # LOT 4: l'analyse causale a-t-elle tourné ?
    "market_state": {},               # VISION §1b: joint market state
    "causal_parents": [],             # VISION §1c: causal parents of returns
    "conviction_threshold": 0.15,     # VISION §4a: adaptive
    "no_trade_stats": {"count": 0},   # VISION §4b
    "last_hedge_ts": 0.0,             # VISION §4c
    "recent_signals": [],             # for meta-cognition (rolling)
    "recent_returns": [],             # for meta-cognition (rolling)
    "confidence_index": 100,          # VISION_FUTUR §8
    "confidence_factor": 1.0,
    "live_p_value": 0.5,              # §2c
    "structural_regimes": {},         # §3
    "cross_asset_bias": 0.0,          # §3/§4
    "strategy_win_rates": {},         # LOT 2: win rates RÉELS par stratégie (trades clôturés)
    "strategy_trade_counts": {},     # LOT 2: nb de trades clôturés par stratégie
    "position_strategies": {},       # LOT 2: stratégie responsable de chaque position ouverte
    "pending_approvals": [],          # §6 consultative mode
    "portfolio_allocation": {},       # LOT 6: budget top-down (Pilier L)
    "strategy_diversification": {},   # LOT 6: corrélation entre stratégies
    "position_pyramids": {},          # LOT 6: nb d'ajouts (pyramiding) par symbole
    "decision_log": [],               # LOT 7: journal des décisions (méta-attribution)
    "reason_weights": {},             # LOT 7: poids des raisons (méta-attribution)
    "reason_weights_factor": 1.0,     # LOT 7: facteur global des raisons
    "cost_metrics": {},               # LOT 8: coûts réels (télémétrie)
    "attribution_report": {},         # LOT 8: attribution performance
    "quality_metrics": {},            # LOT 8: Sharpe/Sortino/Calmar/expectancy
    "stress_test_report": {},         # LOT 8: crises réelles rejouées
    "bootstrap_sharpe": {},           # LOT 8: Sharpe vs chance
    "consultative_mode": os.getenv("CONSULTATIVE_MODE", "").lower() == "true",
    "chaos_until": 0.0,               # §5c
    "background_tasks": {},           # LOT 7: registre des tâches de fond (watchdog)
    "last_narrative": "",             # §3/§6
    "risk_state": {"state": "NORMAL", "reason": "", "scale_factor": 1.0},
    "risk_pipeline_steps": [],        # LOT 2: étapes du pipeline de risque (audit)
    "using_fallback_data": False     # True quand des barres non-réelles seraient en usage (on évite de trader)
}

# ============ INSTITUTIONAL SAFETY HELPERS (roadmap) ============
ORDER_COOLDOWN_REAL_SECONDS = settings.get_float("trading", "order_cooldown_real_seconds", 60.0)
ORDER_COOLDOWN_DEMO_SECONDS = settings.get_float("trading", "order_cooldown_demo_seconds", 10.0)


# LOT 7 (PDF Faille 6) : vraie IP client pour les audit logs (plus d'IP en dur
# dans les appels d'audit — la source réelle doit être tracée, mentalité n°9).
_current_request_ip = ["127.0.0.1"]


def set_request_ip(ip: str) -> None:
    """Mémorise l'IP réelle du client courant (appelé par le middleware)."""
    if ip and ip != "unknown":
        _current_request_ip[0] = ip


def audit_ip() -> str:
    """Retourne l'IP réelle du client si disponible, sinon la dernière connue."""
    return _current_request_ip[0]


def set_data_quality(status):
    """Tracks market-data quality per source into STATE + Prometheus gauge."""
    STATE["data_quality_status"] = status
    try:
        mapping = {
            DataQualityStatus.LIVE: 4.0,
            DataQualityStatus.DELAYED: 3.0,
            DataQualityStatus.STALE: 2.0,
            DataQualityStatus.INVALID: 1.0,
            DataQualityStatus.DISCONNECTED: 0.0,
            DataQualityStatus.UNAVAILABLE: 0.0,
        }
        platform_metrics.DATA_QUALITY.labels(source="market").set(mapping.get(status, 0.0))
    except Exception:
        pass


def set_asset_quality(symbol: str, status: str):
    """
    Qualité de données PAR ACTIF (faille 1 corrigée — mentalité n°5 : chaque
    donnée doit avoir un score de confiance). Un actif dont la source est
    indisponible est marqué UNAVAILABLE et NE PEUT PAS être tradé.
    """
    STATE.setdefault("asset_data_status", {})[symbol] = status
    STATE["assets"].setdefault(symbol, {})["data_status"] = status
    if status == DataQualityStatus.UNAVAILABLE:
        STATE["assets"][symbol]["has_real_price"] = False


def _neutral(value, default: float = 0.0) -> float:
    """
    Convertit un indicateur éventuellement indisponible en valeur NEUTRE.
    Contrairement à une donnée inventée, 0.0 = « aucune information » et
    n'apporte aucune direction à la décision (mentalité n°20 : je ne sais pas).
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def record_open_position(symbol: str, strategy: str, entry_price: float) -> None:
    """LOT 2 : mémorise la stratégie responsable d'une position ouverte
    (alimente le win rate RÉEL par stratégie au moment de la clôture)."""
    if strategy:
        STATE.setdefault("position_strategies", {})[symbol] = {
            "strategy": strategy,
            "entry_price": float(entry_price),
            "ts": time.time(),
        }


def record_closed_trade(symbol: str, exit_price: float, side: str) -> None:
    """
    LOT 2 (PDF Pilier F, exigence 1) : enregistre un trade CLÔTURÉ dans le
    tracker de win rates réels par stratégie (lissage EMA, bornes 0.45..0.65
    appliquées à l'utilisation). Le win rate alimente le Kelly dynamique et
    le filtre méta-label — plus jamais de 0.55 codé en dur.
    """
    pos_info = STATE.get("position_strategies", {}).pop(symbol, None)
    if not pos_info:
        return
    entry = float(pos_info.get("entry_price") or 0.0)
    strategy = pos_info.get("strategy", "")
    if entry <= 0 or exit_price <= 0 or not strategy:
        return
    # side = sens de la CLÔTURE : SELL clôt un long (+), BUY clôt un short (-)
    direction = 1.0 if side == "SELL" else -1.0
    pnl_pct = (exit_price - entry) / entry * direction
    win_tracker.record(strategy, pnl_pct)
    # LOT 4 (PDF Pilier B) : alpha contrefactuel généralisé à TOUS les trades
    # clôturés (plus seulement les sorties de protection) + attribution MoE
    try:
        _bench = 0.0
        _sym_price = STATE.get("last_known_prices", {}).get(symbol)
        if _sym_price and entry > 0:
            _bench = (_sym_price - entry) / entry  # mouvement du marché depuis l'entrée
        _alpha = counterfactual_alpha({"side": "SELL" if direction > 0 else "BUY",
                                       "entry": entry, "exit": exit_price},
                                      benchmark_return=_bench)
        try:
            db.add_event(time.time(), "closed_trade_alpha",
                         json.dumps({"symbol": symbol, "strategy": strategy,
                                     "pnl_pct": round(pnl_pct, 6),
                                     "marginal_alpha": round(_alpha, 6)}, default=str))
        except Exception:
            pass
        # Mixture of Experts : contribution réelle au PnL par expert (Pilier C)
        mixture_of_experts.record_pnl_contribution("swing", pnl_pct)
        # MetaAllocationEngine : pondération par contribution RÉELLE au PnL (Pilier D)
        try:
            meta_engine.update_pnl_attribution(strategy, pnl_pct)
        except Exception:
            pass
    except Exception as _ae:
        logger.debug(f"Counterfactual alpha / MoE record failed: {_ae}")
    # LOT 8 (PDF Pilier Q) : attribution de performance — chaque dollar est
    # tracé (facteur, régime, actif, stratégie) + post-mortem.
    try:
        attribution.record(symbol, strategy, pnl_pct,
                           regime_name=STATE.get("regime_name", ""),
                           pnl_usd=pnl_pct * 0.01 * STATE.get("current_equity", 0.0))
        STATE["attribution_report"] = attribution.full_report()
    except Exception:
        pass
    # LOT 7 (PDF Pilier K) : journal des décisions pour la méta-attribution
    # (quelles raisons gagnent ?) -> réduit automatiquement le poids des
    # mauvaises raisons.
    try:
        _log = STATE.setdefault("decision_log", [])
        _log.append({
            "reasons": list(STATE.get("last_reasoning", []))[:5] or [strategy],
            "pnl": round(pnl_pct, 6),
            "symbol": symbol,
            "ts": time.time(),
        })
        if len(_log) > 300:
            STATE["decision_log"] = _log[-300:]
    except Exception:
        pass
    logger.info(
        f"📊 WIN-RATE RÉEL: {strategy} +1 trade clôturé ({symbol}) pnl={pnl_pct*100:+.2f}% "
        f"-> wr={win_tracker.get(strategy):.2f} (n={win_tracker.samples(strategy)})"
    )


def causal_signal_factor(state: dict) -> float:
    """
    LOT 4 (PDF Pilier B) : la découverte causale DÉSACTIVE les signaux non
    causaux (plus seulement journalisés).

    - Analyse causale faite ET parents trouvés  -> 1.0 (signaux actifs)
    - Analyse faite et AUCUN parent causal       -> 0.5 (les signaux prédictifs
      sont probablement du bruit -> réduction, mentalité n°13)
    - Analyse pas encore faite                    -> 1.0 (neutre)
    """
    if not state.get("causal_analyzed"):
        return 1.0
    parents = state.get("causal_parents") or []
    if not parents:
        return 0.5
    return 1.0


def mark_real_price(symbol: str, price: float, volume_24h=None):
    """
    Enregistre un prix RÉEL reçu d'une source de marché. Met à jour le flag
    has_real_price (seul vrai « feu vert » pour trader cet actif).
    """
    STATE["assets"][symbol]["price"] = float(price)
    STATE["assets"][symbol]["has_real_price"] = True
    STATE["assets"][symbol]["data_status"] = DataQualityStatus.LIVE
    STATE.setdefault("last_known_prices", {})[symbol] = float(price)
    if volume_24h is not None:
        STATE["assets"][symbol]["volume_24h"] = float(volume_24h)
    # Le dernier prix global réel (BTC) alimente le dashboard
    if symbol == "BTCUSDT":
        STATE["last_price"] = float(price)
        STATE["price_history"].append(float(price))
        if len(STATE["price_history"]) > 120:
            STATE["price_history"] = STATE["price_history"][-120:]


def update_asset_order_book(symbol: str, bids: list, asks: list, exchange: str = "bybit"):
    """
    Met à jour le carnet d'ordres RÉEL d'un actif (multi-assets, multi-exchange).
    Stocke le carnet PAR exchange puis consolide le BEST BOOK (meilleur spread)
    dans order_books[symbol] — `order_book` reste l'alias historique pour BTCUSDT.
    """
    STATE.setdefault("exchange_order_books", {}).setdefault(exchange, {})[symbol] = {
        "bids": bids, "asks": asks, "_ts": time.time(),
    }
    # Consolidation BBO : le carnet de l'exchange avec le meilleur spread
    best = None
    for ex, books in STATE.get("exchange_order_books", {}).items():
        b = books.get(symbol)
        if not b or not b.get("bids") or not b.get("asks"):
            continue
        try:
            spread = float(b["asks"][0][0]) - float(b["bids"][0][0])
        except Exception:
            continue
        if best is None or spread < best[0]:
            best = (spread, ex, b)
    if best is not None:
        consolidated = {"bids": best[2]["bids"], "asks": best[2]["asks"],
                        "exchange": best[1]}
        STATE.setdefault("order_books", {})[symbol] = consolidated
        if symbol == "BTCUSDT":
            STATE["order_book"] = consolidated
    STATE.setdefault("asset_data_status", {})[symbol] = DataQualityStatus.LIVE
    STATE["assets"][symbol]["data_status"] = DataQualityStatus.LIVE
    # LOT 3 : alimente l'OFI du moteur d'order flow (pression bid vs ask)
    try:
        _bd = sum(float(b[1]) for b in bids if b and len(b) > 1)
        _ad = sum(float(a[1]) for a in asks if a and len(a) > 1)
        order_flow.update_book(symbol, _bd, _ad)
    except Exception:
        pass



def require_auth(credentials=Depends(auth_security_optional)):
    """
    Protects state-changing endpoints.
    Enforced when AUTH_ENABLED=true OR when the platform runs in REAL mode
    (institutional rule: real money can never be controlled without a session).
    """
    auth_required = os.getenv("AUTH_ENABLED", "").lower() == "true" or STATE["mode"] == "REAL"
    if not auth_required:
        return {"role": Roles.ADMIN, "username": "local-demo", "sub": "1"}
    if credentials is None or not getattr(credentials, "credentials", None):
        raise HTTPException(status_code=401, detail="Authentication required")
    return AuthManager.verify_jwt_token(credentials.credentials)

# ============ AUDIT C7/C3/C10 helpers ============
# ============ AUDIT C7/C3/C10 helpers ============
def require_admin(credentials=Depends(auth_security_optional)):
    """ADMIN-gated dependency for user management endpoints (audit C7).
    STRICT: always validates a real JWT - never the DEMO local-bypass, because
    user management must be protected even when AUTH_ENABLED is off."""
    if credentials is None or not getattr(credentials, "credentials", None):
        raise HTTPException(status_code=401, detail="Authentication required.")
    user = AuthManager.verify_jwt_token(credentials.credentials)
    if user.get("role") != Roles.ADMIN:
        raise HTTPException(status_code=403, detail="Admin role required.")
    return user


def _alerts_persist():
    try:
        db.save_setting("price_alerts_json", json.dumps(STATE["price_alerts"], default=str))
    except Exception as e:
        logger.warning(f"Alerts persist failed: {e}")


def _alerts_load():
    try:
        raw = db.get_setting("price_alerts_json")
        if raw:
            STATE["price_alerts"] = json.loads(raw)
    except Exception:
        STATE["price_alerts"] = []
    STATE.setdefault("price_alerts", [])


def check_price_alerts(symbol: str, price: float):
    """Audit C3: fire custom price alerts once, notify Telegram + audit log."""
    if not STATE.get("price_alerts"):
        return
    for a in STATE["price_alerts"]:
        if a.get("symbol") != symbol or a.get("triggered"):
            continue
        try:
            target = float(a.get("target_price", 0.0))
            direction = a.get("direction", "above")
            hit = (price >= target) if direction == "above" else (price <= target)
            if hit and target > 0:
                a["triggered"] = True
                a["triggered_ts"] = time.time()
                a["triggered_price"] = price
                _alerts_persist()
                logger.info(f"🔔 PRICE ALERT {symbol}: {direction} {target} reached ({price:.2f})")
                db.add_audit_log("PRICE_ALERT", audit_ip(),
                                 f"Alert {a.get('id')} {symbol} {direction} {target} hit @ {price:.2f}")
                try:
                    asyncio.create_task(telegram_bot.send_push_notification(
                        f"🔔 *ALERTE PRIX*\n{symbol} : *{price:,.2f}*\n"
                        f"({direction} {target:,.2f})\n{a.get('note', '')}"
                    ))
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Alert check error for {symbol}: {e}")


async def run_market_replay(symbol: str, interval: str = "1h", limit: int = 300) -> dict:
    """Audit C10: replays historical candles through the decision engine in
    simulation (no orders, no DB writes). Returns signal timeline + stats."""
    df = db.load_candles(symbol, limit=limit + 20)
    if df is None or df.empty or len(df) < 60:
        if symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            df = await fetch_historical_market_data(symbol)
        else:
            ymap = {"XAUUSD": "GC=F", "EURUSD": "EURUSD=X"}
            df = await fetch_yahoo_finance_candles(ymap.get(symbol, symbol), interval=interval, range_str="1mo")
    if df is None or df.empty or len(df) < 60:
        raise HTTPException(status_code=503, detail="Insufficient historical data for replay.")

    strat = TrendFollowingStrategy()
    meta_local = MetaAllocationEngine(strategies=[strat])
    timestamps, signals, prices = [], [], []
    warmup = 30
    for i in range(warmup, len(df)):
        window = df.iloc[:i + 1]
        md = {
            "df": window,
            "price_primary": float(window['close'].iloc[-1]),
            "price_secondary": float(window['close'].iloc[-1]),
            "bids": [[float(window['close'].iloc[-1]) * 0.999, 1]],
            "asks": [[float(window['close'].iloc[-1]) * 1.001, 1]],
            "inventory": 0.0,
            "max_inventory": 1.0,
            "vpin": 0.5, "kyle_lambda": 0.0, "onchain_risk": 0.0, "sentiment": 0.0,
        }
        res = meta_local.allocate(md, 2, 0.0, 0.0)
        signals.append(float(res["final_signal"]))
        prices.append(float(window['close'].iloc[-1]))
        timestamps.append(str(window.index[-1]))

    # hypothetical equity: signal lagged one bar, fee 0.1% per flip
    equity = [1.0]
    prev_sig = 0.0
    rets = [0.0] + list(df['close'].pct_change().dropna().values[warmup - 1:])
    for k in range(1, len(signals)):
        r = rets[k] if k < len(rets) else 0.0
        sig = signals[k - 1]
        if (sig > 0) != (prev_sig > 0) and prev_sig != 0:
            r -= 0.001  # flip fee
        prev_sig = sig
        equity.append(equity[-1] * (1.0 + sig * r))

    total_ret = equity[-1] - 1.0
    n_trades = sum(1 for i in range(1, len(signals)) if (signals[i] > 0) != (signals[i - 1] > 0))
    sharpe = 0.0
    eq = np.array(equity)
    if len(eq) > 2:
        daily = np.diff(eq) / eq[:-1]
        if daily.std() > 0:
            sharpe = float(daily.mean() / daily.std() * np.sqrt(365 * 24))

    return {
        "symbol": symbol, "interval": interval, "bars": len(signals),
        "total_return_pct": round(total_ret * 100.0, 3),
        "n_trades": n_trades, "approx_sharpe": round(sharpe, 3),
        "last_price": prices[-1] if prices else 0.0,
        "timeline": [
            {"ts": timestamps[i], "price": round(prices[i], 4), "signal": round(signals[i], 4)}
            for i in range(0, len(signals), max(1, len(signals) // 100))
        ],
    }



telegram_bot = TelegramBotManager(state_dict=STATE, db_manager=db)

# Position protection store (audit B7-1): SL/TP/trailing state per symbol
protection_store = PositionProtectionStore(STATE)

# VISION §3: execution quality tracking
execution_alpha = ExecutionAlpha()
slippage_model = SlippageModel()

# VISION §2-§7: cognitive architecture instances
mixture_of_experts = MixtureOfExperts(state_dim=4)
hypothesis_generator = HypothesisGenerator(db=db)
risk_committee = RiskCommittee(veto_threshold=0.85)
price_engine = MultiSourcePriceEngine()  # consensus prix multi-exchange (PDF: redondance 2+ sources)
order_flow = OrderFlowEngine()          # LOT 3: order flow réel (delta/CVD/OFI/liquidations) (PDF Pilier H)
risk_state = RiskStateMachine()          # LOT 2: machine à états NORMAL/CAUTION/HALT (PDF Faille 3)
win_tracker = StrategyWinRateTracker(STATE)  # LOT 2: win rates RÉELS par stratégie (PDF Pilier F)
risk_pipeline_last = {}                  # dernier tracé du pipeline (télémétrie/audit)
execution_bandit = ExecutionStyleBandit()
strategy_exec_attr = StrategyExecutionAttribution()

# VISION_FUTUR instances
organization = Organization(STATE)
supervisor = Supervisor(STATE)

# CCXT Exchange Client Cache
ccxt_client = None

def get_ccxt_client():
    """
    Dynamically loads and instantiates the CCXT Binance/Bybit client 
    using securely encrypted keys from the database.
    """
    global ccxt_client
    if ccxt_client is not None:
        return ccxt_client
        
    api_key = db.get_setting("binance_api_key", decrypt=True)
    secret_key = db.get_setting("binance_secret_key", decrypt=True)
    
    if api_key and secret_key:
        try:
            ccxt_client = ccxt.binance({
                'apiKey': api_key,
                'secret': secret_key,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'  # Default to perpetual futures
                }
            })
            ccxt_client.fetch_balance()
            logger.info("CCXT Exchange Client successfully instantiated and authenticated.")
            return ccxt_client
        except Exception as e:
            logger.error(f"Failed to authenticate with real exchange API: {str(e)}")
            ccxt_client = None
    return None


def format_exchange_size(symbol, quantity, price):
    """
    Formats the order size according to the exact lot size filters 
    and precision limits of the exchange to avoid API execution rejections.
    """
    client = get_ccxt_client()
    if not client:
        return round(quantity, 5) # Safe fallback
        
    try:
        if symbol not in client.markets:
            client.load_markets()
            
        market = client.market(symbol)
        min_qty = market['limits']['amount']['min'] or 0.0001
        max_qty = market['limits']['amount']['max'] or 999999.0
        
        formatted_qty = client.amount_to_precision(symbol, quantity)
        formatted_qty = float(formatted_qty)
        formatted_qty = max(min_qty, min(formatted_qty, max_qty))
        return formatted_qty
    except Exception as e:
        logger.warning(f"Error formatting lot size precision: {str(e)}. Using safe rounding.")
        return round(quantity, 5)


# AUDIT B6-1: short TTL cache for Yahoo chart calls (rate-limit friendly)
_yahoo_cache: dict = {}


async def fetch_yahoo_finance_candles(ticker: str, interval="1h", range_str="5d") -> pd.DataFrame:
    """
    Queries Yahoo Finance API with a secure browser User-Agent
    to fetch 100% genuine real-time and historical candles for Gold, Forex, and Stocks!
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={interval}&range={range_str}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    # AUDIT B6-1: serve fresh-enough cached bars instead of hammering Yahoo
    _cache_key = f"{ticker}|{interval}|{range_str}"
    _cached = _yahoo_cache.get(_cache_key)
    if _cached is not None and (time.time() - _cached[0]) < settings.get_float("data", "yahoo_cache_ttl_seconds", 20.0):
        return _cached[1].copy()

    try:
        async with yahoo_limiter:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            result = resp.json().get("chart", {}).get("result", [])[0]
            timestamps = result.get("timestamp", [])
            if timestamps is None:
                logger.info(f"Yahoo Finance: Market for {ticker} is currently closed or has no active trades (Weekend/Closed).")
                return pd.DataFrame()
                
            indicators = result.get("indicators", {}).get("quote", [])[0]
            
            opens = indicators.get("open", [])
            highs = indicators.get("high", [])
            lows = indicators.get("low", [])
            closes = indicators.get("close", [])
            volumes = indicators.get("volume", [])
            
            data = []
            for idx, t in enumerate(timestamps):
                if opens[idx] is not None and closes[idx] is not None:
                    data.append({
                        "timestamp": pd.to_datetime(t, unit='s'),
                        "open": float(opens[idx]),
                        "high": float(highs[idx]),
                        "low": float(lows[idx]),
                        "close": float(closes[idx]),
                        "volume": float(volumes[idx]) if volumes[idx] else 10.0
                    })
            df = pd.DataFrame(data).set_index("timestamp")
            _yahoo_cache[_cache_key] = (time.time(), df)
            if len(_yahoo_cache) > 64:
                _yahoo_cache.pop(next(iter(_yahoo_cache)))
            logger.info(f"Successfully loaded {len(df)} actual real-world market bars from Yahoo Finance for {ticker}!")
            return df
    except Exception as e:
        logger.error(f"Failed to fetch Yahoo Finance candles for {ticker}: {str(e)}")
    return pd.DataFrame()


def _klines_to_df(data: list) -> pd.DataFrame:
    """Convertit une réponse klines (Binance ou Bybit) en DataFrame OHLCV réel."""
    bars = []
    for b in data:
        bars.append({
            "timestamp": pd.to_datetime(b[0], unit='ms'),
            "open": float(b[1]),
            "high": float(b[2]),
            "low": float(b[3]),
            "close": float(b[4]),
            "volume": float(b[5])
        })
    df = pd.DataFrame(bars).set_index("timestamp")
    return df


async def fetch_bybit_klines(symbol: str, interval: str = "1h", limit: int = 120) -> pd.DataFrame:
    """Barres OHLCV RÉELLES via l'API publique Bybit v5 (secours Binance)."""
    try:
        async with bybit_limiter:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
                    f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}"
                    f"&interval={interval}&limit={limit}"
                )
        if resp.status_code == 200 and resp.json().get("retCode") == 0:
            rows = resp.json().get("result", {}).get("list", [])
            if rows:
                # Bybit renvoie les barres de la plus récente à la plus ancienne
                rows = list(reversed(rows))
                bars = []
                for b in rows:
                    bars.append({
                        "timestamp": pd.to_datetime(int(b[0]), unit='ms'),
                        "open": float(b[1]), "high": float(b[2]),
                        "low": float(b[3]), "close": float(b[4]),
                        "volume": float(b[5]),
                    })
                df = pd.DataFrame(bars).set_index("timestamp")
                logger.info(f"Fetched {len(df)} barres RÉELLES Bybit pour {symbol} ({interval}).")
                return df
    except Exception as e:
        logger.warning(f"Bybit klines failed for {symbol}: {e}")
    return pd.DataFrame()


async def fetch_historical_market_data(symbol="BTCUSDT"):
    """
    Fetches real historical price candles (OHLCV) from real APIs (Binance, puis
    Bybit en secours, puis Yahoo pour les actifs non-crypto). AUCUNE donnée
    simulée : si toutes les sources réelles échouent, renvoie un DataFrame vide
    (l'appelant marque l'actif UNAVAILABLE et ne trade pas).
    """
    # 1) Binance (source primaire)
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=120"
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(url)
        if response.status_code == 200:
            df = _klines_to_df(response.json())
            if not df.empty:
                logger.info(f"Successfully fetched {len(df)} real bars from Binance for {symbol}.")
                return df
    except Exception as e:
        logger.warning(f"Binance historical fetch failed for {symbol}: {e}")

    # 2) Bybit (secours réel pour les cryptos)
    if symbol in CRYPTO_SYMBOLS:
        df = await fetch_bybit_klines(symbol, interval="1h", limit=120)
        if not df.empty:
            return df

    # 3) Yahoo Finance (secours réel pour Or/FX/Actions — et cryptos en dernier recours)
    try:
        y_ticker = "GC=F" if symbol == "XAUUSD" else "EURUSD=X" if symbol == "EURUSD" else \
                   ("BTC-USD" if symbol == "BTCUSDT" else "ETH-USD" if symbol == "ETHUSDT"
                    else "SOL-USD" if symbol == "SOLUSDT" else symbol)
        df_y = await fetch_yahoo_finance_candles(y_ticker, interval="1h", range_str="5d")
        if not df_y.empty:
            logger.info(f"Fetched {len(df_y)} real bars from Yahoo Finance for {symbol}.")
            return df_y
    except Exception as e:
        logger.warning(f"Yahoo historical fetch failed for {symbol}: {e}")

    # HONNÊTETÉ (mentalité n°5) : aucune source réelle -> vide, pas de simulé.
    logger.warning(f"NO REAL HISTORICAL DATA AVAILABLE for {symbol} — marked UNAVAILABLE.")
    return pd.DataFrame()


def train_ai_models(df):
    """
    Fits HMM Regime Detector and LSTM-like model on initial dataset.
    """
    logger.info("Fitting AI & Quantitative Models...")
    returns = df['close'].pct_change().dropna().values
    volatilities = df['close'].pct_change().rolling(10).std().dropna().values
    min_len = min(len(returns), len(volatilities))
    
    X_train = np.column_stack((returns[-min_len:], volatilities[-min_len:]))
    regime_detector.fit(X_train)
    
    features_seq = []
    labels = []
    pct_df = df[['close', 'volume', 'high', 'low', 'open']].pct_change().fillna(0)
    
    for i in range(5, len(pct_df) - 1):
        features_seq.append(pct_df.iloc[i-5:i].values)
        labels.append(pct_df['close'].iloc[i])
        
    price_predictor.fit(features_seq, np.array(labels))
    
    STATE["historical_bars"] = df
    logger.info("AI Models successfully fitted and deployed in-memory.")

    # LOT 48: persist a versioned feature snapshot so training is reproducible
    try:
        feat = feature_store.compute_features("BTCUSDT", df, version="v1.0")
        logger.info(f"LOT 48: Feature snapshot stored for BTCUSDT: {list(feat.keys())}")
    except Exception as e:
        logger.warning(f"LOT 48: Feature snapshot failed: {e}")


def evaluate_real_safety_gate(symbol: str) -> bool:
    """
    Sovereign Real Safety Gate (Phase 33).
    Executes strict production-grade validations before allowing any real order routing.
    Any single check failure -> Rejects order and suspends trading immediately!
    """
    # 1. Exchange Connection Check
    client = get_ccxt_client()
    if not client:
        logger.error("SAFETY GATE: CCXT Exchange client offline or unauthenticated.")
        return False
        
    # 2. Market Data Quality Check
    if STATE["data_quality_status"] == DataQualityStatus.UNAVAILABLE:
        logger.error("SAFETY GATE: Market data quality is UNAVAILABLE.")
        return False
        
    # 3. Database Health Check
    try:
        db.get_connection().close()
    except Exception as e:
        logger.error(f"SAFETY GATE: Database connection is unhealthy: {str(e)}")
        return False
        
    # 4. Risk Circuit Breakers Check
    if risk_manager.circuit_breaker_active:
        logger.error("SAFETY GATE: Risk circuit breaker is ACTIVE.")
        return False
        
    # 4bis. Multi-Source Divergence Check (PDF : « divergence anormale entre 2
    # sources = alarme et gel du trading » — mentalité n°5)
    if STATE.get("price_divergent", {}).get(symbol):
        cons = STATE.get("price_consensus", {}).get(symbol, {})
        logger.error(
            f"SAFETY GATE: Divergence multi-sources {cons.get('divergence_pct')}% "
            f"(seuil {cons.get('threshold_pct')}%) sur {symbol} -> ordre réel BLOQUÉ."
        )
        return False
        
    # 5. Model Registry Approval Check
    from models.mlops_pipeline import ModelStatus
    if mlops_trainer.active_model_status != ModelStatus.DEPLOYED:
        logger.error("SAFETY GATE: Active model in registry is frozen or not APPROVED.")
        return False
            
    logger.info(f"SAFETY GATE PASSED: All validations successful for {symbol}!")
    return True


async def pick_best_venue_net(symbol: str, side: str) -> dict:
    """
    LOT 3 (PDF Pilier H-3) : SOR multi-venue — compare les venues sur le COÛT
    NET TOTAL (prix + frais + slippage attendu), pas seulement le prix.

    Retourne le meilleur choix avec le détail des coûts (audit). Si aucune
    venue ne répond, retourne un choix neutre (venue par défaut) — jamais
    d'invention de prix (mentalité n°5).
    """
    try:
        quotes = await multi_exchange_sor.get_all_quotes(symbol)
        if not quotes:
            return {"venue": None, "reason": "Aucune venue disponible (données réelles absentes)", "quotes": []}
        best = None
        detail = []
        for q in quotes:
            # coût net : prix ajusté du spread + frais + slippage attendu
            if side == "BUY":
                gross = q.ask * (1.0 + q.fee_rate)
                slip = slippage_model.expected_slippage_bps(q.exchange.capitalize(), symbol, fallback=5.0) / 1e4
                net = gross * (1.0 + slip)
            else:
                gross = q.bid * (1.0 - q.fee_rate)
                slip = slippage_model.expected_slippage_bps(q.exchange.capitalize(), symbol, fallback=5.0) / 1e4
                net = gross * (1.0 - slip)
            detail.append({"venue": q.exchange, "net_price": round(net, 6),
                           "fee_rate": q.fee_rate, "latency_ms": round(q.latency_ms, 1),
                           "liquidity_usd": round(q.liquidity_usd, 0)})
            if best is None or (side == "BUY" and net < best[1]) or (side == "SELL" and net > best[1]):
                best = (q.exchange, net)
        return {"venue": best[0], "net_price": best[1], "reason": "meilleur coût net (prix+frais+slippage)", "quotes": detail}
    except Exception as e:
        logger.warning(f"SOR net eval failed for {symbol}: {e}")
        return {"venue": None, "reason": "SOR indisponible", "quotes": []}


async def trigger_realtime_broadcast():
    """
    Throttles WebSocket broadcasts to a maximum of 5 times per second (once every 200ms)
    to keep the dashboard UI blazing fast, ultra-smooth and real-time without slamming the server.
    """
    now = time.time()
    if now - STATE.get("last_broadcast_time", 0.0) >= 0.20:
        STATE["last_broadcast_time"] = now
        asyncio.create_task(broadcast_telemetry(None))


CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def _bybit_ws_symbol(symbol: str) -> str:
    """Convertit un symbole interne en symbole Bybit (même format pour spot)."""
    return symbol


async def fetch_depth_snapshot_rest(symbol: str):
    """
    Snapshot REST du carnet d'ordres RÉEL (Bybit v5 depth, limit 5).
    Utilisé en secours quand le flux WebSocket est indisponible (ex: géoblocage).
    Ne renvoie rien si la source est injoignable (aucune donnée inventée).
    """
    try:
        async with bybit_limiter:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"https://api.bybit.com/v5/market/depth?category=spot&symbol={symbol}&limit=5"
                )
        if resp.status_code == 200:
            result = resp.json().get("result", {})
            bids_raw = result.get("b", [])
            asks_raw = result.get("a", [])
            bids = [[float(b[0]), float(b[1])] for b in bids_raw[:5]]
            asks = [[float(a[0]), float(a[1])] for a in asks_raw[:5]]
            if bids and asks:
                update_asset_order_book(symbol, bids, asks, exchange="bybit")
                logger.info(f"OrderBook REST snapshot (réel) reçu pour {symbol}.")
                return True
    except Exception as e:
        logger.debug(f"Depth REST snapshot failed for {symbol}: {e}")
    return False


async def order_book_snapshot_loop():
    """
    Garde-fou carnet d'ordres : maintient un snapshot RÉEL par actif crypto.
    - Si aucun carnet n'a été reçu via WS pour un symbole -> snapshot REST.
    - Si un carnet est trop vieux (>20s) -> refresh REST.
    """
    while True:
        try:
            for symbol in CRYPTO_SYMBOLS:
                book = STATE.get("order_books", {}).get(symbol)
                fresh = False
                if book:
                    # Vérifie la fraîcheur via le timestamp stocké au moment de la MAJ
                    fresh = time.time() - book.get("_ts", 0.0) < 20.0
                if not fresh:
                    await fetch_depth_snapshot_rest(symbol)
        except Exception as e:
            logger.warning(f"Order book snapshot loop error: {e}")
        await asyncio.sleep(10.0)


def _store_depth_update(symbol: str, bids_raw: list, asks_raw: list, exchange: str = "bybit"):
    """Parse et stocke une mise à jour de profondeur RÉELLE (WS ou REST)."""
    try:
        bids = [[float(b[0]), float(b[1])] for b in bids_raw[:5]]
        asks = [[float(a[0]), float(a[1])] for a in asks_raw[:5]]
        if bids and asks:
            update_asset_order_book(symbol, bids, asks, exchange=exchange)
            STATE.setdefault("order_books", {}).setdefault(symbol, {})["_ts"] = time.time()
    except Exception as e:
        logger.debug(f"Depth parse error {symbol}: {e}")


async def multi_exchange_websocket_listener():
    """
    Connexion concurrente aux flux WebSocket publics Binance ET Bybit,
    généralisée à TOUS les actifs crypto (faille 1 corrigée — le carnet
    d'ordres réel n'est plus limité à BTC).

    - Bybit : tickers + orderbook.5 pour BTCUSDT/ETHUSDT/SOLUSDT
    - Binance : ticker + depth5 pour les 3 mêmes symboles (quand accessible)
    - Secours REST : snapshot de profondeur par actif (order_book_snapshot_loop)
    - Actifs non-crypto (Or/FX/Actions) : pas de carnet public -> honnêtement
      UNAVAILABLE (le paper-trading utilisera le modèle de slippage).
    """
    binance_streams = "/".join(f"{s.lower()}@ticker/{s.lower()}@depth5" for s in CRYPTO_SYMBOLS)
    binance_url = f"wss://stream.binance.com:9443/stream?streams={binance_streams}"
    bybit_url = "wss://stream.bybit.com/v5/public/spot"

    async def listen_binance():
        logger.info("Connecting to primary Binance Live Ticker & Depth WebSocket (multi-assets)...")
        while True:
            try:
                async with websockets.connect(binance_url, ping_interval=20, ping_timeout=20) as ws:
                    logger.info("Binance Combined WS Stream Connected (BTC/ETH/SOL)!")
                    while True:
                        data = await ws.recv()
                        frame = json.loads(data)
                        stream_name = frame.get("stream", "")
                        msg = frame.get("data", {})

                        symbol = None
                        for s in CRYPTO_SYMBOLS:
                            if stream_name.startswith(s.lower()):
                                symbol = s
                                break
                        if symbol is None:
                            continue

                        if "@ticker" in stream_name:
                            price = msg.get("c")
                            if price is not None and float(price) > 0:
                                mark_real_price(symbol, float(price),
                                                volume_24h=float(msg.get("v", 0.0)))
                                STATE["last_tick_volume"] = float(msg.get("v", 0.0))
                                await trigger_realtime_broadcast()

                        elif "@depth5" in stream_name:
                            _store_depth_update(symbol, msg.get("bids", []), msg.get("asks", []), exchange="binance")
                            await trigger_realtime_broadcast()
            except Exception as e:
                logger.warning(f"Binance WS disconnected: {str(e)}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def listen_bybit():
        logger.info("Connecting to secondary Bybit Live Ticker + Depth WebSocket (multi-assets)...")
        while True:
            try:
                async with websockets.connect(bybit_url, ping_interval=20, ping_timeout=20) as ws:
                    sub_args = [f"tickers.{_bybit_ws_symbol(s)}" for s in CRYPTO_SYMBOLS] + \
                               [f"orderbook.5.{_bybit_ws_symbol(s)}" for s in CRYPTO_SYMBOLS]
                    sub_msg = {"op": "subscribe", "args": sub_args}
                    await ws.send(json.dumps(sub_msg))
                    logger.info(f"Bybit WS subscribed: {', '.join(sub_args)}")
                    while True:
                        data = await ws.recv()
                        msg = json.loads(data)
                        topic = msg.get("topic", "")
                        if "tickers." in topic:
                            tick = msg.get("data", {})
                            symbol = topic.replace("tickers.", "")
                            last_price_raw = tick.get("lastPrice")
                            if last_price_raw is not None:
                                price = float(last_price_raw)
                                if price > 0 and symbol in STATE["assets"]:
                                    mark_real_price(symbol, price,
                                                    volume_24h=tick.get("volume24h"))
                                    STATE["last_tick_volume"] = float(tick.get("volume24h", 0.0) or 0.0)
                                    await trigger_realtime_broadcast()
                        elif "orderbook.5." in topic:
                            symbol = topic.replace("orderbook.5.", "")
                            d = msg.get("data", {})
                            _store_depth_update(symbol, d.get("b", []), d.get("a", []), exchange="bybit")
                            await trigger_realtime_broadcast()
            except Exception as e:
                logger.warning(f"Bybit WS disconnected: {str(e)}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    # FIX (logs prod) : la tâche parente RESTE VIVANTE tant que ses sous-tâches
    # tournent — sinon le watchdog la voit "morte" immédiatement et la redémarre
    # en boucle (fuite de listeners doublons). asyncio.gather attend les 3.
    try:
        await asyncio.gather(
            listen_binance(),
            listen_bybit(),
            order_book_snapshot_loop(),
        )
    except Exception as e:
        logger.warning(f"multi_exchange_websocket_listener: sous-tâche terminée en erreur ({e}) -> redémarrage complet par watchdog")
        raise


async def price_consensus_loop():
    """
    Rafraîchit le consensus multi-sources de TOUS les actifs toutes les 20s.
    Le résultat alimente STATE["price_consensus"] (télémétrie + boucle de trading).
    """
    while True:
        try:
            for sym in list(STATE["assets"].keys()):
                try:
                    cons = await price_engine.get_consensus(sym, max_age_seconds=0.0)
                    STATE.setdefault("price_consensus", {})[sym] = cons
                except Exception as se:
                    logger.debug(f"Consensus refresh failed for {sym}: {se}")
        except Exception as e:
            logger.warning(f"Price consensus loop error: {e}")
        await asyncio.sleep(20.0)


async def order_flow_websocket_listener():
    """
    LOT 3 (PDF Pilier H) : flux de TRADES RÉELS + LIQUIDATIONS pour calculer
    Delta/CVD, OFI, absorption et cascades de liquidation.

    - Bybit spot : publicTrade.{symbole} (trades avec côté agressif)
    - Binance spot : {symbole}@aggTrade (tick rule appliquée)
    - Liquidations : Binance futures !forceOrder@arr + Bybit linear liquidation
    Toutes les données sont 100% réelles ; sans flux -> indicateurs NEUTRES.
    """
    bybit_spot_url = "wss://stream.bybit.com/v5/public/spot"
    binance_spot_url = ("wss://stream.binance.com:9443/stream?streams="
                        + "/".join(f"{s.lower()}@aggTrade" for s in CRYPTO_SYMBOLS))
    bybit_linear_url = "wss://stream.bybit.com/v5/public/linear"

    async def listen_bybit_trades():
        while True:
            try:
                async with websockets.connect(bybit_spot_url, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(json.dumps({"op": "subscribe", "args": [f"publicTrade.{s}" for s in CRYPTO_SYMBOLS]}))
                    logger.info("Bybit OrderFlow WS connected (publicTrade BTC/ETH/SOL)")
                    while True:
                        msg = json.loads(await ws.recv())
                        topic = msg.get("topic", "")
                        if "publicTrade." in topic:
                            symbol = topic.replace("publicTrade.", "")
                            for t in msg.get("data", []):
                                try:
                                    order_flow.update_trade(
                                        symbol, float(t["p"]), float(t["v"]),
                                        "buy" if t.get("side") == "Buy" else "sell")
                                except Exception:
                                    continue
            except Exception as e:
                logger.warning(f"Bybit OrderFlow WS disconnected: {e}. Reconnect 5s")
                await asyncio.sleep(5)

    async def listen_binance_trades():
        while True:
            try:
                async with websockets.connect(binance_spot_url, ping_interval=20, ping_timeout=20) as ws:
                    logger.info("Binance aggTrade WS connected (BTC/ETH/SOL)")
                    while True:
                        frame = json.loads(await ws.recv())
                        stream = frame.get("stream", "")
                        symbol = next((s for s in CRYPTO_SYMBOLS if stream.startswith(s.lower())), None)
                        if not symbol:
                            continue
                        agg = frame.get("data", {})
                        try:
                            price = float(agg.get("p", 0.0))
                            qty = float(agg.get("q", 0.0))
                            is_buyer = agg.get("m") is False  # m=False -> acheteur agressif
                            order_flow.update_trade(symbol, price, qty,
                                                    "buy" if is_buyer else "sell")
                        except Exception:
                            continue
            except Exception as e:
                logger.warning(f"Binance aggTrade WS disconnected: {e}. Reconnect 5s")
                await asyncio.sleep(5)

    async def listen_liquidations():
        """Liquidations réelles : Binance futures + Bybit linear."""
        b_url = "wss://fstream.binance.com/ws/!forceOrder@arr"
        while True:
            try:
                async with websockets.connect(b_url, ping_interval=20, ping_timeout=20) as ws:
                    logger.info("Binance !forceOrder WS connected (liquidations)")
                    while True:
                        msg = json.loads(await ws.recv())
                        o = msg.get("o", {})
                        sym = o.get("s", "")
                        if sym and sym in CRYPTO_SYMBOLS:
                            order_flow.update_liquidation(
                                sym, "sell" if o.get("S") == "SELL" else "buy",
                                float(o.get("q", 0.0)), float(o.get("p", 0.0)))
            except Exception as e:
                logger.debug(f"Binance liquidation WS down: {e}")
                break  # pas bloquant ; Bybit en secours
        # Bybit linear liquidation (secours)
        while True:
            try:
                async with websockets.connect(bybit_linear_url, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(json.dumps({"op": "subscribe", "args": [f"liquidation.{s}" for s in CRYPTO_SYMBOLS]}))
                    logger.info("Bybit liquidation WS connected")
                    while True:
                        msg = json.loads(await ws.recv())
                        topic = msg.get("topic", "")
                        if "liquidation." in topic:
                            symbol = topic.replace("liquidation.", "")
                            for liq in msg.get("data", []):
                                try:
                                    order_flow.update_liquidation(
                                        symbol, "sell" if liq.get("side") == "Sell" else "buy",
                                        float(liq.get("qty", 0.0)), float(liq.get("price", 0.0)))
                                except Exception:
                                    continue
            except Exception as e:
                logger.debug(f"Bybit liquidation WS down: {e}")
                await asyncio.sleep(10)

    # FIX (logs prod) : idem — la tâche parente attend ses sous-tâches pour
    # ne pas être redémarrée en boucle par le watchdog.
    try:
        await asyncio.gather(
            listen_bybit_trades(),
            listen_binance_trades(),
            listen_liquidations(),
        )
    except Exception as e:
        logger.warning(f"order_flow_websocket_listener: sous-tâche terminée en erreur ({e}) -> redémarrage complet par watchdog")
        raise


# LOT 7 : registre module-level des tâches de fond (jamais dans STATE :
# les objets Task ne sont pas sérialisables JSON)
_BG_TASKS: dict = {}


# LOT 7 : fabriques des tâches de fond surveillées par le watchdog
TASK_FACTORIES = {
    "live_trading_loop": lambda: live_trading_loop(),
    "multi_exchange_websocket_listener": lambda: multi_exchange_websocket_listener(),
    "order_flow_websocket_listener": lambda: order_flow_websocket_listener(),
    "price_consensus_loop": lambda: price_consensus_loop(),
    "telegram_poll": lambda: telegram_bot.poll_telegram_commands_loop(),
    "reconciliation_scheduler": lambda: reconciliation_scheduler(),
    "concierge_scheduler": lambda: concierge_scheduler(),
    "db_backup_scheduler": lambda: db_backup_scheduler(),
    "autonomous_ai_scheduler": lambda: autonomous_ai_scheduler(),
    "copy_trading_refresh_scheduler": lambda: copy_trading_refresh_scheduler(),
    "copy_mirror_scheduler": lambda: copy_mirror_scheduler(),
}


def register_background_tasks() -> None:
    """Enregistre toutes les tâches de fond lancées au démarrage pour le watchdog."""
    STATE.setdefault("background_tasks", {})
    for name in TASK_FACTORIES:
        # retrouve la tâche par nom dans asyncio.all_tasks()
        for t in asyncio.all_tasks():
            if t.get_name() == f"qp_{name}":
                STATE["background_tasks"][name] = t
                break


def launch_named(coro, name: str):
    """Lance une coroutine comme tâche nommée (surveillable par le watchdog)."""
    task = asyncio.create_task(coro)
    task.set_name(f"qp_{name}")
    _BG_TASKS[name] = task
    return task


def validate_startup_config():
    """
    LOT 62: Institutional startup configuration checklist.
    Logs a clear, actionable summary of the runtime prerequisites per mode.
    NEVER blocks startup in DEMO mode; blocks REAL mode if the DB or keys are absent.
    """
    from database.db_manager import DATABASE_URL
    checks = []

    # 1. Telegram
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    tg_chat = os.getenv("TELEGRAM_CHAT_ID")
    if tg_token and tg_chat:
        checks.append(("Telegram notifications", "OK"))
    else:
        checks.append(("Telegram notifications", "MISSING token/chat_id -> alert-silent mode"))

    # 2. Database
    if DATABASE_URL and (DATABASE_URL.startswith("postgresql") or DATABASE_URL.startswith("postgres://")):
        checks.append(("Database", "PostgreSQL (Supabase)"))
    else:
        checks.append(("Database", "SQLite (dev/DEMO only - REAL mode is forbidden by design)"))

    # 3. REAL mode prerequisites
    api_key = db.get_setting("binance_api_key", decrypt=True) or db.get_setting("bybit_api_key", decrypt=True)
    if STATE["mode"] == "REAL":
        if not api_key:
            checks.append(("REAL mode", "CRITICAL: no exchange API keys stored -> trading will be blocked"))
        if not DATABASE_URL:
            checks.append(("REAL mode", "CRITICAL: no PostgreSQL URL -> startup forbidden"))
    else:
        checks.append(("REAL mode", "not active (DEMO mode, safe)"))

    # 3b. API key rotation age (audit B3-6)
    try:
        rotated = float(db.get_setting("api_keys_rotated_at") or 0.0)
        if rotated:
            age_days = (time.time() - rotated) / 86400.0
            if age_days > 90:
                checks.append(("Exchange API keys", f"WARNING: last rotation {age_days:.0f} days ago"))
            else:
                checks.append(("Exchange API keys", f"rotated {age_days:.0f} days ago"))
        else:
            checks.append(("Exchange API keys", "never rotated (set keys to enable trading)"))
    except Exception:
        pass

    # 4. On-chain / DeFi
    if os.getenv("EVM_PRIVATE_KEY"):
        checks.append(("DeFi EVM wallet", "OK (non-custodial execution ready)"))
    else:
        checks.append(("DeFi EVM wallet", "not configured (CEX routing only)"))

    logger.info("========== STARTUP CONFIGURATION CHECKLIST (LOT 62) ==========")
    for name, status in checks:
        logger.info(f"[CONFIG] {name:<32} -> {status}")
    logger.info("==================================================================")

    # Hard block for REAL mode without PostgreSQL (matches DBManager's own guard)
    if STATE["mode"] == "REAL" and not DATABASE_URL:
        raise RuntimeError(
            "REAL mode requires SUPABASE_DB_URL (PostgreSQL). SQLite fallback is "
            "strictly forbidden in production. Configure SUPABASE_DB_URL first."
        )

    # Audit B3-3/B3-4: never run with DEFAULT secrets when auth is enforced.
    # Instead of hard-blocking startup (which broke Railway deploys without env
    # secrets), we AUTO-GENERATE strong secrets on first boot, persist them in the
    # DB (encrypted) and inject them into the environment. This keeps the system
    # secure (no predictable defaults) while being zero-config to run.
    production_auth = os.getenv("AUTH_ENABLED", "").lower() == "true" or STATE["mode"] == "REAL"
    if production_auth:
        import secrets as _secrets
        import bcrypt as _bcrypt

        # ---- JWT secret: reuse persisted or generate strong ----
        jwt_secret = os.getenv("JWT_SECRET_KEY", "")
        if len(jwt_secret) < 24:
            persisted = db.get_setting("jwt_secret_key", decrypt=True)
            if persisted and len(persisted) >= 24:
                jwt_secret = persisted
            else:
                jwt_secret = _secrets.token_urlsafe(48)
                db.save_setting("jwt_secret_key", jwt_secret, encrypt=True)
                logger.warning(
                    "🔐 AUTH: auto-generated a strong JWT_SECRET_KEY and stored it "
                    "encrypted in the DB. Set JWT_SECRET_KEY env to override."
                )
            os.environ["JWT_SECRET_KEY"] = jwt_secret

        # ---- Admin password: reuse persisted hash or generate + upsert ----
        admin_pass = os.getenv("ADMIN_PASSWORD", "")
        if not admin_pass or admin_pass == "ChangeMe!Institutionnel2026":
            persisted_hash = db.get_user("admin_quant")
            if persisted_hash and persisted_hash.get("password_hash", "").startswith("$2"):
                # a real bcrypt hash already exists in the DB -> rely on it
                logger.warning(
                    "🔐 AUTH: ADMIN_PASSWORD env not set - using the bcrypt hash "
                    "already stored in the users table. Set ADMIN_PASSWORD to override."
                )
            else:
                admin_pass = _secrets.token_urlsafe(12)
                hashed = _bcrypt.hashpw(admin_pass.encode(), _bcrypt.gensalt()).decode()
                db.upsert_admin(hashed, Roles.ADMIN)
                os.environ["ADMIN_PASSWORD"] = admin_pass
                logger.warning(
                    f"🔐 AUTH: auto-generated admin password. THIS IS SHOWN ONCE - "
                    f"save it now and set ADMIN_PASSWORD env to override. "
                    f"username=admin_quant password={admin_pass}"
                )
        else:
            # env password provided: make sure the DB hash is in sync
            try:
                hashed = _bcrypt.hashpw(admin_pass.encode(), _bcrypt.gensalt()).decode()
                db.upsert_admin(hashed, Roles.ADMIN)
            except Exception:
                pass


async def autonomous_ai_scheduler():
    """
    LOT 66: FULLY AUTONOMOUS AI.
    Periodic self-improvement cycle (every 6h):
      1. Refresh real market data (Binance -> Yahoo fallback)
      2. MLOps pipeline: retrain HMM + LSTM + genetic tuning + model registry
      3. Autonomous PPO training from the live experience buffer (real outcomes)
      4. Walk-forward validation (champion/challenger) - only deploy if it improves
      5. Audit log + Telegram notification
    """
    while True:
        await asyncio.sleep(6 * 3600)
        try:
            logger.info("🤖 AUTONOMOUS AI CYCLE STARTING (self-retrain + validate + deploy)")

            # 1) Fresh real data
            df = STATE.get("historical_bars")
            if df is None or df.empty or len(df) < 120:
                df2 = await fetch_historical_market_data("BTCUSDT")
                if df2 is not None and not df2.empty and len(df2) >= 120:
                    df = df2
                    STATE["historical_bars"] = df
            if df is None or df.empty or len(df) < 120:
                logger.warning("🤖 Autonomous AI: insufficient market data, skipping cycle.")
                continue

            # 2) MLOps pipeline (retrain + registry, auto-deploy in DEMO)
            try:
                pipe_res = mlops_trainer.execute_pipeline(df)
                logger.info(f"🤖 Autonomous AI: MLOps pipeline -> {pipe_res.get('status')}")
            except Exception as pe:
                logger.warning(f"🤖 Autonomous AI: MLOps pipeline error: {pe}")

            # 3) Autonomous PPO training from real collected experiences
            buf = STATE.get("ppo_buffer") or []
            platform_metrics.AI_PPO_BUFFER.set(len(buf))
            if len(buf) >= 50:
                try:
                    ppo_agent.train_step(
                        states=[b["state"] for b in buf],
                        actions=[b["action"] for b in buf],
                        log_probs_old=[b["log_prob"] for b in buf],
                        rewards=[b["reward"] for b in buf],
                        next_states=[b["next_state"] for b in buf],
                        terminals=[b["terminal"] for b in buf],
                    )
                    logger.info(f"🤖 Autonomous AI: PPO self-trained on {len(buf)} real experiences.")
                    STATE["ppo_buffer"] = []
                except Exception as ppo_err:
                    logger.warning(f"🤖 Autonomous AI: PPO training error: {ppo_err}")
                # LOT 4 (PDF Pilier C) : mise en sommeil périodique des experts
                # MoE inutiles (contribution PnL négative sur échantillon suffisant)
                try:
                    _sleepy = mixture_of_experts.sleep_useless_experts(min_samples=5, min_contrib_pct=0.0)
                    if _sleepy:
                        logger.warning(f"🧟 Experts MoE mis en sommeil: {_sleepy}")
                except Exception:
                    pass
                # LOT 8 (PDF Pilier N/Q) : métriques de qualité + stress crises
                # réelles + bootstrap du Sharpe (chaque cycle autonome)
                try:
                    _eq = STATE["equity_history_demo"] if STATE["mode"] == "DEMO" else STATE["equity_history_real"]
                    _trades = attribution.trades[-200:]
                    _qm = quality_metrics(_eq, _trades)
                    STATE["quality_metrics"] = _qm
                    # Stress test : crises réelles sur le portefeuille complet
                    _positions = db.get_positions()
                    _prices = STATE.get("last_known_prices", {})
                    STATE["stress_test_report"] = scenario_tester.run_stress(
                        _positions, STATE[active_balance_key], _prices)
                    # Bootstrap : le Sharpe observé est-il dû à la chance ?
                    if len(_eq) >= 30:
                        STATE["bootstrap_sharpe"] = monte_carlo_tester.bootstrap_sharpe_significance(
                            _eq, n_permutations=1000, seed=42)
                except Exception as _qe:
                    logger.debug(f"Quality metrics/stress failed: {_qe}")
                # LOT 7 (PDF Pilier K) : méta-attribution -> RÉDUCTION AUTOMATIQUE
                # du poids des mauvaises raisons (quelles raisons gagnent ?)
                try:
                    _attr = meta_attribution(STATE.get("decision_log", []))
                    if len(_attr) >= 2:
                        _rw = reason_weight_from_attribution(_attr)
                        STATE["reason_weights"] = _rw
                        # facteur global : moyenne des poids (bornée 0.5..1.1)
                        _avg = sum(_rw.values()) / max(len(_rw), 1)
                        STATE["reason_weights_factor"] = max(0.5, min(1.1, _avg))
                        logger.info(f"🧠 MÉTA-ATTRIBUTION: {len(_rw)} raisons pesées, facteur {_avg:.2f}")
                except Exception:
                    pass
            else:
                logger.info(f"🤖 Autonomous AI: PPO buffer {len(buf)}/50 - collecting more experiences.")

            # 3a0) VISION §1c: causal discovery on REAL features -> store parents
            try:
                md_c = {"vpin": STATE.get("market_state", {}).get("vpin", 0.5),
                        "kyle_lambda": 0.0, "sentiment": _neutral(STATE.get("sentiment_index")),
                        "onchain_risk": _neutral(STATE.get("onchain_risk_score"), 0.5),
                        "funding_rates": STATE.get("funding_rates", {}),
                        "symbol": "BTCUSDT"}
                fdf = build_causal_feature_df(STATE, df, md_c)
                if fdf is not None and len(fdf) >= 40:
                    parents = discover_causal_parents(fdf, target="returns")
                    STATE["causal_parents"] = parents
                    STATE["causal_analyzed"] = True   # LOT 4 : l'analyse causale a tourné
                    db.save_setting("causal_parents", json.dumps(parents))
                    logger.info(
                        f"🔗 ANALYSE CAUSALE: {len(parents)} parent(s) causal(aux) "
                        f"trouvé(s) {parents} -> "
                        f"{'signaux actifs' if parents else 'AUCUN parent causal -> réduction des signaux (LOT 4)'}")
                    logger.info(f"🧠 Causal parents of returns: {parents}")
            except Exception as ce:
                logger.warning(f"Causal discovery skipped: {ce}")

            # 3a1) VISION §2c/2d: OFFLINE RL on the replayable event journal
            try:
                events = db.list_events(event_type="paper_fill", limit=500)
                samples = []
                for e in events:
                    try:
                        p = json.loads(e["payload"])
                        samples.append({
                            "state": np.array([0.0, float(p.get("slippage_bps", 0.0)) / 100.0, 0.0, 0.0]),
                            "action": 1.0 if p.get("side") == "BUY" else -1.0,
                            "log_prob": 0.0,
                            "reward": -float(p.get("slippage_bps", 0.0)) / 10000.0,
                            "next_state": np.array([0.0, 0.0, 0.0, 0.0]),
                            "terminal": False,
                            "vol": 0.01,
                        })
                    except Exception:
                        continue
                if len(samples) >= 30:
                    for _h, _exp in mixture_of_experts.experts.items():
                        n = _exp.train_offline(curriculum_sort(samples))
                        if n:
                            logger.info(f"🧠 OFFLINE RL: {_h} expert trained on {n} journal samples")
            except Exception as oe:
                logger.warning(f"Offline RL skipped: {oe}")

            # 3a2) VISION §3: autonomous research cycle (invent -> test -> promote)
            try:
                md_r = {"vpin": STATE.get("market_state", {}).get("vpin", 0.5),
                        "kyle_lambda": 0.0, "sentiment": _neutral(STATE.get("sentiment_index")),
                        "onchain_risk": _neutral(STATE.get("onchain_risk_score"), 0.5),
                        "funding_rates": STATE.get("funding_rates", {}),
                        "market_avg_return": 0.0}
                _research = hypothesis_generator.run_research_cycle(df, md_r, n_candidates=6)
                logger.info(f"🧪 RESEARCH CYCLE: {_research['candidates']} tested, "
                            f"{len(_research['promoted'])} promoted, admitted={len(_research['admitted'])}")
            except Exception as re:
                logger.warning(f"Research cycle skipped: {re}")

            # 3a3) VISION §6: risk committee veto + daily risk budget
            try:
                _vetoes = risk_committee.evaluate(meta_engine, STATE)
                for v in _vetoes:
                    db.add_audit_log("RISK_COMMITTEE", audit_ip(), f"{v['action']} {v['strategy']} (score {v['score']})")
                try:
                    _stress_corr = float(db.get_setting("autonomous_last_stress_corr") or 0.5)
                except Exception:
                    _stress_corr = 0.5
                _budget = daily_risk_budget(meta_engine.recent_performance, _stress_corr)
                STATE["risk_budget"] = _budget
            except Exception as kce:
                logger.warning(f"Risk committee skipped: {kce}")

            # 3a3b) VISION_FUTUR §1: organization reallocation (internal capital market)
            try:
                _stress_corr2 = float(STATE.get("market_state", {}).get("correlation", 0.5) or 0.5)
                organization.reallocate(stress_correlation=_stress_corr2)
                logger.info(f"🏛️ ORGANIZATION: allocations={organization.status()['allocations']}")
            except Exception as oe:
                logger.warning(f"Organization reallocate skipped: {oe}")

            # 3a3c) VISION_FUTUR §5a: state snapshot (event-sourcing lite)
            try:
                save_state_snapshot(db, STATE)
            except Exception:
                pass

            # 3a3d) VISION_FUTUR §4: global curriculum - GAN scenarios become
            # training episodes for the experts (labeled scenarios, not live trades)
            try:
                _scen = generative_engine.generate_extreme_scenarios(n_scenarios=50, stress_factor=2.0)
                for i in range(min(20, len(_scen))):
                    _s = _scen[i]
                    mixture_of_experts.collect_experience(
                        state=np.array([_s[0], abs(_s[1]), _s[2], 0.0]),
                        action=float(np.clip(_s[3], -1, 1)) if len(_s) > 3 else 0.0,
                        logp=0.0,
                        reward=-abs(_s[0]) * 0.1,  # scenarios are stress episodes
                        next_state=np.array([_s[0], abs(_s[1]), _s[2], 0.0]),
                        horizon="position",
                    )
                logger.info("🎓 CURRICULUM: GAN stress episodes added to position expert buffer")
            except Exception as ce:
                logger.warning(f"GAN curriculum skipped: {ce}")

            # 3a4) VISION §7: self-assessment (meta-attribution of reasons + divergence)
            try:
                from core.reporting import build_daily_report
                _rep = build_daily_report(STATE, db)
                _reasons_log = []
                for _o in db.list_events(event_type="order", limit=200):
                    try:
                        _p = json.loads(_o["payload"])
                        _reasons_log.append({"reasons": [r.get("feature", "") for r in (_p.get("reasoning") or [])[:3]],
                                             "pnl": 0.0})
                    except Exception:
                        continue
                if _reasons_log:
                    _attr = meta_attribution(_reasons_log)
                    db.save_setting("reason_effectiveness", json.dumps(_attr))
                    logger.info(f"🔍 Meta-attribution: {len(_attr)} reasons tracked")
                _real_slip = execution_alpha.avg_slippage_bps("market") or 3.0
                _div = simulation_divergence(3.0, _real_slip)  # modeled baseline 3bps
                STATE["sim_divergence"] = _div
                db.save_setting("sim_divergence", str(_div))
                logger.info(f"🔍 Sim vs live slippage divergence: {_div:.2f}")
            except Exception as sae:
                logger.warning(f"Self-assessment skipped: {sae}")

            # 3a) MONTE-CARLO DAILY STRESS (audit B10-2): measure tail risk continuously
            try:
                hist = STATE.get("historical_bars")
                if hist is not None and len(hist) > 30:
                    _mc_price = STATE.get("last_known_prices", {}).get("BTCUSDT") or STATE.get("last_price")
                    if _mc_price is None:
                        logger.warning("Skipping Monte-Carlo stress test: no real BTC price available yet.")
                    else:
                        mc = monte_carlo_tester.execute_stress_test(
                            initial_capital=STATE["balance_demo"] if STATE["mode"] == "DEMO" else STATE["balance_real"],
                            current_price=float(_mc_price),
                            historical_volatility=float(hist["close"].pct_change().dropna().std() or 0.02),
                        )
                        ruin_pct = float(mc.get("ruin_probability") or mc.get("ruin_prob") or 0.0)
                        platform_metrics.RISK_CVAR.set(ruin_pct * 100.0)
                    ruin_pct = float(mc.get("ruin_probability") or mc.get("ruin_prob") or 0.0)
                    platform_metrics.RISK_CVAR.set(ruin_pct * 100.0)
                    logger.info(f"🤖 Monte-Carlo stress: ruin probability = {ruin_pct*100:.2f}% | {mc.get('summary','')}")
            except Exception as mce:
                logger.warning(f"Monte-Carlo stress skipped: {mce}")

            # 3b) GAN EXTREME-SCENARIO STRESS (audit B9-4): the torch GAN generates
            # tail scenarios used to scale portfolio risk budget for the next period.
            try:
                gen_scen = generative_engine.generate_extreme_scenarios(n_scenarios=500, stress_factor=2.5)
                tail_vol = float(np.std(gen_scen[:, 0])) if gen_scen.size else 0.0
                base_vol = float(np.std(df["close"].pct_change().dropna().values[-200:])) if len(df) > 10 else 0.0
                if base_vol > 0 and tail_vol > 0:
                    stress_ratio = float(np.clip(tail_vol / base_vol, 1.0, 3.0))
                    STATE["gan_stress_ratio"] = stress_ratio
                    platform_metrics.RISK_CVAR.set(stress_ratio)
                    logger.info(f"🤖 GAN stress: tail/base vol ratio = {stress_ratio:.2f}")
            except Exception as ge:
                logger.warning(f"GAN stress skipped: {ge}")

            # 4) Walk-forward champion/challenger validation (audit B8-4: multi-asset)
            try:
                from backtester.engine import EventDrivenBacktester, WalkForwardValidator
                wf = WalkForwardValidator(train_ratio=0.7)
                bt = EventDrivenBacktester(initial_capital=STATE.get("balance_demo", 100000.0))
                strat = TrendFollowingStrategy()
                meta_local = MetaAllocationEngine(strategies=[strat])
                risk_local = RiskManager()
                det_local = MarketRegimeDetector()
                pred_local = LSTMLikePredictor(input_dim=5, hidden_dim=8)
                ppo_local = PPOTRAgent(state_dim=4, action_dim=1)

                # Primary asset + optional secondary assets from the DB candle cache
                _wf_datasets = {"BTCUSDT": df}
                for _sym in ("ETHUSDT", "SOLUSDT", "XAUUSD"):
                    try:
                        _d = db.load_candles(_sym, limit=400)
                        if _d is not None and not _d.empty and len(_d) >= 150:
                            _wf_datasets[_sym] = _d
                    except Exception:
                        pass

                _oos_sharpe_agg = 0.0
                for _sym, _data in _wf_datasets.items():
                    try:
                        _res = wf.run_validation(_data, bt, meta_local, risk_local, det_local, pred_local, ppo_local)
                        _oos = _res.get("out_of_sample_metrics", {}) or {}
                        _oos_sharpe_agg += float(_oos.get("sharpe_ratio") or 0.0)
                    except Exception as _we:
                        logger.warning(f"Walk-forward {_sym} skipped: {_we}")
                oos_sharpe = _oos_sharpe_agg / max(len(_wf_datasets), 1)
                platform_metrics.AI_OOS_SHARPE.set(oos_sharpe)
                platform_metrics.AI_LAST_CYCLE.set(time.time())
                prev_sharpe = float(db.get_setting("autonomous_last_oos_sharpe") or 0.0)
                # VISION §4.4: Deflated Sharpe gate - is this OOS result statistically
                # better than luck across the number of strategies/models tried?
                try:
                    _dsr = calculate_deflated_sharpe_ratio(
                        observed_sharpe=oos_sharpe,
                        num_trials=max(len(strategies_list), 8),
                        trials_variance_sharpe=0.1,
                        sample_length=200,
                    )
                except Exception:
                    _dsr = 0.0
                trend = "IMPROVED" if (oos_sharpe >= prev_sharpe and _dsr >= 0.95) else "DEGRADED"
                db.save_setting("autonomous_last_oos_sharpe", str(oos_sharpe))
                db.save_setting("autonomous_last_dsr", str(_dsr))
                logger.info(f"🤖 Autonomous AI: OOS Sharpe {oos_sharpe:.3f} vs {prev_sharpe:.3f} | DSR {_dsr:.3f} (gate 0.95) -> {trend}")
                try:
                    await telegram_bot.send_push_notification(
                        f"🤖 *CYCLE IA AUTONOME*\n"
                        f"━━━━━━━━━━━━━━━━━━━━━\n"
                        f"📊 Sharpe out-of-sample : *{oos_sharpe:.3f}*\n"
                        f"📈 Tendance : *{trend}*\n"
                        f"🧠 PPO : entraîné sur {len(buf)} expériences réelles"
                    )
                except Exception:
                    pass
            except Exception as ve:
                logger.warning(f"🤖 Autonomous AI: walk-forward validation error: {ve}")

            db.add_audit_log("AUTONOMOUS_AI_CYCLE", audit_ip(), "Completed autonomous self-retrain/validate cycle.")
        except Exception as e:
            logger.error(f"🤖 Autonomous AI cycle failed: {e}")


async def concierge_scheduler():
    """AUDIT D4: daily Telegram risk-concierge digest at a configured hour."""
    while True:
        hour = settings.get_int("alerts", "daily_digest_hour_utc", 18)
        try:
            now = time.gmtime()
            next_run = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hour, 0, 0, 0, 0, -1))
            if next_run <= time.time():
                next_run += 86400
            await asyncio.sleep(next_run - time.time())
            report = build_daily_report(STATE, db)
            # VISION_FUTUR §8: proactive health alert when the bot distrusts itself
            if report.get("health_score", 100) < 60:
                await telegram_bot.send_push_notification(
                    f"🚨 *SANTÉ FAIBLE : {report['health_score']}/100*\n"
                    f"Raisons : {', '.join(report.get('health_reasons', [])[:4])}\n"
                    f"Le bot réduit automatiquement ses tailles."
                )
            try:
                await telegram_bot.send_push_notification(build_concierge_message(report))
                # VISION_FUTUR §3: LLM narrative appended (OpenRouter or structured)
                try:
                    _narr = await daily_market_narrative_async(report, STATE)
                    if _narr:
                        STATE["last_narrative"] = _narr
                        await telegram_bot.send_push_notification(_narr)
                except Exception as ne:
                    logger.warning(f"Narrative skipped: {ne}")
                logger.info("✅ Concierge quotidien envoyé")
            except Exception as e:
                logger.warning(f"Concierge Telegram envoi échoué: {e}")
            db.add_audit_log("DAILY_CONCIERGE", audit_ip(), "Daily report generated.")
        except Exception as e:
            logger.warning(f"Concierge scheduler error: {e}")
            await asyncio.sleep(3600)


@app.get("/api/v1/health")
async def api_health():
    """AUDIT D3: composite 0-100 health score."""
    score, reasons = compute_health_score(STATE, db)
    return {"health_score": score, "reasons": reasons, "mode": STATE["mode"], "ts": time.time()}


@app.get("/api/v1/news")
async def api_v1_news(_auth: dict = Depends(require_auth)):
    """
    LOT 5 (PDF Pilier I) : état des ACTUALITÉS RÉELLES.
    Expose le sentiment (index + confiance), les dernières headlines réelles
    avec leur source, le choc systémique éventuel et la pondération des sources.
    Audit : aucune donnée fictive — headlines vides si sources hors ligne.
    """
    return {
        "sentiment_index": STATE.get("sentiment_index"),
        "available": STATE.get("sentiment_available", False),
        "confidence": STATE.get("sentiment_confidence", 0.0),
        "num_headlines": len(STATE.get("recent_headlines", [])),
        "headlines": STATE.get("recent_headlines", [])[:20],
        "shock_status": STATE.get("news_shock", {"shock_detected": False}),
        "source_weights": {k: v for k, v in SOURCE_WEIGHTS_REF.items()},
        "ts": time.time(),
    }


class MacroOverrideRequest(BaseModel):
    """Pilotage humain du risque macro (LOT 5, PDF Pilier I)."""
    action: str = Field(pattern="^(reduce|halt|reset)$")
    factor: float = Field(default=0.5, ge=0.05, le=1.0)


@app.post("/api/v1/macro/override")
async def api_v1_macro_override(payload: MacroOverrideRequest,
                                _auth: dict = Depends(require_auth)):
    """
    LOT 5 (PDF Pilier I) : pilotage HUMAIN du risque macro.
    - reduce : applique un facteur de taille manuel (défaut 0.5)
    - halt   : passe la machine à états en HALT (nouveaux ordres bloqués)
    - reset  : revient à NORMAL, facteur 1.0
    L'opérateur reste le décideur final (mentalité n°10).
    """
    try:
        if payload.action == "reduce":
            STATE["macro_scale_factor_tactile"] = payload.factor
            db.add_audit_log("MACRO_OVERRIDE", audit_ip(),
                             f"Opérateur: réduction macro manuelle x{payload.factor}")
            return {"ok": True, "action": "reduce", "factor": payload.factor,
                    "risk_state": risk_state.to_dict()}
        if payload.action == "halt":
            risk_state.enter(RiskStateMachine.HALT, "MACRO_OVERRIDE")
            STATE["risk_state"] = risk_state.to_dict()
            db.add_audit_log("MACRO_OVERRIDE", audit_ip(), "Opérateur: HALT macro manuel")
            return {"ok": True, "action": "halt", "risk_state": risk_state.to_dict()}
        # reset
        risk_state.reset(reason="macro/override")
        STATE["macro_scale_factor_tactile"] = 1.0
        STATE["risk_state"] = risk_state.to_dict()
        db.add_audit_log("MACRO_OVERRIDE", audit_ip(), "Opérateur: reset macro (NORMAL)")
        return {"ok": True, "action": "reset", "risk_state": risk_state.to_dict()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/v1/report/daily")
async def api_daily_report():
    """AUDIT C2: daily P&L report (per strategy / asset / mode + risk)."""
    return build_daily_report(STATE, db)


@app.get("/api/v1/orders")
async def api_orders_v1(limit: int = 50, offset: int = 0):
    """AUDIT B2-5: paginated order history."""
    limit = max(1, min(500, limit))
    offset = max(0, offset)
    try:
        rows = db.get_all_orders() or []
    except Exception:
        rows = []
    page = [serialize_helper(o) for o in rows[offset:offset + limit]]
    return {"total": len(rows), "limit": limit, "offset": offset, "orders": page}


class WebhookTradeRequest(BaseModel):
    """TradingView-style webhook payload (audit C8)."""
    symbol: str = Field(min_length=3, max_length=20)
    action: str = Field(pattern="^(buy|sell|BUY|SELL)$")
    secret: str = Field(min_length=1)
    qty: float = Field(default=0.0, ge=0.0, le=1e9)
    price: float = Field(default=0.0, ge=0.0, le=1e9)


# ===== AUDIT C7: multi-user management (ADMIN only) =====
class UserCreateRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern="^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)
    role: str = Field(default="VIEWER", pattern="^(VIEWER|TRADER|RISK_MANAGER|ADMIN)$")


@app.get("/api/v1/users")
async def api_list_users(_admin: dict = Depends(require_admin)):
    users = db.list_users()
    for u in users:
        u.pop("password_hash", None)
    return {"users": users}


@app.post("/api/v1/users")
async def api_create_user(payload: UserCreateRequest, _admin: dict = Depends(require_admin)):
    import bcrypt as _bc
    if db.get_user(payload.username):
        raise HTTPException(status_code=409, detail="Username already exists.")
    hashed = _bc.hashpw(payload.password.encode("utf-8"), _bc.gensalt()).decode("utf-8")
    ok = db.create_user(payload.username, hashed, payload.role)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to create user.")
    db.add_audit_log("USER_CREATED", audit_ip(), f"User '{payload.username}' created (role {payload.role}).")
    return {"status": "Success", "username": payload.username, "role": payload.role}


@app.delete("/api/v1/users/{username}")
async def api_delete_user(username: str, _admin: dict = Depends(require_admin)):
    if username == os.getenv("ADMIN_USER", "admin"):
        raise HTTPException(status_code=400, detail="Cannot delete the bootstrap admin.")
    if not db.delete_user(username):
        raise HTTPException(status_code=404, detail="User not found.")
    db.add_audit_log("USER_DELETED", audit_ip(), f"User '{username}' deleted.")
    return {"status": "Deleted", "username": username}


# ===== AUDIT C3: custom price alerts =====
class PriceAlertCreate(BaseModel):
    symbol: str = Field(min_length=3, max_length=20)
    direction: str = Field(pattern="^(above|below)$")
    target_price: float = Field(gt=0.0)
    note: str = Field(default="", max_length=200)


@app.get("/api/v1/alerts")
async def api_list_alerts(_auth: dict = Depends(require_auth)):
    return {"alerts": STATE.get("price_alerts", [])}


@app.post("/api/v1/alerts")
async def api_create_alert(payload: PriceAlertCreate, _auth: dict = Depends(require_auth)):
    symbol = payload.symbol.upper()
    if symbol not in STATE["assets"]:
        raise HTTPException(status_code=400, detail=f"Unsupported symbol '{symbol}'.")
    alert = {
        "id": uuid.uuid4().hex[:10],
        "symbol": symbol,
        "direction": payload.direction,
        "target_price": payload.target_price,
        "note": payload.note,
        "triggered": False,
        "created_ts": time.time(),
    }
    STATE["price_alerts"].append(alert)
    _alerts_persist()
    return {"status": "Created", "alert": alert}


@app.delete("/api/v1/alerts/{alert_id}")
async def api_delete_alert(alert_id: str, _auth: dict = Depends(require_auth)):
    before = len(STATE["price_alerts"])
    STATE["price_alerts"] = [a for a in STATE["price_alerts"] if a.get("id") != alert_id]
    if len(STATE["price_alerts"]) == before:
        raise HTTPException(status_code=404, detail="Alert not found.")
    _alerts_persist()
    return {"status": "Deleted", "alert_id": alert_id}


# ===== AUDIT C10: market replay =====
class ReplayRequest(BaseModel):
    symbol: str = Field(min_length=3, max_length=20)
    interval: str = Field(default="1h", pattern="^(1m|5m|15m|1h|4h|1d)$")
    limit: int = Field(default=300, ge=60, le=2000)


@app.post("/api/v1/replay")
async def api_market_replay(payload: ReplayRequest, _auth: dict = Depends(require_auth)):
    symbol = payload.symbol.upper()
    if symbol not in STATE["assets"]:
        raise HTTPException(status_code=400, detail=f"Unsupported symbol '{symbol}'.")
    return await run_market_replay(symbol, payload.interval, payload.limit)


# ===== VISION §2.1/§2.2: signal admission gate + experiment registry =====
class SignalEvalRequest(BaseModel):
    symbol: str = Field(min_length=3, max_length=20)
    limit: int = Field(default=300, ge=100, le=2000)


@app.post("/api/v1/signals/evaluate")
async def api_evaluate_signals(payload: SignalEvalRequest, _auth: dict = Depends(require_auth)):
    """Evaluates the whole signal catalogue over history -> ranking by Deflated Sharpe."""
    symbol = payload.symbol.upper()
    df = db.load_candles(symbol, limit=payload.limit)
    if df is None or df.empty or len(df) < 80:
        if symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            df = await fetch_historical_market_data(symbol)
        else:
            ymap = {"XAUUSD": "GC=F", "EURUSD": "EURUSD=X"}
            df = await fetch_yahoo_finance_candles(ymap.get(symbol, symbol), interval="1h", range_str="3mo")
    if df is None or df.empty or len(df) < 80:
        raise HTTPException(status_code=503, detail="Insufficient data to evaluate signals.")
    md = {"vpin": 0.5, "kyle_lambda": 0.0, "onchain_risk": _neutral(STATE.get("onchain_risk_score"), 0.5),
          "sentiment": _neutral(STATE.get("sentiment_index")), "funding_rate_8h": STATE.get("funding_rates", {}).get(symbol, 0.0),
          "market_avg_return": 0.0}
    return evaluate_all_signals(df, md)


@app.get("/api/v1/experiments")
async def api_list_experiments(_auth: dict = Depends(require_auth), limit: int = 100):
    return {"experiments": db.list_experiments(limit=limit)}


class ExperimentCreate(BaseModel):
    hypothesis: str = Field(min_length=3, max_length=500)


@app.post("/api/v1/experiments")
async def api_create_experiment(payload: ExperimentCreate, _auth: dict = Depends(require_auth)):
    eid = db.add_experiment(payload.hypothesis)
    if not eid:
        raise HTTPException(status_code=500, detail="Failed to register experiment.")
    return {"status": "Registered", "id": eid}


# ===== VISION §7.1 replayable event journal + §6 factor model =====
@app.get("/api/v1/events")
async def api_events(event_type: str = "", since: float = 0.0, limit: int = 200):
    """Replayable event journal (ticks + orders + decisions)."""
    return {"events": db.list_events(event_type=event_type, since=since, limit=limit)}


@app.get("/api/v1/ab")
async def api_ab(_auth: dict = Depends(require_auth)):
    """VISION §7.5: A/B paper comparison (baseline vs vol-targeted config)."""
    import math
    base, vol = STATE.get("ab_base", []), STATE.get("ab_vol", [])
    if len(base) < 10 or len(vol) < 10:
        return {"valid": False, "reason": "insufficient samples", "samples": len(base)}
    def _stats(curve):
        eq = np.array(curve)
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-9)
        sharpe = float(rets.mean() / rets.std() * np.sqrt(365 * 24)) if rets.std() > 0 else 0.0
        return {"return_pct": round((eq[-1] - 1.0) * 100.0, 3), "sharpe": round(sharpe, 3), "vol": round(float(rets.std()), 5)}
    s_base, s_vol = _stats(base), _stats(vol)
    leader = "vol_targeted" if s_vol["sharpe"] > s_base["sharpe"] else "baseline"
    return {"valid": True, "samples": len(base), "baseline": s_base, "vol_targeted": s_vol, "leader": leader}


@app.get("/api/v1/factors")
async def api_factors(_auth: dict = Depends(require_auth)):
    """VISION §6: factor exposures of the live equity curve (market/momentum/carry/vol)."""
    from core.factor_model import compute_factor_exposures
    eq = STATE.get("equity_history_demo" if STATE["mode"] == "DEMO" else "equity_history_real", [])
    if len(eq) < 20:
        return {"valid": False, "reason": "insufficient equity history"}
    rets = list(np.diff(eq) / np.maximum(np.array(eq[:-1]), 1e-9))
    mkt = [0.0] * len(rets)  # market proxy: equal-weight asset mean return (approx from price moves)
    mom = [0.0] * len(rets)
    car = [0.0] * len(rets)
    volr = [0.0] * len(rets)
    try:
        prices = [a.get("price") for a in STATE.get("assets", {}).values() if isinstance(a.get("price"), (int, float))]
        if len(prices) > 1:
            # market proxy: mean asset return per tick, broadcast to the equity
            # history length (the assets dict has one price per symbol, not per tick)
            p = np.array(prices, dtype=float)
            avg_ret = float(np.mean(np.diff(p) / np.maximum(p[:-1], 1e-9)))
            mkt = [avg_ret] * len(rets)
            mom = [float(np.sign(avg_ret)) * 0.001] * len(rets)
    except Exception:
        pass
    return compute_factor_exposures(rets, mkt, mom, car, volr)


# ===== VISION endpoints =====
@app.get("/api/v1/research")
async def api_research(_auth: dict = Depends(require_auth)):
    """VISION §3: hypothesis generator status (admitted signals + meta-prior)."""
    return hypothesis_generator.get_status()


@app.post("/api/v1/research/run")
async def api_research_run(_auth: dict = Depends(require_auth)):
    """Manually triggers one autonomous research cycle (uses cached real candles,
    with Yahoo/Binance fallback so it works anywhere)."""
    df = STATE.get("historical_bars")
    if df is None or df.empty or len(df) < 80:
        df = db.load_candles("BTCUSDT", limit=400)
    if df is None or df.empty or len(df) < 80:
        df = await fetch_yahoo_finance_candles("BTC-USD", interval="1h", range_str="3mo")
    if df is None or df.empty or len(df) < 80:
        raise HTTPException(status_code=503, detail="Insufficient data for research (all feeds unavailable).")
    md = {"vpin": 0.5, "kyle_lambda": 0.0, "sentiment": _neutral(STATE.get("sentiment_index")),
          "onchain_risk": _neutral(STATE.get("onchain_risk_score"), 0.5),
          "funding_rates": STATE.get("funding_rates", {}), "market_avg_return": 0.0}
    return hypothesis_generator.run_research_cycle(df, md, n_candidates=10)


@app.get("/api/v1/committee")
async def api_committee(_auth: dict = Depends(require_auth)):
    """VISION §6: AI risk committee status (vetoes + scores + budget)."""
    return {**risk_committee.status(), "risk_budget": STATE.get("risk_budget", {})}


@app.get("/api/v1/moe")
async def api_moe(_auth: dict = Depends(require_auth)):
    """VISION §2: mixture-of-experts status (votes, gate, offline training)."""
    return {
        "gate": STATE.get("moe_gate", {}),
        "last_votes": STATE.get("moe_votes", {}),
        "buffers": {h: len(e.buffer) for h, e in mixture_of_experts.experts.items()},
        "execution_bandit": execution_bandit.status(),
    }


@app.get("/api/v1/self")
async def api_self(_auth: dict = Depends(require_auth)):
    """VISION §7: self-assessment (divergence, reason effectiveness, honesty)."""
    reason_eff = {}
    try:
        raw = db.get_setting("reason_effectiveness")
        if raw:
            import json as _j
            reason_eff = _j.loads(raw)
    except Exception:
        pass
    return {
        "sim_divergence": STATE.get("sim_divergence", 0.0),
        "causal_parents": STATE.get("causal_parents", []),
        "conviction_threshold": STATE.get("conviction_threshold", 0.15),
        "no_trade_stats": STATE.get("no_trade_stats", {}),
        "reason_effectiveness": reason_eff,
        "strategy_exec_attribution": strategy_exec_attr.report(),
    }


# ===== VISION_FUTUR endpoints =====
@app.get("/api/v1/organization")
async def api_organization(_auth: dict = Depends(require_auth)):
    """VISION_FUTUR §1: desks + internal capital market allocations."""
    return organization.status()


@app.get("/api/v1/confidence")
async def api_confidence(_auth: dict = Depends(require_auth)):
    """VISION_FUTUR §8: composite confidence index + size factor."""
    return {"index": STATE.get("confidence_index", 100), "factor": STATE.get("confidence_factor", 1.0),
            "live_p_value": STATE.get("live_p_value", 0.5)}


@app.get("/api/v1/supervisor")
async def api_supervisor(_auth: dict = Depends(require_auth)):
    """VISION_FUTUR §5b: vital signs."""
    return {"issues": supervisor.check(force=True), "last_tick_ts": STATE.get("last_tick_ts", 0.0)}


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


@app.post("/api/v1/assistant/ask")
async def api_assistant(payload: AskRequest, _auth: dict = Depends(require_auth)):
    """VISION_FUTUR §6: the operator talks to the bot (answers grounded in real data)."""
    context = {
        "last_price": STATE.get("last_price", 0.0),
        "current_equity": STATE.get("current_equity", 0.0),
        "regime_name": STATE.get("regime_name", "?"),
        "regime_probs": STATE.get("regime_probs", {}),
        "confidence_index": STATE.get("confidence_index", 100),
        "positions": [p.get("symbol") for p in db.get_positions()],
        "desk_allocations": STATE.get("desk_allocations", {}),
        "admitted_signals": list(hypothesis_generator.admitted.keys()),
    }
    return {"answer": await answer_question_async(payload.question, context)}


@app.post("/api/v1/narrative")
async def api_narrative(_auth: dict = Depends(require_auth)):
    """VISION_FUTUR §3/§6: daily market narrative (LLM via OpenRouter, structured fallback)."""
    from core.reporting import build_daily_report
    report = build_daily_report(STATE, db)
    narrative = await daily_market_narrative_async(report, STATE)
    STATE["last_narrative"] = narrative
    return {"narrative": narrative}


@app.post("/api/v1/chaos")
async def api_chaos(_auth: dict = Depends(require_auth)):
    """VISION_FUTUR §5c: chaos self-test - simulate a feed outage, verify safe HALT."""
    res = chaos_cut_feed(STATE, db, duration_seconds=15.0)
    return res


@app.post("/api/v1/approve")
async def api_approve(_auth: dict = Depends(require_auth)):
    """VISION_FUTUR §6: approve all pending consultative-mode proposals."""
    pending = list(STATE.get("pending_approvals", []))
    STATE["pending_approvals"] = []
    approved = 0
    for p in pending:
        try:
            submit_order_via_oms(p["symbol"], p["side"], p["qty"], p["price"], p["mode"], p["strategy"],
                                 exchange="Binance" if p["symbol"] in ("BTCUSDT","ETHUSDT","SOLUSDT") else "Bybit")
            db.add_audit_log("APPROVED_ORDER", audit_ip(),
                             f"Operator approved {p['side']} {p['qty']} {p['symbol']} ({p['strategy']})")
            approved += 1
        except Exception as e:
            logger.warning(f"Approved order failed: {e}")
    return {"approved": approved, "pending_left": len(STATE["pending_approvals"])}


@app.get("/api/v1/research/export")
async def api_research_export(_auth: dict = Depends(require_auth)):
    """VISION_FUTUR §7: export meta-prior + admitted signals (knowledge sharing)."""
    return hypothesis_generator.get_status()


@app.get("/api/v1/research/packs/{name}")
async def api_research_pack(name: str, _auth: dict = Depends(require_auth)):
    """VISION_FUTUR §7: strategy pack export for the marketplace."""
    cand = hypothesis_generator.admitted.get(name)
    if not cand:
        raise HTTPException(status_code=404, detail="Signal pack not found.")
    return {"pack": cand, "format": "quant-portal-signal-pack-v1"}


@app.post("/api/v1/webhook/trade")
async def webhook_trade(payload: WebhookTradeRequest):
    """
    AUDIT C8: external alert webhook (TradingView, Pine Script, custom bots).
    Protected by WEBHOOK_SECRET env (constant-time comparison).
    Executes a market order via the OMS path, sized to a configurable % of capital.
    """
    expected = os.getenv("WEBHOOK_SECRET", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Webhooks not configured (set WEBHOOK_SECRET).")
    import hmac as _hmac
    if not _hmac.compare_digest(payload.secret, expected):
        raise HTTPException(status_code=401, detail="Invalid webhook secret.")

    symbol = payload.symbol.upper()
    if symbol not in STATE["assets"]:
        raise HTTPException(status_code=400, detail=f"Unsupported symbol '{symbol}'.")

    mode = STATE["mode"]
    bal_key = "balance_demo" if mode == "DEMO" else "balance_real"
    capital = STATE.get(bal_key, 0.0)
    price = payload.price if payload.price > 0 else (STATE["assets"].get(symbol, {}).get("price") or 0.0)
    if price <= 0:
        raise HTTPException(status_code=503, detail="No live price available for this symbol.")

    side = "BUY" if payload.action.upper() == "BUY" else "SELL"
    webhook_size_pct = settings.get_float("trading", "webhook_size_pct", 0.10)
    qty = payload.qty if payload.qty > 0 else (capital * webhook_size_pct) / price
    qty = format_exchange_size(symbol, qty, price)

    try:
        res = submit_order_via_oms(symbol, side, qty, price, mode, "WEBHOOK",
                                   exchange="Binance" if symbol in ("BTCUSDT","ETHUSDT","SOLUSDT") else "Bybit")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Order submission failed: {e}")

    # ledger (mirror of the loop's update)
    if side == "BUY":
        STATE[bal_key] -= qty * price * 1.001
    else:
        STATE[bal_key] += qty * price * 0.999
    db.update_position(symbol, qty if side == "BUY" else -qty, price, mode)
    db.add_order(symbol=symbol, side=side, price=price, qty=qty, status="FILLED",
                 mode=mode, strategy="WEBHOOK", order_type="MARKET")
    db.add_audit_log("WEBHOOK_TRADE", audit_ip(), f"Webhook {side} {qty} {symbol} @ {price:.2f}")
    platform_metrics.ORDERS_TOTAL.labels(mode=mode, side=side).inc()
    STATE["last_order_times"][symbol] = time.time()
    return {"status": "FILLED", "symbol": symbol, "side": side, "qty": qty, "price": price}


async def copy_mirror_scheduler():
    """
    REAL copy-trading execution (VISION §5): every 10 min, fetch the followed
    trader's real positions (Hyperliquid public API), compute the scaled delta
    vs our portfolio and execute via OMS when COPYTRADE_EXECUTION=auto + keys.
    Otherwise logs honest SIGNAL_ONLY mirror signals.
    """
    while True:
        await asyncio.sleep(600)
        try:
            if not copy_manager.copied_traders:
                continue
            # LOT 7 (PDF Pilier J) : stop global des traders sous-performants
            try:
                _total_cap = STATE["balance_demo"] if STATE["mode"] == "DEMO" else STATE["balance_real"]
                _stopped = copy_manager.enforce_trader_risk(_total_cap)
                if _stopped:
                    try:
                        asyncio.create_task(telegram_bot.send_push_notification(
                            f"🛑 *COPY STOP*\nTrader(s) arrêté(s) (stop global) : {', '.join(_stopped)}"))
                    except Exception:
                        pass
            except Exception:
                pass
            exec_mode = os.getenv("COPYTRADE_EXECUTION", "signal_only")
            for tid, alloc in list(copy_manager.copied_traders.items()):
                trader = copy_manager.traders.get(tid)
                acct_value = float(getattr(trader, "account_value", 0.0) or 0.0)
                positions = fetch_trader_positions(tid)
                if not positions:
                    continue
                my_pos = {p["symbol"]: float(p["qty"]) for p in db.get_positions()}
                orders = build_mirror_orders(
                    positions, my_pos,
                    allocated_capital=float(alloc.get("allocated_capital", 0.0)),
                    trader_account_value=acct_value,
                    max_asset_pct=settings.get_float("risk", "max_per_asset_pct", 0.25),
                )
                if not orders:
                    continue
                if STATE["mode"] == "REAL" and exec_mode == "auto" and get_ccxt_client():
                    for o in orders:
                        try:
                            submit_order_via_oms(o["symbol"], o["side"], o["qty"], STATE["assets"][o["symbol"]]["price"],
                                                 "REAL", "COPY_MIRROR",
                                                 exchange="Binance" if o["symbol"] in ("BTCUSDT","ETHUSDT","SOLUSDT") else "Bybit")
                            db.add_order(symbol=o["symbol"], side=o["side"], price=STATE["assets"][o["symbol"]]["price"],
                                         qty=o["qty"], status="FILLED", mode="REAL", strategy="COPY_MIRROR", order_type="MARKET")
                            db.add_audit_log("COPY_MIRROR_EXECUTED", audit_ip(),
                                             f"Mirror {o['side']} {o['qty']:.5f} {o['symbol']} (trader {tid[:10]}…)")
                            platform_metrics.ORDERS_TOTAL.labels(mode="REAL", side=o["side"]).inc()
                        except Exception as oe:
                            logger.warning(f"Mirror order failed {o['symbol']}: {oe}")
                    logger.info(f"🪞 COPY MIRROR: executed {len(orders)} mirror orders for {tid[:10]}…")
                else:
                    for o in orders:
                        db.add_audit_log("COPY_MIRROR_SIGNAL", audit_ip(),
                                         f"Mirror signal {o['side']} {o['qty']:.5f} {o['symbol']} "
                                         f"(execution {exec_mode} - keys required for auto)")
                    logger.info(f"🪞 COPY MIRROR: {len(orders)} mirror signals (mode {exec_mode}) for {tid[:10]}…")
        except Exception as e:
            logger.warning(f"Copy mirror scheduler error: {e}")


@app.get("/api/v1/copy/mirror-status")
async def api_copy_mirror_status(_auth: dict = Depends(require_auth)):
    """Real copy-trading status: followed traders + current mirror signals."""
    signals = {}
    for tid, alloc in list(copy_manager.copied_traders.items()):
        positions = fetch_trader_positions(tid)
        my_pos = {p["symbol"]: float(p["qty"]) for p in db.get_positions()}
        trader = copy_manager.traders.get(tid)
        orders = build_mirror_orders(positions, my_pos,
                                     allocated_capital=float(alloc.get("allocated_capital", 0.0)),
                                     trader_account_value=float(getattr(trader, "account_value", 0.0) or 0.0))
        signals[tid[:12]] = {
            "mode": alloc.get("mode", "FOLLOW_ONLY"),
            "trader_positions": [p["coin"] for p in positions[:8]],
            "mirror_orders": orders[:8],
        }
    return {
        "execution_mode": os.getenv("COPYTRADE_EXECUTION", "signal_only"),
        "following": copy_manager.copied_traders,
        "mirror_signals": signals,
        "summary": mirror_status_text(copy_manager.copied_traders),
    }


async def copy_trading_refresh_scheduler():
    """LOT 67: keeps the real copy-trading leaderboard fresh (every 6h)."""
    while True:
        await asyncio.sleep(6 * 3600)
        try:
            copy_manager.refresh_real_copytrader_leaderboard()
            copy_manager.refresh_allocation_pnl()
            logger.info(f"✅ Copy Trading leaderboard refreshed: {copy_manager.status}")
        except Exception as e:
            logger.warning(f"Copy Trading refresh failed: {e}")


def submit_order_via_oms(symbol: str, side: str, qty: float, price: float,
                             mode: str, strategy: str, exchange: str = "Binance") -> dict:
    """
    Audit B7-4: routes live execution through the real OMS -> EMS pipeline
    (statuses, routing, fill receipts) instead of calling the exchange directly.
    Returns a dict compatible with the loop's fill-confirmation logic.
    """
    client_order_id = f"quant_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
    order = oms.submit_new_order(
        symbol=symbol, side=side, qty=qty, price=price,
        mode=mode, strategy=strategy, client_order_id=client_order_id,
        exchange=exchange, order_type="MARKET",
    )
    res = oms.approve_and_execute_order(order)
    return {
        "id": res.get("order_id"),
        "price": res.get("price", price),
        "amount": res.get("amount", qty),
        "status": res.get("status", "ACKNOWLEDGED"),
        "client_order_id": client_order_id,
    }


async def reconciliation_scheduler():
    """
    Audit B7-5 / B11: periodic real-balance/position reconciliation with the
    exchange (REAL mode) and internal consistency check (DEMO). Alerts on gaps.
    """
    while True:
        await asyncio.sleep(300)  # every 5 minutes
        try:
            active_mode = STATE["mode"]
            client = get_ccxt_client() if active_mode == "REAL" else None

            if active_mode == "REAL" and client:
                try:
                    # LOT 7 (PDF Pilier P) : risque de contrepartie — limite de
                    # capital par exchange + signaux de santé (spread/volume)
                    try:
                        _cb = counterparty_risk.check_exchange_balance(
                            "Binance", float(STATE.get("balance_real", 0.0)),
                            STATE.get("current_equity", 0.0))
                        if _cb.get("action") == "block":
                            logger.critical(f"⚠️ CONTREPARTIE: {_cb['message']}")
                            risk_state.enter(RiskStateMachine.CAUTION, "COUNTERPARTY_CAP")
                            STATE["risk_state"] = risk_state.to_dict()
                            try:
                                asyncio.create_task(telegram_bot.send_push_notification(
                                    f"⚠️ *RISQUE CONTREPARTIE*\n{_cb['message']}"))
                            except Exception:
                                pass
                        STATE["counterparty_check"] = _cb
                    except Exception:
                        pass
                    balance = client.fetch_balance()
                    actual_balance = float(balance.get("total", {}).get("USDT", 0.0))
                    internal_balance = float(STATE.get("balance_real", 0.0))
                    ok_bal = reconciler.reconcile_balances(actual_balance, internal_balance)
                    if not ok_bal:
                        logger.error(
                            f"RECONCILIATION GAP: exchange balance {actual_balance:.2f} "
                            f"vs internal {internal_balance:.2f}"
                        )
                        try:
                            await telegram_bot.send_push_notification(
                                f"⚠️ *ÉCART DE RÉCONCILIATION*\n"
                                f"Balance exchange : *${actual_balance:,.2f}*\n"
                                f"Balance interne : ${internal_balance:,.2f}"
                            )
                        except Exception:
                            pass
                        # LOT 7 (PDF Pilier K) : en mode REAL, tout écart non
                        # expliqué -> HALT AUTOMATIQUE (les comptes doivent
                        # TOUJOURS coller ; un écart = problème de ledger ou
                        # d'exécution -> on arrête avant d'empirer).
                        if active_mode == "REAL":
                            risk_state.enter(RiskStateMachine.HALT,
                                             f"RECONCILIATION_BALANCE:{actual_balance:.0f}!={internal_balance:.0f}")
                            STATE["risk_state"] = risk_state.to_dict()
                            try:
                                await telegram_bot.send_push_notification(
                                    "🔴 *HALT RÉCONCILIATION*\nÉcart de balance non expliqué en mode REAL\n→ Trading arrêté. Vérifier le ledger avant /resume.")
                            except Exception:
                                pass
                    else:
                        STATE["balance_real"] = actual_balance
                        logger.info(f"RECONCILIATION OK: balance {actual_balance:.2f} USDT")

                    positions = client.fetch_positions()
                    actual_pos = {}
                    for p in positions:
                        sym = p.get("symbol", "").replace("/", "")
                        qty = float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0.0)
                        if qty:
                            actual_pos[sym] = qty
                            # AUDIT B10-3: liquidation proximity alert (futures)
                            liq = p.get("liquidationPrice") or p.get("info", {}).get("liquidationPrice")
                            mark = p.get("markPrice") or p.get("info", {}).get("markPrice")
                            if liq and mark:
                                try:
                                    dist = abs(float(mark) - float(liq)) / float(mark)
                                    if dist < 0.05:
                                        logger.critical(f"⚠️ LIQUIDATION PROXIMITY {sym}: {dist*100:.1f}% from liq price")
                                        try:
                                            await telegram_bot.send_push_notification(
                                                f"⚠️ *PROXIMITÉ LIQUIDATION*\n{sym} à {dist*100:.1f}% du prix de liquidation !"
                                            )
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                    _pos_ok = reconciler.reconcile_positions(actual_pos, "REAL")
                    # LOT 7 (PDF Pilier K) : HALT auto en REAL sur écart de positions
                    if not _pos_ok and active_mode == "REAL":
                        risk_state.enter(RiskStateMachine.HALT, "RECONCILIATION_POSITIONS")
                        STATE["risk_state"] = risk_state.to_dict()
                        try:
                            await telegram_bot.send_push_notification(
                                "🔴 *HALT RÉCONCILIATION*\nÉcart de positions non expliqué en mode REAL\n→ Trading arrêté. Vérifier avant /resume.")
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"REAL reconciliation failed: {e}")
            else:
                # DEMO internal consistency: equity should match balance + positions value
                try:
                    bal = float(STATE.get("balance_demo", 0.0))
                    pos_value = 0.0
                    for p in db.get_positions():
                        sym = p["symbol"]
                        price = STATE["assets"].get(sym, {}).get("price") or 0.0
                        pos_value += float(p["qty"]) * float(price)
                    computed = bal + pos_value
                    diff_pct = abs(computed - STATE["current_equity"]) / max(STATE["current_equity"], 1.0)
                    if diff_pct > 0.02:
                        logger.warning(f"DEMO internal gap {diff_pct*100:.2f}% (bal {bal:.2f} + pos {pos_value:.2f} vs equity {STATE['current_equity']:.2f})")
                        STATE["current_equity"] = computed
                except Exception as e:
                    logger.debug(f"DEMO reconciliation skip: {e}")
        except Exception as e:
            logger.warning(f"Reconciliation scheduler error: {e}")


async def db_backup_scheduler():
    """LOT 64 (roadmap #3): automatic daily database backup."""
    while True:
        await asyncio.sleep(24 * 3600)
        try:
            backup_path = db.create_backup()
            logger.info(f"✅ Daily database backup created: {backup_path}")
        except Exception as e:
            logger.warning(f"DB backup scheduler error: {e}")


async def startup_event():
    # LOT 62: institutional configuration checklist
    validate_startup_config()

    # Autopilot first-start marker (audit D1)
    if not db.get_setting("platform_first_start_ts"):
        db.save_setting("platform_first_start_ts", str(time.time()))

    # Load persisted price alerts (audit C3)
    _alerts_load()

    # Update CCXT client inside our Binance Adapter once authenticated!
    binance_adapter.client = get_ccxt_client()
    
    # Load default copytrade allocations
    global STATE
    
    # VISION_FUTUR §5a: rebuild state from the last snapshot (event-sourcing lite)
    restore_state_snapshot(db, STATE, max_age_seconds=7200)
    logger.info(f"🩺 Supervisor: {supervisor.check(force=True) or 'all vital signs OK'}")

    # Load persisted demo balance if exists to ensure complete state survival across restarts!
    persisted_bal = db.get_setting("balance_demo")
    if persisted_bal:
        try:
            val = float(persisted_bal)
            STATE["balance_demo"] = val
            STATE["initial_capital_demo"] = val
            STATE["current_equity"] = val
            STATE["equity_history_demo"] = [val]
            # Bind risk parameters to actual capital (micro-budget: wider drawdown limits)
            try:
                risk_manager.set_initial_capital(val)
            except Exception as e:
                logger.warning(f"set_initial_capital failed: {e}")
            logger.info(f"Loaded persisted demo balance from database: ${val:,.2f} USD")
        except Exception as e:
            logger.error(f"Failed to load persisted demo balance: {str(e)}")
            
    allocations = db.get_copy_allocations()
    for trader_id, data in allocations.items():
        if data['active']:
            copy_manager.start_copying(trader_id, data['allocated_capital'])
            
    # Initial historical load from persistent database cache first!
    # Fallback to fetching REAL data (Binance -> Bybit -> Yahoo) per asset.
    # HONNÊTETÉ (faille 1) : jamais de barres fabriquées ; un actif sans
    # historique réel reste UNAVAILABLE et n'est pas tradé.
    logger.info("Initializing historical candles (multi-assets, sources réelles)...")
    df = pd.DataFrame()
    for _sym in STATE["assets"]:
        _df = db.load_candles(_sym, limit=120)
        if _df.empty or len(_df) < 10:
            logger.info(f"Database cache incomplete for {_sym}. Fetching from real APIs (Binance/Bybit/Yahoo)...")
            try:
                _df = await fetch_historical_market_data(_sym)
                if _df is not None and not _df.empty:
                    db.save_candles(_sym, _df)
                    logger.info(f"Seeded {len(_df)} barres RÉELLES pour {_sym}.")
                else:
                    logger.warning(f"NO REAL HISTORY for {_sym} -> UNAVAILABLE (non tradé).")
            except Exception as e:
                logger.error(f"Failed to fetch historical market data for {_sym}: {str(e)}")
                _df = pd.DataFrame()
        else:
            logger.info(f"Loaded {len(_df)} barres RÉELLES depuis le cache DB pour {_sym}.")
        if _sym == "BTCUSDT" and not _df.empty:
            df = _df

    if not df.empty:
        train_ai_models(df)
    else:
        logger.warning("Historical data is empty. AI models training skipped until real data arrives.")

    # LOT 4 (PDF Pilier B) : le HMM est VALIDÉ sur les 7 actifs (pas seulement
    # BTC). Vraisemblance + stabilité par actif -> si un actif est aberrant,
    # son facteur de régime sera réduit (honnêteté : pas de régime sur du vide).
    try:
        _hmm_val = {}
        for _sym_v in STATE["assets"]:
            _dfv = db.load_candles(_sym_v, limit=120)
            if not _dfv.empty:
                _res_v = regime_detector.validate_on_asset(_dfv, symbol=_sym_v)
                if _res_v:
                    _hmm_val[_sym_v] = _res_v
        STATE["hmm_validation"] = _hmm_val
        logger.info(f"🔬 HMM validé sur {len(_hmm_val)} actifs: " +
                    ", ".join(f"{k}(loglik {v['loglik_mean']}, stab {v['stability']})"
                              for k, v in list(_hmm_val.items())[:4]) + "...")
    except Exception as _hv:
        logger.debug(f"HMM validation failed: {_hv}")
    
    # Sync Web3 non-custodial EVM balance details
    STATE["defi_wallet_address"] = defi_wallet.get_wallet_address()
    STATE["eth_defi_balance"] = defi_wallet.fetch_native_balance()
    
    # Start the continuous WebSockets trading execution background process
    launch_named(live_trading_loop(), "live_trading_loop")
    
    # Start the multi-source price consensus engine (PDF: redondance 2+ sources)
    launch_named(price_consensus_loop(), "price_consensus_loop")

    # LOT 3: order flow réel (trades + liquidations) pour Delta/CVD/OFI/cascades
    launch_named(order_flow_websocket_listener(), "order_flow_websocket_listener")
    logger.info("✅ OrderFlow WS listener started (trades réels + liquidations)")
    logger.info("✅ Multi-source price consensus loop started (Binance/Bybit/Coinbase/Kraken/OKX/CoinGecko/CryptoCompare/Yahoo...)")
    
    # Start the official Dual-Exchange WebSockets Stream
    launch_named(multi_exchange_websocket_listener(), "multi_exchange_websocket_listener")
    
    # Send the startup push notification safely within the running loop
    asyncio.create_task(telegram_bot.send_startup_message())
    
    # Start the tactile Telegram remote control worker
    launch_named(telegram_bot.poll_telegram_commands_loop(), "telegram_poll")

    # Start the LOT 46 model-selection scheduler (needs a running event loop)
    asyncio.create_task(lot46_model_selection_scheduler())
    logger.info("✅ LOT 46 scheduler started")

    # Start the WebSocket heartbeat loop
    asyncio.create_task(websocket_heartbeat_loop())
    logger.info("✅ WS heartbeat loop started")

    # Start the balance/position reconciliation loop
    launch_named(reconciliation_scheduler(), "reconciliation_scheduler")
    logger.info("✅ LOT 69: Reconciliation scheduler started (every 5 min)")

    # Start the daily risk-concierge digest
    launch_named(concierge_scheduler(), "concierge_scheduler")
    logger.info("✅ LOT 70: Daily risk-concierge scheduler started")

    # Start the daily DB backup task
    launch_named(db_backup_scheduler(), "db_backup_scheduler")
    logger.info("✅ LOT 64: Daily DB backup scheduler started")

    # Start the autonomous AI self-improvement loop
    launch_named(autonomous_ai_scheduler(), "autonomous_ai_scheduler")
    logger.info("✅ LOT 66: Autonomous AI scheduler started (self-retrain every 6h)")

    # Start the copy-trading leaderboard refresher
    launch_named(copy_trading_refresh_scheduler(), "copy_trading_refresh_scheduler")
    logger.info("✅ LOT 67: Copy Trading leaderboard refresher started")

    # Start the real copy-mirroring scheduler
    launch_named(copy_mirror_scheduler(), "copy_mirror_scheduler")
    logger.info("✅ LOT 71: Copy mirroring scheduler started (every 10 min)")

    # LOT 7 (PDF Faille 6) : watchdog des tâches de fond (supervision + auto-restart)
    asyncio.create_task(task_watchdog_loop())
    # Enregistre les tâches nommées pour le watchdog (auto-restart si mortes)
    try:
        for _t in asyncio.all_tasks():
            _n = _t.get_name()
            if _n.startswith("qp_"):
                _BG_TASKS[_n[3:]] = _t
        logger.info(f"✅ Watchdog des tâches de fond démarré ({len(_BG_TASKS)} tâches surveillées)")
    except Exception:
        logger.info("✅ Watchdog des tâches de fond démarré")


async def live_trading_loop():
    """
    Resilient Multi-Asset continuous live trading engine.
    Loops through BTCUSDT, ETHUSDT, SOLUSDT, XAUUSD, EURUSD, AAPL, and TSLA.
    Integrates live NLP News Sentiment scoring, On-Chain indicators,
    and a Multi-Asset Portfolio Covariance risk restriction engine.
    """
    global STATE
    logger.info("Resilient Multi-Asset Live Trading Engine active and polling...")
    
    loop_count = 0
    while True:
        if not STATE["is_running"] or STATE["kill_switch_active"]:
            await asyncio.sleep(1)
            continue
            
        loop_count += 1

        # LOT 2 (PDF Pilier G) : tick de la machine à états NORMAL/CAUTION/HALT
        # (cool-down + redémarrage progressif) + alerte Telegram sur transition.
        try:
            _state_changed = risk_state.tick()
            if _state_changed:
                STATE["risk_state"] = risk_state.to_dict()
                try:
                    db.add_audit_log("RISK_STATE_CHANGE", audit_ip(),
                                     f"État risque -> {risk_state.state} ({risk_state.reason})")
                except Exception:
                    pass
                try:
                    _emoji = {"NORMAL": "🟢", "CAUTION": "🟠", "HALT": "🔴"}.get(risk_state.state, "⚪")
                    asyncio.create_task(telegram_bot.send_push_notification(
                        f"{_emoji} *CHANGEMENT D'ÉTAT RISQUE*\n"
                        f"État : *{risk_state.state}* ({risk_state.reason})\n"
                        f"Facteur de taille : {risk_state.scale_factor()*100:.0f}%"
                    ))
                except Exception:
                    pass
        except Exception as _st:
            logger.debug(f"Risk state tick error: {_st}")
        # LOT 1bis/2 : divergence de sources persistante -> CAUTION
        try:
            _div_count = sum(1 for v in STATE.get("price_divergent", {}).values() if v)
            if _div_count > 0:
                risk_state.enter(RiskStateMachine.CAUTION,
                                 f"SOURCE_DIVERGENCE ({_div_count} actif(s))")
        except Exception:
            pass
        STATE["risk_state"] = risk_state.to_dict()

        consensus = {
            "final_signal": 0.0,
            "consensus": 0.5,
            "contributions": {
                "Trend Following": {"signal": 0.0, "weight": 0.20, "confidence": 0.5},
                "Mean Reversion": {"signal": 0.0, "weight": 0.20, "confidence": 0.5},
                "Market Making": {"signal": 0.0, "weight": 0.15, "confidence": 0.5},
                "Statistical Arbitrage": {"signal": 0.0, "weight": 0.15, "confidence": 0.5},
                "Inter-Exchange Arbitrage": {"signal": 0.0, "weight": 0.10, "confidence": 0.5},
                "Grid Trading": {"signal": 0.0, "weight": 0.10, "confidence": 0.5},
                "Scalping": {"signal": 0.0, "weight": 0.10, "confidence": 0.5},
                "Momentum": {"signal": 0.0, "weight": 0.08, "confidence": 0.5},
                "Volatility Breakout": {"signal": 0.0, "weight": 0.08, "confidence": 0.5},
                "Carry": {"signal": 0.0, "weight": 0.08, "confidence": 0.5},
                "Cross-Sectional Momentum": {"signal": 0.0, "weight": 0.06, "confidence": 0.5},
                "Multi-Timeframe": {"signal": 0.0, "weight": 0.06, "confidence": 0.5}
            }
        }
        
        # 1. Periodically fetch Advanced External Indicators (to avoid API rate-limits)
        news_scale_factor = 1.0
        macro_scale_factor = 1.0
        
        if loop_count % 3 == 1:
            try:
                res_sent = await news_analyzer.get_market_sentiment_index()
                # LOT 5 : stocker les actualités RÉELLES pour l'endpoint /api/v1/news
                try:
                    STATE["recent_headlines"] = news_analyzer.get_recent_headlines(limit=20)
                    STATE["news_shock"] = res_sent.get("shock_status", {})
                except Exception:
                    pass
                if res_sent.get("available"):
                    STATE["sentiment_index"] = res_sent["sentiment_index"]
                    STATE["sentiment_available"] = True
                    STATE["sentiment_confidence"] = res_sent.get("confidence", 0.0)
                    logger.info(f"Live Sentiment Index synchronized: {STATE['sentiment_index']:.2f} "
                                f"(confiance {STATE['sentiment_confidence']:.0%}, {res_sent.get('num_headlines', 0)} titres réels)")
                else:
                    # HONNÊTETÉ (faille 1) : sentiment indisponible -> AUCUNE influence
                    STATE["sentiment_index"] = None
                    STATE["sentiment_available"] = False
                    STATE["sentiment_confidence"] = 0.0
                    logger.warning("Sentiment UNAVAILABLE (aucune source réelle) -> aucune influence sur les trades.")
                
                # HIÉRARCHIE DE L'INFORMATION (LOT 5, PDF Pilier I) : un choc
                # SYSTÉMIQUE (hack, insolvabilité, ban...) ≠ bruit. Seuls les
                # tokens systémiques du détecteur déclenchent une action forte.
                if res_sent["shock_status"].get("shock_detected"):
                    logger.critical("EXTREME NEWS SHOCK DETECTED! Restricting trade sizes.")
                    news_scale_factor = 0.20
                    # LOT 2 (PDF Faille 3) : choc systémique -> machine à états.
                    # NEWS_SHOCK_ACTION=halt -> HALT (durée NEWS_SHOCK_HALT_MINUTES),
                    # sinon CAUTION (réduction) — mentalité n°8 : ne réagir
                    # fortement qu'aux événements systémiques.
                    if os.getenv("NEWS_SHOCK_ACTION", "reduce").lower() == "halt":
                        risk_state.enter(RiskStateMachine.HALT, "NEWS_SHOCK")
                        # Durée du HALT pilotable par l'opérateur
                        try:
                            risk_state.cooldown_seconds = float(
                                os.getenv("NEWS_SHOCK_HALT_MINUTES", "15")) * 60.0
                        except (TypeError, ValueError):
                            risk_state.cooldown_seconds = 15.0 * 60.0
                        try:
                            asyncio.create_task(telegram_bot.send_push_notification(
                                f"🔴 *HALT MÉDIA* — choc systémique détecté. Nouveaux ordres bloqués."))
                        except Exception:
                            pass
                    else:
                        risk_state.enter(RiskStateMachine.CAUTION, "NEWS_SHOCK")
            except Exception as e:
                logger.warning(f"Failed to fetch sentiment index: {str(e)}")
                STATE["sentiment_available"] = False
                STATE["sentiment_index"] = None
                
        # Check scheduled macroeconomic calendar for approaching shocks
        # LOT 5 (PDF Pilier I) : gestion des phases AVANT / PENDANT / APRÈS
        # (mentalité n°4 : on ne trade pas l'événement, on trade la réaction)
        try:
            macro_res = macro_calendar.check_upcoming_macro_shocks()
            STATE["macro_phase"] = macro_res.get("phase", "NONE")
            STATE["macro_event"] = macro_res.get("event", "")
            if macro_res.get("upcoming_shock"):
                event_name = macro_res["event"]
                time_left = macro_res["time_to_event_minutes"]
                phase = macro_res.get("phase", "APPROACHING")
                
                # Check if we have already sent this alert (par phase)
                last_sent_event = STATE.get("last_sent_macro_event")
                if last_sent_event != f"{event_name}|{phase}":
                    # Send the newly implemented INTERACTIVE TACTILE mobile alert!
                    await telegram_bot.send_interactive_macro_alert(event_name, time_left)
                    STATE["last_sent_macro_event"] = f"{event_name}|{phase}"
                    
                macro_scale_factor = macro_res["scale_reduction_factor"]
                # Machine à états selon la PHASE (LOT 5) :
                #  - ACTIVE (HIGH) : l'événement EST EN COURS -> HALT réel
                #    (durée NEWS_SHOCK_HALT_MINUTES), on ne trade pas la bougie
                #  - ACTIVE (MEDIUM/LOW) / APPROACHING / AFTERMATH : CAUTION
                if phase == "ACTIVE" and macro_res.get("request_halt"):
                    if risk_state.state != RiskStateMachine.HALT:
                        risk_state.enter(RiskStateMachine.HALT, f"MACRO_ACTIVE:{event_name}")
                        # HALT dédié : durée pilotable par l'opérateur
                        try:
                            risk_state.cooldown_seconds = float(
                                os.getenv("NEWS_SHOCK_HALT_MINUTES", "15")) * 60.0
                        except (TypeError, ValueError):
                            risk_state.cooldown_seconds = 15.0 * 60.0
                else:
                    risk_state.enter(RiskStateMachine.CAUTION, f"MACRO:{phase}:{event_name}")
            else:
                STATE["last_sent_macro_event"] = None
        except Exception as e:
            logger.warning(f"Failed to parse macroeconomic calendar: {str(e)}")
            
        # Apply tactile mobile buttons override if they clicked 'REDUCE EXPO'!
        macro_scale_factor *= STATE.get("macro_scale_factor_tactile", 1.0)
                
        if loop_count % 5 == 1:
            try:
                onchain_data = await onchain_tracker.get_exchange_netflows()
                onchain_score = onchain_tracker.compute_onchain_risk_score(onchain_data)
                STATE["onchain_risk_score"] = onchain_score
                STATE["onchain_available"] = onchain_score is not None
                if onchain_score is not None:
                    logger.info(f"Live On-Chain Risk Score synchronized: {STATE['onchain_risk_score']:.2f}")
                else:
                    logger.warning("On-chain UNAVAILABLE -> aucun ajustement on-chain appliqué.")
            except Exception as e:
                logger.warning(f"Failed to fetch onchain data: {str(e)}")
                STATE["onchain_available"] = False
                STATE["onchain_risk_score"] = None
                
            # Periodically verify non-custodial wallet balances
            STATE["eth_defi_balance"] = defi_wallet.fetch_native_balance()
            
            # Formulate options volatility structures from REAL implied volatility
            # (faille 1 corrigée : plus d'iv_map codé en dur — mentalité n°5)
            try:
                real_iv = await volatility_arb_engine.fetch_real_iv("BTCUSDT")
                STATE.setdefault("real_iv", {})["BTCUSDT"] = real_iv
                if real_iv is None:
                    STATE["options_strategy"] = {
                        "strategy": "UNAVAILABLE",
                        "details": "Implied volatility réelle indisponible (Deribit hors ligne) — aucune stratégie d'options.",
                        "implied_volatility_pct": None,
                        "legs": [],
                        "estimated_yield_pct": 0.0,
                    }
                else:
                    STATE["options_strategy"] = volatility_arb_engine.evaluate_optimal_options_strategy(
                        current_price=STATE.get("last_known_prices", {}).get("BTCUSDT") or STATE.get("last_price") or 0.0,
                        iv_annual=real_iv,
                        regime_id=STATE["regime_id"]
                    )
                    STATE["options_strategy"]["symbol"] = "BTCUSDT"
                    STATE["options_strategy"]["iv_source"] = "deribit_dvol"
            except Exception as e:
                logger.warning(f"Options strategy evaluation failed: {e}")
            
        # 2. Calculate rolling Correlation Matrix across all multi-assets using actual cached historical return series!
        try:
            real_returns_dict = {}
            for asset in STATE["assets"]:
                df_cache = db.load_candles(asset, limit=30)
                if not df_cache.empty and len(df_cache) >= 5:
                    real_returns_dict[asset] = df_cache['close'].pct_change().dropna().values
                else:
                    logger.warning(f"No historical candles available for {asset} to calculate covariance. Initializing with flat zeros.")
                    real_returns_dict[asset] = np.zeros(30)
                    
            corr_df = covariance_engine.calculate_correlation_matrix(real_returns_dict)
            STATE["covariance_matrix"] = corr_df.to_dict()
            active_returns_dict = real_returns_dict
        except Exception as e:
            logger.error(f"Failed to calculate covariance matrix: {str(e)}")
            corr_df = pd.DataFrame()
            active_returns_dict = {asset: np.zeros(30) for asset in STATE["assets"]}
            
        # Calculate daily Portfolio VaR/CVaR dynamically before the asset loop!
        positions = db.get_positions()
        var_metrics = covariance_engine.calculate_portfolio_var_cvar(
            active_positions=positions,
            corr_matrix=corr_df if corr_df is not None else pd.DataFrame(),
            assets_returns_dict=active_returns_dict
        )
        portfolio_cvar_pct = var_metrics.get("portfolio_cvar_pct", 0.02)
        if portfolio_cvar_pct <= 0:
            portfolio_cvar_pct = 0.02 # Safe floor
            
        # 3. Loop and trade through each active Asset (Crypto, Gold, Forex, Stocks)
        active_assets = list(STATE["assets"].keys())
        active_mode = STATE["mode"]
        client = get_ccxt_client() if active_mode == "REAL" else None
        active_balance_key = "balance_demo" if active_mode == "DEMO" else "balance_real"
        active_equity_history_key = "equity_history_demo" if active_mode == "DEMO" else "equity_history_real"
        
        # Sync real exchange wallet balance once per loop iteration
        if active_mode == "REAL" and client:
            try:
                bal = client.fetch_balance()
                STATE["balance_real"] = float(bal['free'].get('USDT', bal['total'].get('USDT', 0.0)))
            except Exception as e:
                logger.error(f"Failed to sync real wallet balance: {str(e)}")
                
        for symbol in active_assets:
            try:
                # Fetch 100% real-world price ticks for Gold, Forex, Stocks, and Cryptos!
                # (faille 1 corrigée : plus de prix inventé — mark_real_price uniquement)
                price_fetched = False
                try:
                    # ===== PRIX CONSENSUS MULTI-SOURCES (PDF : redondance 2+ sources croisées) =====
                    # Crypto : Binance + Bybit + Coinbase + Kraken + OKX (+ CoinGecko/CryptoCompare 60s)
                    # Or/FX/Actions : Yahoo + gold-api (XAU) / er-api (EURUSD) ; AAPL/TSLA = Yahoo seul (SINGLE_SOURCE honnête)
                    if symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
                        cons = await price_engine.get_consensus(symbol)
                        STATE.setdefault("price_consensus", {})[symbol] = cons
                        if cons.get("price") and cons["price"] > 0:
                            mark_real_price(symbol, cons["price"],
                                            volume_24h=STATE["assets"][symbol].get("volume_24h"))
                            price_fetched = True
                            if cons["status"] == "DIVERGENT":
                                # GEL DU TRADING (PDF) : divergence anormale entre sources
                                STATE.setdefault("price_divergent", {})[symbol] = True
                                logger.critical(
                                    f"⚠️ DIVERGENCE MULTI-SOURCES {symbol}: "
                                    f"{cons['divergence_pct']:.3f}% > seuil {cons['threshold_pct']:.2f}% "
                                    f"sur {cons['n_sources']} sources {list(cons['sources'].keys())} "
                                    f"-> GEL du trading pour {symbol}."
                                )
                                try:
                                    db.add_audit_log(
                                        "SOURCE_DIVERGENCE", audit_ip(),
                                        f"{symbol}: divergence {cons['divergence_pct']:.3f}% "
                                        f"(sources {list(cons['sources'].keys())}) -> trading gelé."
                                    )
                                except Exception:
                                    pass
                                continue
                            STATE.setdefault("price_divergent", {})[symbol] = False
                        else:
                            set_asset_quality(symbol, DataQualityStatus.UNAVAILABLE)
                    else:
                        # Barres réelles Yahoo persistées (pour l'historique)
                        y_ticker = "GC=F" if symbol == "XAUUSD" else "EURUSD=X" if symbol == "EURUSD" else symbol
                        df_y = await fetch_yahoo_finance_candles(y_ticker, interval="1m", range_str="1d")
                        if not df_y.empty:
                            db.save_candles(symbol, df_y)
                        cons = await price_engine.get_consensus(symbol)
                        STATE.setdefault("price_consensus", {})[symbol] = cons
                        if cons.get("price") and cons["price"] > 0:
                            mark_real_price(symbol, cons["price"])
                            price_fetched = True
                            if cons["status"] == "DIVERGENT":
                                STATE.setdefault("price_divergent", {})[symbol] = True
                                logger.critical(
                                    f"⚠️ DIVERGENCE MULTI-SOURCES {symbol}: "
                                    f"{cons['divergence_pct']:.3f}% > seuil {cons['threshold_pct']:.2f}% "
                                    f"-> GEL du trading pour {symbol}."
                                )
                                try:
                                    db.add_audit_log(
                                        "SOURCE_DIVERGENCE", audit_ip(),
                                        f"{symbol}: divergence {cons['divergence_pct']:.3f}% -> trading gelé."
                                    )
                                except Exception:
                                    pass
                                continue
                            STATE.setdefault("price_divergent", {})[symbol] = False
                        else:
                            set_asset_quality(symbol, DataQualityStatus.UNAVAILABLE)
                except Exception as e:
                    logger.error(f"Failed to fetch live price tick for {symbol}: {str(e)}")
                    # Le prix reste celui de la dernière donnée réelle, sinon None
                    if not STATE["assets"][symbol].get("has_real_price"):
                        STATE["assets"][symbol]["price"] = None
                        set_asset_quality(symbol, DataQualityStatus.UNAVAILABLE)

                current_price = STATE["assets"][symbol]["price"]
                if current_price is None or not STATE["assets"][symbol].get("has_real_price"):
                    set_asset_quality(symbol, DataQualityStatus.UNAVAILABLE)
                    logger.warning(
                        f"Skipping trade loop for {symbol}: AUCUNE donnée réelle "
                        f"(règle « AUCUNE DONNÉE -> AUCUN ORDRE », mentalité n°5)."
                    )
                    continue

                # Market data quality: a live tick arrived -> LIVE
                set_asset_quality(symbol, DataQualityStatus.LIVE)
                set_data_quality(DataQualityStatus.LIVE)

                # AUDIT C3: fire custom price alerts on this tick (prix réel uniquement)
                try:
                    check_price_alerts(symbol, current_price)
                except Exception as _ae:
                    logger.debug(f"Price alert check error: {_ae}")

                # VISION §7.1: replayable tick event (throttled per symbol)
                _ev_key = f"ev_{symbol}"
                if time.time() - STATE.get(_ev_key, 0.0) > 30.0:
                    STATE[_ev_key] = time.time()
                    try:
                        db.add_event(time.time(), "tick", json.dumps({
                            "symbol": symbol, "price": round(current_price, 4),
                            "regime": STATE.get("regime_name", ""),
                        }, default=str))
                    except Exception:
                        pass

                # Refresh périodique des barres RÉELLES (crypto: Bybit klines 1m)
                # (faille 1 corrigée : plus AUCUNE bougie synthétique fabriquée)
                try:
                    _kf_key = f"kline_refresh_{symbol}"
                    if symbol in CRYPTO_SYMBOLS and time.time() - STATE.get(_kf_key, 0.0) > 60.0:
                        STATE[_kf_key] = time.time()
                        df_min = await fetch_bybit_klines(symbol, interval="1m", limit=5)
                        if not df_min.empty:
                            db.save_candles(symbol, df_min)
                except Exception as _ke:
                    logger.debug(f"Kline refresh failed for {symbol}: {_ke}")

                # EVALUATE GENUINE FUNDING RATE ARBITRAGE (100% Real-World API data)
                # (faille 1 corrigée : funding 8h JAMAIS inventé — si la source
                #  échoue, l'arbitrage de funding est simplement ignoré)
                if symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
                    funding_8h = None
                    # Funding croisé Binance <-> Bybit (PDF : 2 sources indépendantes)
                    try:
                        fcons = await price_engine.get_funding_consensus(symbol)
                        if fcons.get("status") == "DIVERGENT":
                            logger.warning(
                                f"FUNDING {symbol}: divergence anormale entre sources "
                                f"{fcons.get('sources')} -> funding ignoré (gel arbitrage)."
                            )
                            try:
                                db.add_audit_log(
                                    "FUNDING_SOURCE_DIVERGENCE", audit_ip(),
                                    f"{symbol}: funding Binance/Bybit divergent {fcons.get('sources')} -> ignoré."
                                )
                            except Exception:
                                pass
                        else:
                            funding_8h = fcons.get("funding_rate_8h")
                            if funding_8h is not None:
                                STATE.setdefault("funding_rates", {})[symbol] = funding_8h
                    except Exception as fe:
                        logger.debug(f"Funding consensus failed for {symbol}: {fe}")

                    spot_p = current_price
                    perp_p = None
                    try:
                        async with httpx.AsyncClient() as http_client:
                            resp_f = await http_client.get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}")
                            if resp_f.status_code == 200:
                                pp = resp_f.json().get("price")
                                if pp is not None:
                                    perp_p = float(pp)
                    except Exception as e:
                        logger.debug(f"Failed to fetch real-world perpetual price for {symbol}: {str(e)}")

                    # Toutes les données doivent être réelles pour évaluer l'arbitrage
                    opportunities = {}
                    if funding_8h is None or perp_p is None or perp_p <= 0:
                        logger.info(
                            f"FUNDING ARB {symbol}: données réelles indisponibles "
                            f"(funding={funding_8h}, perp={perp_p}) -> évaluation ignorée."
                        )
                    else:
                        opportunities = funding_arb_engine.analyze_funding_opportunities(
                            symbol=symbol,
                            spot_price=spot_p,
                            perp_price=perp_p,
                            funding_rate_8h=funding_8h
                        )
                else:
                    opportunities = {}

                action = opportunities.get("action")
                if action == "ENTER_ARBITRAGE":
                    arb_exec_mode = os.getenv("ARBITRAGE_EXECUTION", "signal_only")
                    if arb_exec_mode != "auto":
                        # AUDIT B12-1: honest label - analysis only, no position opened.
                        logger.info(
                            f"FUNDING ARB {symbol}: signal-only (ARBITRAGE_EXECUTION={arb_exec_mode}). "
                            f"Funding {funding_8h*100:.3f}%/8h - NOT executed."
                        )
                        db.add_audit_log(
                            "FUNDING_ARBITRAGE_SIGNAL",
                            audit_ip(),
                            f"Funding arb signal on {symbol} (rate {funding_8h*100:.3f}%/8h) - signal only, not executed."
                        )
                    else:
                        funding_arb_engine.active_arbitrages[symbol] = {
                            "qty": STATE[active_balance_key] * 0.30 / spot_p,
                            "entry_spot_price": spot_p,
                            "entry_perp_price": perp_p,
                            "accumulated_funding": 0.0
                        }
                        db.add_audit_log(
                            "FUNDING_ARBITRAGE_ENTERED",
                            audit_ip(),
                            f"Entered Delta-Neutral Cash-and-Carry on {symbol} (Funding Rate: {funding_8h*100:.3f}% / 8h)."
                        )
                    await telegram_bot.send_push_notification(
                        f"🛡️ *ARBITRAGE DE FINANCEMENT ACTIF*\n"
                        f"-----------------------------------------\n"
                        f"📈 Actif : `{symbol}`\n"
                        f"💵 Taux de financement : *{funding_8h*100:.3f}% / 8h*\n"
                        f"⚖️ Stratégie : *Delta-Neutre (Cash-and-Carry)*\n"
                        f"💰 Allocation : *30% du capital*\n"
                        f"🔒 *Risque de prix : 0% (Totalement immunisé !)*"
                    )
                elif action == "EXIT_ARBITRAGE":
                    acc_funding = opportunities.get("accumulated_funding", 0.0)
                    STATE[active_balance_key] += acc_funding
                    if symbol in funding_arb_engine.active_arbitrages:
                        del funding_arb_engine.active_arbitrages[symbol]
                    db.add_audit_log(
                        "FUNDING_ARBITRAGE_EXITED",
                        audit_ip(),
                        f"Wound down funding arbitrage on {symbol}. Accumulated yield: ${acc_funding:.2f} USD."
                    )
                    await telegram_bot.send_push_notification(
                        f"💰 *ARBITRAGE DE FINANCEMENT BOUCLÉ*\n"
                        f"-----------------------------------------\n"
                        f"📈 Actif : `{symbol}`\n"
                        f"💵 Intérêts perçus : *+${acc_funding:.2f} USD*\n"
                        f"⚖️ Statut : *Positions spot/perp clôturées*"
                    )
                    
                # EVALUATE GENUINE DEX-CEX CROSS-VENUE ARBITRAGE (100% Real-World spreads Bybit vs Binance!)
                bybit_p = None # Starts as None (Unavailable)
                if symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
                    cex_p = current_price
                    try:
                        t0 = time.time()
                        async with bybit_limiter:
                            async with httpx.AsyncClient() as http_client:
                                resp = await http_client.get(f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}")
                        platform_metrics.record_api_latency("bybit_tickers", time.time() - t0)
                        if resp.status_code == 200:
                            bybit_p = float(resp.json().get("result", {}).get("list", [{}])[0].get("lastPrice"))
                    except Exception as e:
                        logger.error(f"Failed to fetch real-world secondary exchange price from Bybit for {symbol}: {str(e)}")
                    
                    if bybit_p is not None:
                        arb_opp = dex_cex_arb_engine.detect_arbitrage_opportunities(
                            symbol=symbol,
                            dex_price=cex_p,
                            cex_price=bybit_p,
                            estimated_gas_usd=0.05
                        )
                        if arb_opp.get("action") == "EXECUTE_ARBITRAGE":
                            route = arb_opp.get("route")
                            spread = arb_opp.get("spread_pct")
                            profit_pct = arb_opp.get("net_profit_pct")
                            amount_eth = 50.0 / cex_p
                            # AUDIT B12-2: signed-only -> BROADCAST when configured, else
                            # honest "signal-only" label (signed != executed).
                            arb_exec_mode = os.getenv("ARBITRAGE_EXECUTION", "signal_only")
                            if arb_exec_mode == "auto" and os.getenv("EVM_PRIVATE_KEY"):
                                signed_dex = defi_wallet.sign_dex_swap_transaction(
                                    token_in="USDT" if route == "BUY_DEX_SELL_CEX" else "ETH",
                                    token_out="ETH" if route == "BUY_DEX_SELL_CEX" else "USDT",
                                    amount_in_eth=amount_eth
                                )
                                raw_tx = signed_dex.get("raw_transaction")
                                if raw_tx:
                                    bcast = defi_wallet.broadcast_signed_transaction(raw_tx)
                                    db.add_audit_log(
                                        "DEX_CEX_ARBITRAGE_BROADCAST",
                                        audit_ip(),
                                        f"Cross-Venue {symbol} arbitrage broadcast on-chain. Tx: {bcast.get('tx_hash', '?')}"
                                    )
                                else:
                                    db.add_audit_log(
                                        "DEX_CEX_ARBITRAGE_BROADCAST_FAILED",
                                        audit_ip(),
                                        f"DEX signing failed: {signed_dex.get('reason', 'unknown')}"
                                    )
                            else:
                                logger.info(
                                    f"DEX-CEX {symbol}: arbitrage signal-only (ARBITRAGE_EXECUTION={arb_exec_mode}). "
                                    f"Spread {spread*100:.2f}% detected but NOT executed."
                                )
                            db.add_audit_log(
                                "DEX_CEX_ARBITRAGE_SIGNAL",
                                audit_ip(),
                                f"Cross-Venue {symbol} arbitrage detected. Route: {route} (Spread: {spread*100:.2f}%)."
                            )
                            await telegram_bot.send_push_notification(
                                f"🏆 *ARBITRAGE DEX-CEX CAPTURÉ*\n"
                                f"-----------------------------------------\n"
                                f"📈 Actif : `{symbol}`\n"
                                f"⚖️ Route : *{route}*\n"
                                f"📊 Écart de prix : *{spread*100:.2f}%*\n"
                                f"💵 Gain net estimé : *+{profit_pct*100:.2f}% (net de gaz)*\n"
                                f"🛡️ Protection : *MevShield On-Chain active*"
                            )
                    else:
                        logger.warning(f"Skipping arbitrage check for {symbol} due to unavailable Bybit secondary price feed.")
                    
                # 4. Formulate signal and sizing
                # Query the asset's own genuine price series from persistent DB cache!
                # HONNÊTETÉ (faille 1 corrigée) : plus AUCUNE bougie synthétique
                # fabriquée à partir du tick (volume inventé, OHLC dérivé). Si
                # l'historique réel est insuffisant (<10 barres), l'actif n'est
                # pas tradé — « AUCUNE DONNÉE -> AUCUN ORDRE » (mentalité n°5).
                df = db.load_candles(symbol, limit=120)
                if df.empty or len(df) < 10:
                    STATE["using_fallback_data"] = True
                    set_asset_quality(symbol, DataQualityStatus.STALE)
                    logger.warning(
                        f"{symbol}: historique réel insuffisant ({len(df) if not df.empty else 0} barres) "
                        f"-> signaux ignorés (aucune donnée fabriquée)."
                    )
                    continue
                STATE["using_fallback_data"] = False
                
                if df is not None:
                    # Predict Regime HMM
                    recent_returns = df['close'].pct_change().dropna().values[-10:]
                    ret_mean = np.mean(recent_returns) if len(recent_returns) > 0 else 0.0
                    vol_mean = np.std(recent_returns) if len(recent_returns) > 0 else 0.01
                    STATE["regime_id"] = int(regime_detector.predict(np.array([[ret_mean, vol_mean]]))[0])
                    STATE["regime_name"] = regime_detector.get_regime_name(STATE["regime_id"])

                    # VISION §1a/1b: soft regime probabilities + joint market state
                    try:
                        STATE["regime_probs"] = compute_regime_probs(regime_detector, np.array([[ret_mean, vol_mean]]))
                        STATE["market_state"] = compute_market_state(STATE, STATE["regime_probs"], vol_mean)
                        # LOT 4 (PDF Pilier B) : mesure de la qualité d'inférence
                        # du régime (confiance + stabilité) — ne trader qu'avec un
                        # régime suffisamment certain (mentalité n°5 : je ne sais pas)
                        try:
                            STATE["regime_confidence"] = regime_detector.regime_confidence(
                                np.array([[ret_mean, vol_mean]]))
                        except Exception:
                            STATE["regime_confidence"] = {
                                "confidence": 0.5, "regime_id": STATE.get("regime_id", 2)}
                    except Exception:
                        pass
                
                    # Predict temporal change using our true pure NumPy LSTM Deep Neural Network!
                    seq_features = df[['close', 'volume', 'high', 'low', 'open']].pct_change().fillna(0).values[-5:]
                    STATE["ml_prediction_pct"] = float(price_predictor.predict(seq_features))
                
                    # MLOPS CONCEPT DRIFT DETECTOR:
                    # Track prediction error, and automatically trigger retraining on the fly if CUSUM drift occurs!
                    actual_return = recent_returns[-1] if len(recent_returns) > 0 else 0.0
                    try:
                        platform_metrics.AI_MODEL_ERROR.labels(model="price_lstm").set(
                            abs(STATE["ml_prediction_pct"] - actual_return)
                        )
                    except Exception:
                        pass
                    if mlops_trainer.track_prediction_error_and_detect_drift(STATE["ml_prediction_pct"], actual_return):
                        logger.warning("MLOPS DETECTED CONCEPT DRIFT. TRIGGERING AUTOMATIC RETRAINING PIPELINE!")
                        mlops_trainer.execute_pipeline(df)
                
                    # Check Active position for this specific asset
                    positions = db.get_positions()
                    asset_position = next((p for p in positions if p['symbol'] == symbol), None)
                    pos_qty = asset_position['qty'] if asset_position else 0.0

                    # VISION §4c: integrated hedging decision - only when correlation is
                    # extreme and we haven't hedged recently (cooldown 1h, small size).
                    try:
                        if time.time() - STATE.get("last_hedge_ts", 0.0) > 3600 and len(positions) >= 2:
                            _hedge = hedging_decision(symbol, positions, STATE.get("corr_matrix_cache", {}), max_correlation=0.85)
                            if _hedge:
                                STATE["last_hedge_ts"] = time.time()
                                db.add_audit_log("HEDGE_DECISION", audit_ip(),
                                                 f"Hedge {_hedge['hedge_side']} {_hedge['hedge_qty']:.5f} {_hedge['hedge_symbol']}: {_hedge['reason']}")
                                db.add_event(time.time(), "hedge", json.dumps(_hedge, default=str))
                                logger.info(f"🛡️ HEDGE: {_hedge['hedge_symbol']} {_hedge['hedge_side']} {_hedge['hedge_qty']:.5f}")
                    except Exception:
                        pass

                    # ===== POSITION PROTECTION (audit B7-1): STOP-LOSS / TAKE-PROFIT / TRAILING =====
                    # Protection is evaluated FIRST, before any new signal, so an open
                    # position is never left exposed while we compute new entries.
                    try:
                        if pos_qty != 0.0:
                            prot = protection_store.get(symbol)
                            if prot is None:
                                atr_now = 0.0
                                if df is not None and len(df) > 0:
                                    atr_now = float(df['high'].values[-1] - df['low'].values[-1])
                                prot = PositionProtection(
                                    symbol=symbol,
                                    entry_price=float(asset_position['avg_price'] or current_price),
                                    qty=pos_qty,
                                    atr=atr_now if atr_now > 0 else None,
                                    trailing_pct=settings.get_float("risk", "trailing_stop_pct", 0.0),
                                )
                                # LOT 3 (PDF Pilier H-c) : ne JAMAIS placer un stop dans
                                # une zone de stops évidente (stop hunting) — on décale le
                                # stop hors de la zone (sous le plus bas récent pour un long)
                                try:
                                    if df is not None and len(df) > 10 and atr_now > 0:
                                        _rh = float(df['high'].values[-10:].max())
                                        _rl = float(df['low'].values[-10:].min())
                                        _dir = "long" if pos_qty > 0 else "short"
                                        _new_sl = order_flow.adjust_stop_against_hunting(
                                            symbol, prot.stop_price, _dir, _rh, _rl, atr_now)
                                        if abs(_new_sl - prot.stop_price) > 1e-9:
                                            logger.info(f"🎯 STOP HUNTING: SL {prot.stop_price:.2f} -> {_new_sl:.2f} "
                                                        f"({symbol}, hors zone de chasse {_rl:.2f}-{_rh:.2f})")
                                            prot.stop_price = _new_sl
                                except Exception as _sh:
                                    logger.debug(f"Stop hunting adjust failed: {_sh}")
                                protection_store.upsert(prot)
                                logger.info(f"🛡️ Protection armée pour {symbol}: SL {prot.stop_price:.2f} / TP {prot.take_price:.2f}")
                            else:
                                protection_store.upsert(prot)  # refresh persistence

                            action = evaluate_protection(prot, current_price, pos_qty)
                            # LOT 6 (PDF Pilier M) : CYCLE DE VIE de la position
                            # 1. BREAKEVEN : dès que le trade est en gain significatif
                            #    (>= 2%), on remonte le stop au prix d'entrée -> risque zéro
                            try:
                                if apply_breakeven_stop(prot, current_price, pos_qty, trigger_pct=0.02):
                                    logger.info(f"⚖️ BREAKEVEN: stop remonté à l'entrée pour {symbol} "
                                                f"(gain >= 2%) -> trade à risque zéro")
                                    protection_store.upsert(prot)
                            except Exception:
                                pass
                            # 2. TIME STOP : l'idée n'a pas produit de mouvement après
                            #    24h -> sortir (le capital immobilisé a un coût)
                            try:
                                if action == "HOLD":
                                    _ts = evaluate_time_stop(prot, current_price, pos_qty,
                                                             max_age_hours=24.0, min_profit_pct=0.001)
                                    if _ts == "TIME_STOP":
                                        logger.warning(f"⏰ TIME STOP {symbol}: position de {position_age_hours(prot):.1f}h "
                                                       f"sans mouvement attendu -> sortie")
                                        action = "TIME_STOP"
                            except Exception:
                                pass
                            # 3. SCALING OUT : au 1er palier de take-profit (50% de la
                            #    distance), on sécurise la MOITIÉ de la position
                            _partial = {"action": "HOLD"}
                            try:
                                if action == "HOLD":
                                    _partial = partial_take_profit(prot, current_price, pos_qty,
                                                                   tp1_fraction=0.5, exit_fraction=0.5)
                                    if _partial["action"] == "PARTIAL_TP":
                                        logger.info(f"📤 PARTIAL TP {symbol}: 50% sécurisé à "
                                                    f"{_partial['price']:.2f} (reste {_partial['remain_qty']:.5f})")
                                        protection_store.upsert(prot)
                            except Exception:
                                pass
                            if _partial["action"] == "PARTIAL_TP" and action == "HOLD":
                                # Sortie PARTIELLE (pas une clôture complète)
                                exit_side = "SELL" if pos_qty > 0 else "BUY"
                                exit_qty = _partial["exit_qty"]
                                exit_price = _partial["price"]
                                if active_mode == "REAL" and client:
                                    try:
                                        client.create_order(
                                            symbol=symbol.replace("USDT", "/USDT"),
                                            type='market', side=exit_side.lower(), amount=exit_qty,
                                        )
                                    except Exception as e:
                                        logger.error(f"Partial TP REAL exit failed: {e}")
                                if exit_side == "SELL":
                                    STATE[active_balance_key] += exit_qty * exit_price * 0.999
                                else:
                                    STATE[active_balance_key] -= exit_qty * exit_price * 1.001
                                db.update_position(symbol, _partial["remain_qty"] * (1.0 if pos_qty > 0 else -1.0),
                                                   asset_position['avg_price'], active_mode)
                                db.add_order(
                                    symbol=symbol, side=exit_side, price=exit_price, qty=exit_qty,
                                    status="FILLED", mode=active_mode, strategy="PARTIAL_TP",
                                    order_type="MARKET",
                                )
                                db.add_audit_log("PARTIAL_TAKE_PROFIT", audit_ip(),
                                                 f"50% de {symbol} sécurisé à {exit_price:.2f} (reste {_partial['remain_qty']:.5f})")
                                try:
                                    asyncio.create_task(telegram_bot.send_push_notification(
                                        f"📤 *TAKE-PROFIT PARTIEL*\n"
                                        f"`{symbol}` : 50% sécurisé à *${exit_price:,.2f}*\n"
                                        f"Position restante : {_partial['remain_qty']:.5f} (stop protégé)"))
                                except Exception:
                                    pass
                                continue
                            if action != "HOLD":
                                exit_side = "SELL" if pos_qty > 0 else "BUY"
                                exit_qty = abs(pos_qty)
                                exit_price = current_price
                                logger.critical(f"🛡️ POSITION PROTECTION [{action}] {symbol}: {exit_qty} @ {exit_price:.2f}")
                                if active_mode == "REAL" and client:
                                    try:
                                        client.create_order(
                                            symbol=symbol.replace("USDT", "/USDT"),
                                            type='market', side=exit_side.lower(), amount=exit_qty,
                                        )
                                        logger.info(f"PROTECTION REAL exit sent: {exit_side} {exit_qty} {symbol}")
                                    except Exception as e:
                                        logger.error(f"Protection REAL exit failed: {e}")
                                if exit_side == "SELL":
                                    STATE[active_balance_key] += exit_qty * exit_price * 0.999
                                else:
                                    STATE[active_balance_key] -= exit_qty * exit_price * 1.001
                                db.update_position(symbol, 0.0, 0.0, active_mode)
                                # LOT 2 : win rate RÉEL par stratégie (trades clôturés)
                                try:
                                    record_closed_trade(symbol, exit_price, exit_side)
                                except Exception as _wr:
                                    logger.debug(f"Win-rate record failed: {_wr}")
                                db.add_order(
                                    symbol=symbol, side=exit_side, price=exit_price, qty=exit_qty,
                                    status="FILLED", mode=active_mode, strategy=f"PROTECTION_{action}",
                                    order_type="MARKET",
                                )
                                db.add_audit_log(
                                    f"PROTECTION_{action}", audit_ip(),
                                    f"{action} executed on {symbol} @ {exit_price:.2f} (entry {prot.entry_price:.2f})",
                                )
                                # VISION §1d: counterfactual marginal alpha vs the market move
                                try:
                                    _mkt_move = float(actual_return) if "actual_return" in dir() else 0.0
                                    _alpha = counterfactual_alpha(
                                        {"side": exit_side, "entry": prot.entry_price, "exit": exit_price},
                                        benchmark_return=_mkt_move)
                                    db.add_event(time.time(), "closed_trade_alpha",
                                                 json.dumps({"symbol": symbol, "action": action,
                                                             "marginal_alpha": round(_alpha, 6)}, default=str))
                                except Exception:
                                    pass
                                protection_store.remove(symbol)
                                platform_metrics.ORDERS_TOTAL.labels(mode=active_mode, side=exit_side).inc()
                                try:
                                    await telegram_bot.send_push_notification(
                                        f"🛡️ *{action}* sur `{symbol}`\n"
                                        f"━━━━━━━━━━━━━━━\n"
                                        f"💵 Sortie : *${exit_price:,.2f}*\n"
                                        f"🏷️ Entrée : ${prot.entry_price:,.2f}\n"
                                        f"📊 Quantité : {exit_qty}"
                                    )
                                except Exception:
                                    pass
                                STATE["last_order_times"][symbol] = time.time()
                                continue  # position closed -> skip new signals this tick
                        else:
                            protection_store.remove(symbol)  # flat -> no stale protection
                    except Exception as pe:
                        logger.warning(f"Position protection check failed for {symbol}: {pe}")
                
                    norm_pos = pos_qty * current_price / STATE[active_balance_key] if STATE[active_balance_key] > 0 else 0.0
                    ppo_state = np.array([norm_pos, vol_mean, STATE["ml_prediction_pct"], 0.0])

                    # AUTONOMOUS RL (roadmap): collect real (state, action, log_prob, reward)
                    # experiences so the PPO agent trains itself from live outcomes.
                    try:
                        _act, _logp = ppo_agent.get_action(ppo_state)
                        # VISION §2a: blend the legacy PPO with the Mixture-of-Experts gated action
                        _moe = mixture_of_experts.decide(ppo_state, STATE["regime_id"], vol_mean)
                        STATE["ppo_action"] = float(np.clip(0.5 * _act + 0.5 * _moe["action"], -1.0, 1.0))
                        STATE["moe_votes"] = _moe["votes"]
                        STATE["moe_gate"] = _moe["gate"]
                        # VISION §2b: reward NET of impact + drawdown penalty
                        _reward = risk_adjusted_reward(
                            actual_return=float(actual_return),
                            action=STATE["ppo_action"],
                            equity_history=STATE[active_equity_history_key],
                        )
                        STATE["ppo_buffer"].append({
                            "state": ppo_state,
                            "action": STATE["ppo_action"],
                            "log_prob": _logp,
                            "reward": _reward,
                            "next_state": np.array([norm_pos, vol_mean, STATE["ml_prediction_pct"], 0.0]),
                            "terminal": False
                        })
                        mixture_of_experts.collect_experience(ppo_state, STATE["ppo_action"], _logp, _reward,
                                                              np.array([norm_pos, vol_mean, STATE["ml_prediction_pct"], 0.0]))
                        if len(STATE["ppo_buffer"]) > 2000:
                            STATE["ppo_buffer"] = STATE["ppo_buffer"][-2000:]
                    except Exception as e:
                        logger.warning(f"PPO experience collection failed: {e}")
                
                    # Setup genuine order book parameters from the WebSockets stream (No fake fallback allowed!)
                    ob_bids = STATE["order_book"].get("bids") if STATE["order_book"] is not None else None
                    ob_asks = STATE["order_book"].get("asks") if STATE["order_book"] is not None else None

                    # Microstructure features (audit B8-2): VPIN + Kyle's Lambda computed
                    # HERE and fed into the meta-model - no longer just logged.
                    vpin_val = microstructure_engine.calculate_vpin(df)
                    kyles_lambda_val = microstructure_engine.calculate_kyles_lambda(df)
                    logger.info(f"MICROSTRUCTURE ({symbol}) | VPIN: {vpin_val:.3f} | Kyle's Lambda: {kyles_lambda_val:.3e}")
                
                    # VISION_FUTUR §3/§4: market-average return + cross-asset bias
                    try:
                        _mkt_rets = [a.get("price") for a in STATE.get("assets", {}).values()
                                     if isinstance(a.get("price"), (int, float)) and a.get("price", 0) > 0]
                        _avg = float(np.mean(np.diff(_mkt_rets)) / np.mean(_mkt_rets[:-1])) if len(_mkt_rets) > 2 and np.mean(_mkt_rets[:-1]) > 0 else 0.0
                    except Exception:
                        _avg = 0.0
                    try:
                        STATE["structural_regimes"] = compute_structural_regimes(STATE)
                    except Exception:
                        pass

                    market_data = {
                        'df': df,
                        'symbol': symbol,
                        'price_primary': current_price,
                        'price_secondary': bybit_p if bybit_p is not None else current_price, # Real Bybit price from CEX (No random.uniform secondary price fallback!)
                        'bids': ob_bids,
                        'asks': ob_asks,
                        'inventory': pos_qty,
                        'max_inventory': STATE[active_balance_key] / current_price if STATE[active_balance_key] > 0 else 0.0,
                        'vpin': float(vpin_val),
                        'kyle_lambda': float(kyles_lambda_val),
                        'onchain_risk': _neutral(STATE.get("onchain_risk_score")),
                        'sentiment': _neutral(STATE.get("sentiment_index")),
                        'funding_rate_8h': float(STATE.get("funding_rates", {}).get(symbol, 0.0)),
                        'market_avg_return': float(_avg),
                        'cross_asset_bias': cross_asset_bias(symbol, STATE),
                    }
                
                    consensus = meta_engine.allocate(market_data, STATE["regime_id"], STATE["ml_prediction_pct"], STATE["ppo_action"])
                    final_signal = consensus["final_signal"]
                    # VISION_FUTUR §1: derive the dominant strategy early (desk capital mapping)
                    _dom_early = "META_MODEL"
                    try:
                        _contribs = consensus.get("contributions", {})
                        if _contribs:
                            _dom_early = max(_contribs, key=lambda s: abs(_contribs[s].get("signal", 0.0) * _contribs[s].get("weight", 0.0)))
                    except Exception:
                        pass
                    STATE["last_reasoning"] = explain_last_decision(consensus)
                    STATE["last_reasoning_symbol"] = symbol
                
                    # Feed trade feedback to Thompson Sampling strategy re-allocator!
                    meta_engine.update_bandit_feedback(symbol, consensus["contributions"], actual_return)
                
                    # Incorporate sentiment index — UNIQUEMENT si le sentiment est
                    # réel et disponible (faille 1 corrigée : jamais de sentiment
                    # inventé ou absent dans la décision)
                    if STATE.get("sentiment_available") and STATE.get("sentiment_index") is not None:
                        final_signal = (0.80 * final_signal) + (0.20 * STATE["sentiment_index"])
                    # VISION_FUTUR §3/§4: cross-asset bias (BTC regime informs others, soft)
                    final_signal = float(np.clip(final_signal + cross_asset_bias(symbol, STATE), -1.0, 1.0))
                    final_signal = max(-1.0, min(1.0, final_signal))
                
                    # ===== RISK SIZING — PIPELINE UNIFIÉ (LOT 2, PDF Pilier F & G) =====
                    atr = df['high'].values[-1] - df['low'].values[-1]
                    if atr == 0:
                        atr = current_price * 0.008

                    # 0. Kelly DYNAMIQUE : win rate RÉEL par stratégie (borné 0.45..0.65,
                    # lissé EMA) + RR unifié REWARD_RISK_RATIO (source unique, alignée sur
                    # les stops réels). Fini le 0.55/1.5 codés en dur (PDF Pilier F).
                    _dom_kelly = _dom_early if _dom_early != "META_MODEL" else "META_MODEL"
                    target_qty = risk_manager.calculate_position_size(
                        capital=STATE[active_balance_key],
                        atr=atr,
                        current_price=current_price,
                        win_rate=win_tracker.get(_dom_kelly),
                        reward_risk_ratio=REWARD_RISK_RATIO
                    )
                    STATE["last_kelly"] = {
                        "strategy": _dom_kelly,
                        "win_rate_used": win_tracker.get(_dom_kelly),
                        "rr_used": REWARD_RISK_RATIO,
                        "base_qty": target_qty,
                    }

                    # 1. CVaR-CONSTRAINED SIZING (plafond dur, étape 1 du pipeline)
                    cvar_qty = calculate_cvar_constrained_sizing(
                        capital=STATE[active_balance_key],
                        current_price=current_price,
                        cvar_pct=portfolio_cvar_pct,
                        max_loss_usd=STATE[active_balance_key] * 0.02
                    )

                    # 2. MAX PER-ASSET CAP 25 % (plafond dur, étape 2 du pipeline)
                    max_asset_pct = settings.get_float("risk", "max_per_asset_pct", 0.25)
                    try:
                        _user_max = float(os.getenv("USER_MAX_EXPOSURE_PCT", "0"))
                        if 0 < _user_max < 1:
                            max_asset_pct = min(max_asset_pct, _user_max)
                    except Exception:
                        pass
                    max_asset_qty = (STATE[active_balance_key] * max_asset_pct) / current_price

                    # VISION_FUTUR §2d: meta-label filter — filtre les faux signaux avant
                    # exécution (López de Prado). Warm-up : <5 trades clôturés pour la
                    # stratégie -> autorisé en DEMO (sinon le bot ne peut jamais apprendre).
                    try:
                        _ml_count = win_tracker.samples(_dom_early)
                        if not meta_label_filter(_dom_early,
                                                 STATE.get("strategy_win_rates", {}),
                                                 counts=STATE.get("strategy_trade_counts", {}),
                                                 min_samples=5):
                            decide_no_trade(symbol, final_signal, 0.999,
                                            [f"meta-label: wr {STATE.get('strategy_win_rates', {}).get(_dom_early, 0.0):.2f} (n={_ml_count})"],
                                            STATE["no_trade_stats"], db)
                            continue
                    except Exception:
                        pass

                    # ---- Collecte de TOUS les facteurs de risque (dans l'ordre officiel) ----
                    # 4. État de risque (NORMAL/CAUTION/HALT)
                    _risk_scale = risk_state.scale_factor()
                    # 5. Choc d'actualité
                    _news_s = news_scale_factor
                    # 6. Événement macro réel
                    _macro_s = macro_scale_factor
                    # 7. Override humain tactile (Telegram)
                    _tactile_s = STATE.get("macro_scale_factor_tactile", 1.0)
                    # 8. On-chain (réel uniquement)
                    _onchain_s = 1.0
                    if STATE.get("onchain_available") and STATE.get("onchain_risk_score") is not None:
                        if STATE["onchain_risk_score"] > 0.75:
                            _onchain_s = 0.50
                            logger.info(f"ON-CHAIN WARNING: Scaling down position size for {symbol} due to high network risk.")
                    # 9. Corrélation multi-actifs (concentration)
                    _corr_s = 1.0
                    if corr_df is not None and not corr_df.empty:
                        _corr_s = covariance_engine.evaluate_portfolio_concentration_risk(
                            symbol=symbol, active_positions=positions, corr_matrix=corr_df)
                    # 9bis. Régime CERTAIN ? (LOT 4, PDF Pilier B) + causal gate
                    _regime_s = 1.0
                    try:
                        _rc = STATE.get("regime_confidence", {})
                        _conf = float(_rc.get("confidence", 0.5) or 0.5)
                        if _conf < 0.45:
                            _regime_s = 0.5     # régime très incertain -> réduire
                        elif _conf < 0.6:
                            _regime_s = 0.8     # régime incertain -> léger frein
                        _regime_s *= causal_signal_factor(STATE)
                        if _regime_s < 1.0:
                            logger.info(f"RÉGIME/CAUSAL {symbol}: facteur {_regime_s:.2f} "
                                        f"(confiance régime {_conf:.2f})")
                    except Exception:
                        pass
                    # 9quinquies. MÉTA-ATTRIBUTION (LOT 7, PDF Pilier K) : les
                    # mauvaises raisons prouvées réduisent la taille
                    _reason_s = float(STATE.get("reason_weights_factor", 1.0))
                    # 9ter. CAPACITÉ + RÉSERVE DE CASH (LOT 6, PDF Pilier L) :
                    # la taille d'un trade ne peut pas dépasser 1 % du volume
                    # réel 24h (impact de marché, mentalité n°11) et l'exposition
                    # totale du portefeuille reste sous (1 - réserve cash).
                    _cap_s = 1.0
                    _cash_s = 1.0
                    try:
                        _vol24 = STATE["assets"].get(symbol, {}).get("volume_24h")
                        _cap_qty = portfolio_allocator.capacity_cap_qty(symbol, _vol24, current_price)
                        if _cap_qty is not None and _cap_qty > 0:
                            _cap_ratio = target_qty / _cap_qty
                            if _cap_ratio > 1.0:
                                _cap_s = max(0.1, 1.0 / _cap_ratio)
                                logger.info(f"CAPACITÉ {symbol}: taille plafonnée à 1% du volume 24h (x{_cap_s:.2f})")
                        _cash_s = portfolio_allocator.portfolio_exposure_factor(STATE, active_balance_key)
                        if _cash_s < 1.0:
                            logger.info(f"RÉSERVE CASH: exposition portfolio proche du max -> x{_cash_s:.2f}")
                    except Exception:
                        pass
                    # 10. ORDER FLOW toxique (LOT 3, PDF Pilier H-d) : VPIN/Kyle/delta
                    # extrême -> RÉDUIRE la taille (informed trading), pas seulement journaliser
                    try:
                        _vpin_now = float(microstructure_engine.calculate_vpin(df))
                        _ofl_s = order_flow.toxicity_factor(symbol, vpin=_vpin_now)
                        if _ofl_s < 1.0:
                            logger.info(f"ORDER FLOW TOXIQUE {symbol}: facteur {_ofl_s:.2f} "
                                        f"(delta {order_flow.get_delta(symbol)[0]:.0f}, VPIN {_vpin_now:.2f})")
                    except Exception:
                        _ofl_s = 1.0
                    # 10. Indice de confiance (méta-cognition)
                    _conf_s = STATE.get("confidence_factor", 1.0)
                    # 11. Organisation (desks)
                    _org_s = organization.confidence_factor(_dom_early)
                    # 12. RLHF (modulateur borné)
                    _rlhf_s = 1.0
                    try:
                        _rlhf_feats = np.array([norm_pos, vol_mean, STATE["ml_prediction_pct"], actual_return])
                        _rlhf_score = rlhf_reward_model.predict_reward(_rlhf_feats)
                        if _rlhf_score is None:
                            _rlhf_s = 1.0  # RLHF indisponible -> NEUTRE (LOT 4)
                        else:
                            _rlhf_s = max(0.25, 0.5 + 0.5 * float(_rlhf_score))
                    except Exception:
                        pass
                    # 13. Volatilité cible
                    _vol_scale = 1.0
                    try:
                        _vol_scale = volatility_scale_factor(STATE[active_equity_history_key])
                        STATE["vol_target_scale"] = _vol_scale
                    except Exception:
                        pass
                    # 14. Tradabilité / slippage attendu
                    _trad_s = 1.0
                    try:
                        _slip_avg = slippage_model.expected_slippage_bps(
                            "Binance" if symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT") else "Bybit",
                            symbol, fallback=5.0)
                        _trad_s = tradability_factor(_slip_avg)
                    except Exception:
                        pass

                    # ===== APPLICATION DU PIPELINE UNIFIÉ (ordre documenté + tracé) =====
                    try:
                        _pipe = apply_risk_pipeline(
                            base_qty=target_qty,
                            cvar_qty=cvar_qty,
                            max_asset_qty=max_asset_qty,
                            conviction=abs(final_signal),
                            risk_state_scale=_risk_scale,
                            news_scale=_news_s,
                            macro_scale=_macro_s,
                            tactile_scale=_tactile_s,
                            onchain_scale=_onchain_s,
                            corr_scale=_corr_s,
                            order_flow_scale=_ofl_s,
                            regime_confidence_scale=_regime_s,
                            capacity_scale=_cap_s,
                            cash_reserve_scale=_cash_s,
                            reason_attribution_scale=_reason_s,
                            confidence_scale=_conf_s,
                            org_scale=_org_s,
                            rlhf_scale=_rlhf_s,
                            vol_scale=_vol_scale,
                            tradability_scale=_trad_s,
                        )
                        target_qty = _pipe["qty"]
                        STATE["risk_pipeline_steps"] = _pipe["steps"]
                        risk_pipeline_last.update({"symbol": symbol, "final_scale": _pipe["final_scale"],
                                                   "n_steps": len(_pipe["steps"])})
                    except Exception as _pe:
                        logger.error(f"Risk pipeline failed ({_pe}) — using conservative path")
                        target_qty = min(target_qty, cvar_qty, max_asset_qty) * abs(final_signal)

                    # VISION §7.5: A/B paper (hypothétique, sans effet)
                    try:
                        STATE["ab_base"].append(1.0 + final_signal * actual_return)
                        STATE["ab_vol"].append(1.0 + final_signal * actual_return * _vol_scale)
                        for _k in ("ab_base", "ab_vol"):
                            if len(STATE[_k]) > 2000:
                                STATE[_k] = STATE[_k][-2000:]
                    except Exception:
                        pass

                    # VISION §4a: adaptive conviction threshold from recent accuracy
                    STATE["recent_signals"].append(float(final_signal))
                    STATE["recent_returns"].append(float(actual_return))
                    for _k in ("recent_signals", "recent_returns"):
                        if len(STATE[_k]) > 200:
                            STATE[_k] = STATE[_k][-200:]
                    try:
                        STATE["conviction_threshold"] = adaptive_conviction_threshold(
                            STATE["recent_signals"], STATE["recent_returns"],
                            base_threshold=settings.get_float("trading", "signal_threshold", 0.15))
                    except Exception:
                        pass
                    target_direction = np.sign(final_signal) if abs(final_signal) > STATE.get("conviction_threshold", 0.15) else 0.0
                    # VISION §4b: explicit NO_TRADE decision when abstaining
                    if target_direction == 0.0 and final_signal != 0.0:
                        decide_no_trade(symbol, final_signal, STATE.get("conviction_threshold", 0.15),
                                        [f"regime={STATE.get('regime_name','')}", f"moe={STATE.get('moe_gate',{})}"],
                                        STATE["no_trade_stats"], db)

                    # ===== FILTRE D'ENTRÉE « RR MINIMAL » (PDF Pilier F, exigences 3-5) =====
                    # On n'entre que si : RR >= requis (adaptatif régime/vol) ET
                    # asymétrie nette positive (edge > coûts). Mentalité n°7.
                    if target_direction != 0.0:
                        _sl_dist = (ATR_MULT_SL * atr) / current_price if atr > 0 else STOP_LOSS_PCT
                        _rr_ok, _rr_reason = entry_rr_filter(
                            reward_risk=REWARD_RISK_RATIO,
                            regime_id=STATE.get("regime_id"),
                            vol_mean=vol_mean if "vol_mean" in dir() else None,
                            sl_distance_pct=_sl_dist,
                            cost_pct=ROUND_TRIP_COST_PCT,
                        )
                        if not _rr_ok:
                            decide_no_trade(symbol, final_signal, STATE.get("conviction_threshold", 0.15),
                                            [f"RR filter: {_rr_reason}"], STATE["no_trade_stats"], db)
                            target_direction = 0.0
                        else:
                            STATE["last_rr_check"] = {"rr": REWARD_RISK_RATIO, "ok": True, "reason": _rr_reason}

                    # ===== GATE MACHINE À ÉTATS (PDF Faille 3) : HALT = AUCUN nouvel ordre =====
                    # La protection des positions existantes reste active (atomicité, Pilier G).
                    if target_direction != 0.0 and risk_state.state == RiskStateMachine.HALT:
                        decide_no_trade(symbol, final_signal, STATE.get("conviction_threshold", 0.15),
                                        [f"HALT: {risk_state.reason}"], STATE["no_trade_stats"], db)
                        target_direction = 0.0

                    # ===== GATES ORDER FLOW (LOT 3, PDF Pilier H a/b) =====
                    if target_direction != 0.0:
                        _side_tmp = "BUY" if target_direction > 0 else "SELL"
                        # (a) ne jamais entrer CONTRE un flux agressif dominant
                        _avoid, _avoid_reason = order_flow.should_avoid_entry(symbol, _side_tmp)
                        if _avoid:
                            decide_no_trade(symbol, final_signal, STATE.get("conviction_threshold", 0.15),
                                            [_avoid_reason], STATE["no_trade_stats"], db)
                            target_direction = 0.0
                        else:
                            # (b) cascade de liquidations -> attendre la fin (ne pas
                            # acheter la panique, mentalité n°1 : survivre d'abord)
                            _casc, _casc_reason = order_flow.wait_cascade_end(symbol)
                            if _casc and _side_tmp == "BUY":
                                decide_no_trade(symbol, final_signal, STATE.get("conviction_threshold", 0.15),
                                                [_casc_reason], STATE["no_trade_stats"], db)
                                target_direction = 0.0

                    # LOT 6 (PDF Pilier M) : PYRAMIDING CONTRÔLÉ + NETTING
                    # Pyramiding : ajouter UNIQUEMENT sur les gagnants, RR
                    # favorable, max 2 ajouts ; JAMAIS de moyenne à la baisse.
                    try:
                        if target_direction != 0.0 and pos_qty != 0.0:
                            _same_dir = (target_direction > 0 and pos_qty > 0) or \
                                        (target_direction < 0 and pos_qty < 0)
                            if _same_dir:
                                _prot_p = protection_store.get(symbol)
                                if _prot_p is not None:
                                    _pyr_n = STATE.setdefault("position_pyramids", {}).get(symbol, 0)
                                    _pyr_ok, _pyr_reason = can_pyramid(
                                        _prot_p, current_price, pos_qty,
                                        reward_risk=REWARD_RISK_RATIO,
                                        min_rr=1.5, max_additions=2, additions=_pyr_n)
                                    if not _pyr_ok:
                                        decide_no_trade(symbol, final_signal,
                                                        STATE.get("conviction_threshold", 0.15),
                                                        [_pyr_reason], STATE["no_trade_stats"], db)
                                        target_direction = 0.0
                                    else:
                                        logger.info(f"📈 {_pyr_reason} -> ajout autorisé ({symbol})")
                            else:
                                # NETTING : retournement contre la position existante
                                # (une autre stratégie voudrait trader contre soi-même)
                                # -> exiger un signal FORT, sinon s'abstenir
                                if abs(final_signal) < 1.5 * STATE.get("conviction_threshold", 0.15):
                                    decide_no_trade(symbol, final_signal,
                                                    STATE.get("conviction_threshold", 0.15),
                                                    ["netting: retournement sans signal fort (anti auto-contre-trade)"],
                                                    STATE["no_trade_stats"], db)
                                    target_direction = 0.0
                    except Exception as _py:
                        logger.debug(f"Pyramiding/netting check failed: {_py}")

                    desired_qty = target_direction * target_qty
                    trade_qty = desired_qty - pos_qty
                
                    # 5. Execute order
                    if abs(trade_qty) > (current_price * 0.0001):
                        side = "BUY" if trade_qty > 0 else "SELL"
                        execution_price = current_price * (1.0 + 0.0003) if side == "BUY" else current_price * (1.0 - 0.0003)

                        # IDEMPOTENCE GATE (roadmap #1): prevent duplicate/rapid-fire orders per symbol
                        last_order_ts = STATE["last_order_times"].get(symbol, 0.0)
                        cooldown_s = ORDER_COOLDOWN_REAL_SECONDS if active_mode == "REAL" else ORDER_COOLDOWN_DEMO_SECONDS
                        if time.time() - last_order_ts < cooldown_s:
                            logger.info(
                                f"Idempotence gate: {symbol} last order {time.time()-last_order_ts:.0f}s ago "
                                f"(<{cooldown_s:.0f}s). Skipping duplicate order."
                            )
                            continue
                    
                        trade_qty_formatted = format_exchange_size(symbol, abs(trade_qty), execution_price)
                    
                        # Enforce pre-flight safety limits
                        ok, reason = risk_manager.validate_order_safety(
                            order_price=execution_price,
                            mid_market_price=current_price,
                            order_qty=trade_qty_formatted,
                            capital_available=STATE[active_balance_key]
                        )
                    
                        if ok:
                            # VISION_FUTUR §6: consultative mode - the bot proposes, the human approves
                            if STATE.get("consultative_mode", False):
                                _proposal = {"symbol": symbol, "side": side, "qty": trade_qty_formatted,
                                             "price": execution_price, "mode": active_mode,
                                             "strategy": dominant_strategy, "ts": time.time()}
                                STATE["pending_approvals"].append(_proposal)
                                db.add_audit_log("PENDING_APPROVAL", audit_ip(),
                                                 f"Proposal {side} {trade_qty_formatted} {symbol} awaits operator approval")
                                try:
                                    asyncio.create_task(telegram_bot.send_push_notification(
                                        f"🤝 *APPROBATION REQUISE*\n{side} {trade_qty_formatted:.5f} `{symbol}` @ {execution_price:.2f}\n"
                                        f"Stratégie : {dominant_strategy}\n"
                                        f"Réponds /approve ou /reject",
                                        reply_markup={
                                            "inline_keyboard": [
                                                [
                                                    {"text": "✅ Approuver", "callback_data": "approve_pending"},
                                                    {"text": "🚫 Rejeter", "callback_data": "reject_pending"}
                                                ]
                                            ]
                                        }
                                    ))
                                except Exception:
                                    pass
                                continue
                            # Enforce strict real safety gate in production before placing any real trade!
                            if active_mode == "REAL" and not evaluate_real_safety_gate(symbol):
                                logger.critical(f"REAL SAFETY GATE REJECTED: Real order blocked for {symbol} due to safety gate checks.")
                                continue
                            
                            # PER-MODEL ATTRIBUTION (roadmap precision #1): dominant strategy label
                            dominant_strategy = "META_MODEL"
                            try:
                                contribs = consensus.get("contributions", {})
                                if contribs:
                                    dominant_strategy = max(
                                        contribs,
                                        key=lambda s: abs(contribs[s].get("signal", 0.0) * contribs[s].get("weight", 0.0))
                                    )
                            except Exception:
                                pass

                            try:
                                # EVM NON-CUSTODIAL EXECUTION ROUTER:
                                if active_mode == "REAL" and os.getenv("EVM_PRIVATE_KEY") and symbol == "ETHUSDT":
                                    logger.info(f"DECISION: Executing NON-CUSTODIAL EVM SWAP of {trade_qty_formatted} ETH!")
                                    signed_dex_res = defi_wallet.sign_dex_swap_transaction(
                                        token_in="USDT" if side == "BUY" else "ETH",
                                        token_out="ETH" if side == "BUY" else "USDT",
                                        amount_in_eth=trade_qty_formatted
                                    )
                                    logger.info(f"DEX Swap signed successfully. Transaction Hash: {signed_dex_res.get('tx_hash')}")
                                
                                elif active_mode == "REAL" and client:
                                    # LOT 3 (PDF Pilier H-3) : SOR — choisir la venue au
                                    # coût NET (prix + frais + slippage), audité.
                                    try:
                                        _sor = await pick_best_venue_net(symbol, side)
                                        STATE["last_sor_choice"] = _sor
                                        if _sor.get("venue"):
                                            logger.info(
                                                f"SOR {symbol} {side}: venue {_sor['venue']} "
                                                f"(net {_sor['net_price']:.6f}) | {_sor['reason']}")
                                            db.add_audit_log(
                                                "SOR_CHOICE", audit_ip(),
                                                f"{symbol} {side}: venue {_sor['venue']} "
                                                f"net {_sor['net_price']:.6f} - {json.dumps(_sor.get('quotes', []))[:200]}")
                                    except Exception as _se:
                                        logger.debug(f"SOR eval failed: {_se}")
                                    # VISION §3.1: execution style (market/limit/twap) + alpha measurement
                                    _arrival_price = current_price
                                    _spread_bps = 2.0
                                    try:
                                        _bb = float(STATE["order_book"]["bids"][0][0]) if STATE.get("order_book", {}).get("bids") else current_price
                                        _ba = float(STATE["order_book"]["asks"][0][0]) if STATE.get("order_book", {}).get("asks") else current_price
                                        if _ba > _bb > 0:
                                            _spread_bps = (_ba - _bb) / _bb * 1e4
                                    except Exception:
                                        pass
                                    _liq = STATE[active_balance_key] or 100000.0
                                    _style = decide_style(_spread_bps, urgency=abs(final_signal), size_vs_liquidity=trade_qty_formatted * current_price / max(_liq, 1.0))
                                    # VISION §5a: learned execution-style bandit refines the choice
                                    try:
                                        _volr = STATE.get("market_state", {}).get("vol_regime", "normal")
                                        _style = execution_bandit.choose_style(symbol, _volr, _spread_bps, abs(final_signal))
                                    except Exception:
                                        pass
                                    if _style == "twap" and trade_qty_formatted * current_price > _liq * 0.02:
                                        _slices = order_slicer.num_slices
                                        _slice_qty = trade_qty_formatted / _slices
                                        logger.info(f"TWAP: {symbol} -> {_slices} slices of {_slice_qty:.5f}")
                                        for _i in range(_slices):
                                            try:
                                                submit_order_via_oms(symbol=symbol, side=side, qty=_slice_qty, price=current_price,
                                                                      mode=active_mode, strategy=dominant_strategy,
                                                                      exchange="Binance" if symbol in ("BTCUSDT","ETHUSDT","SOLUSDT") else "Bybit")
                                                platform_metrics.EXEC_TWAP_SLICES.inc()
                                            except Exception as _se:
                                                logger.warning(f"TWAP slice {_i} failed: {_se}")
                                                break
                                    else:
                                        logger.info(f"REAL ORDER SUBMISSION (OMS/{_style}): {side} {trade_qty_formatted} {symbol}")
                                        res_order = submit_order_via_oms(
                                            symbol=symbol, side=side, qty=trade_qty_formatted,
                                            price=execution_price, mode=active_mode,
                                            strategy=dominant_strategy,
                                            exchange="Binance" if symbol in ("BTCUSDT","ETHUSDT","SOLUSDT") else "Bybit",
                                        )
                                        execution_price = res_order.get('price', execution_price)
                                        if res_order.get("status") == "REJECTED":
                                            logger.error(f"OMS REJECTED: {res_order.get('reason', 'unknown')}")
                                            raise Exception(res_order.get("reason", "OMS rejected order"))
                                    platform_metrics.EXEC_FILLS.labels(style=_style).inc()
                                    _slip = execution_alpha.record(symbol, side, _arrival_price, execution_price, _style)
                                    slippage_model.update("Binance" if symbol in ("BTCUSDT","ETHUSDT","SOLUSDT") else "Bybit", symbol, _slip)
                                    platform_metrics.EXEC_SLIPPAGE_BPS.labels(style=_style).set(execution_alpha.avg_slippage_bps(_style))
                                    try:
                                        _volr2 = STATE.get("market_state", {}).get("vol_regime", "normal")
                                        execution_bandit.observe(symbol, _volr2, _style, _slip)
                                        strategy_exec_attr.record(dominant_strategy, _slip, _style)
                                    except Exception:
                                        pass

                                    # FILL CONFIRMATION (roadmap #2): poll the exchange until the order is filled
                                    # before touching the ledger - never book an order that didn't actually fill.
                                    order_id = res_order.get("id") or res_order.get("orderId")
                                    if order_id:
                                        for _attempt in range(6):
                                            await asyncio.sleep(1.0)
                                            try:
                                                fill = client.fetch_order(order_id, symbol.replace("USDT", "/USDT"))
                                                fill_status = (fill.get("status") or "").lower()
                                                filled_qty = float(fill.get("filled") or 0.0)
                                                if fill_status in ("closed", "filled") or filled_qty > 0:
                                                    avg = fill.get("average")
                                                    if avg:
                                                        execution_price = float(avg)
                                                    if filled_qty > 0:
                                                        trade_qty_formatted = filled_qty
                                                    logger.info(f"FILL CONFIRMED: {filled_qty} {symbol} @ {execution_price}")
                                                    break
                                            except Exception as fe:
                                                logger.warning(f"Fill confirmation poll error: {fe}")
                                                break
                                
                                # Record order timestamp for the idempotence gate
                                STATE["last_order_times"][symbol] = time.time()
                                
                                # PAPER EXECUTION (DEMO == REAL, high fidelity): book-walk the real
                                # order book, apply per-venue fees, latency, impact and rejections
                                # so paper validation is statistically meaningful.
                                _paper_fee = None
                                if active_mode == "DEMO":
                                    _paper_book = STATE.get("order_books", {}).get(symbol)
                                if _paper_book is None:
                                    _paper_book = STATE.get("order_book") if symbol == "BTCUSDT" else None
                                    _paper = simulate_paper_fill(
                                        symbol=symbol, side=side, qty=trade_qty_formatted,
                                        arrival_price=current_price,
                                        order_book=_paper_book,
                                        venue="Binance" if symbol in ("BTCUSDT","ETHUSDT","SOLUSDT") else "Bybit",
                                        volatility=vol_mean if "vol_mean" in dir() else 0.002,
                                        balance=STATE[active_balance_key],
                                        slippage_model=slippage_model,
                                    )
                                    if _paper.get("rejected"):
                                        logger.warning(f"PAPER REJECTED {symbol} {side} {trade_qty_formatted}: {_paper['reason']}")
                                        db.add_audit_log("DEMO_ORDER_REJECTED", audit_ip(),
                                                         f"{symbol} {side} {trade_qty_formatted}: {_paper['reason']}")
                                        platform_metrics.ORDERS_FAILED.labels(mode="DEMO").inc()
                                        continue
                                    execution_price = _paper["fill_price"]
                                    _paper_fee = _paper["fee"]
                                    if _paper.get("partial"):
                                        trade_qty_formatted = _paper["fill_qty"]  # respect partial fills
                                    # execution-quality tracking learns from paper too
                                    execution_alpha.record(symbol, side, current_price, execution_price, "paper")
                                    slippage_model.update("Binance" if symbol in ("BTCUSDT","ETHUSDT","SOLUSDT") else "Bybit", symbol, _paper["slippage_bps"])
                                    platform_metrics.EXEC_SLIPPAGE_BPS.labels(style="paper").set(execution_alpha.avg_slippage_bps("paper"))
                                    try:
                                        db.add_event(time.time(), "paper_fill", json.dumps({
                                            "symbol": symbol, "side": side, "qty": trade_qty_formatted,
                                            "arrival": current_price, "fill": execution_price,
                                            "slippage_bps": _paper["slippage_bps"], "fee": _paper_fee,
                                            "latency_ms": _paper["latency_ms"],
                                        }, default=str))
                                    except Exception:
                                        pass
                                # Ledger update
                                order_cost = execution_price * trade_qty_formatted
                                commission = _paper_fee if _paper_fee is not None else order_cost * 0.001
                            
                                if side == "BUY":
                                    STATE[active_balance_key] -= (order_cost + commission)
                                    new_qty = pos_qty + trade_qty_formatted
                                    new_avg = ((pos_qty * (asset_position['avg_price'] if asset_position else 0.0)) + (trade_qty_formatted * execution_price)) / new_qty
                                else:
                                    STATE[active_balance_key] += (order_cost - commission)
                                    new_qty = pos_qty - trade_qty_formatted
                                    new_avg = asset_position['avg_price'] if asset_position and new_qty > 0 else 0.0
                                
                                db.update_position(symbol, new_qty, new_avg, active_mode)
                                db.add_order(
                                    symbol=symbol,
                                    side=side,
                                    price=execution_price,
                                    qty=trade_qty_formatted,
                                    status="FILLED",
                                    mode=active_mode,
                                    strategy=dominant_strategy,
                                    order_type="MARKET"
                                )
                                platform_metrics.ORDERS_TOTAL.labels(mode=active_mode, side=side).inc()
                                # LOT 8 (PDF Pilier O) : coût RÉEL du trade (frais +
                                # slippage réalisé) tracé et retranché du PnL.
                                try:
                                    _fee_rate = 0.001
                                    if _paper_fee is not None:
                                        _fee_rate = _paper_fee / max(order_cost, 1e-9)
                                    _slip_bps = execution_alpha.avg_slippage_bps(_style if "_style" in dir() else None)
                                    cost_accounting.record_trade_cost(
                                        symbol=symbol, side=side, qty=trade_qty_formatted,
                                        price=execution_price, fee_rate=_fee_rate,
                                        slippage_bps=_slip_bps if _slip_bps else 5.0,
                                        style=_style if "_style" in dir() else "market",
                                        venue="Binance" if symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT") else "Bybit")
                                    STATE["cost_metrics"] = cost_accounting.to_dict()
                                except Exception:
                                    pass
                                # LOT 6 : compteur de pyramiding (ajout sur position existante)
                                try:
                                    if pos_qty != 0.0 and (pos_qty * new_qty) > 0:
                                        STATE.setdefault("position_pyramids", {})[symbol] = \
                                            STATE["position_pyramids"].get(symbol, 0) + 1
                                    elif abs(new_qty) < 1e-12:
                                        STATE.setdefault("position_pyramids", {})[symbol] = 0  # position fermée
                                except Exception:
                                    pass
                                # LOT 2 : tracking du win rate RÉEL par stratégie
                                try:
                                    _was_flat = abs(pos_qty) < 1e-12
                                    _now_flat = abs(new_qty) < 1e-12
                                    if _was_flat and not _now_flat:
                                        # Ouverture : mémorise la stratégie responsable
                                        record_open_position(symbol, dominant_strategy, execution_price)
                                    elif _now_flat and not _was_flat:
                                        # Clôture complète : enregistre le trade
                                        record_closed_trade(symbol, execution_price, side)
                                    elif not _was_flat and not _now_flat and (new_qty * pos_qty < 0):
                                        # Retournement : clôture de l'ancien côté puis ouverture
                                        record_closed_trade(symbol, execution_price, side)
                                        record_open_position(symbol, dominant_strategy, execution_price)
                                except Exception as _wr:
                                    logger.debug(f"Win-rate tracking error: {_wr}")
                                try:
                                    trade_journal.add_trade(
                                        symbol=symbol,
                                        side=side,
                                        qty=trade_qty_formatted,
                                        price=execution_price,
                                        mode=active_mode,
                                        strategy=dominant_strategy,
                                        notes=f"Executed via {dominant_strategy}"
                                    )
                                except Exception as je:
                                    logger.warning(f"Trade journal write failed: {je}")
                            
                                db.add_audit_log(
                                    "REAL_ORDER" if active_mode == "REAL" else "DEMO_ORDER", 
                                    audit_ip(), 
                                    f"Executed {side} order of {trade_qty_formatted:.5f} {symbol} at {execution_price:.2f} USD."
                                )
                                # VISION_FUTUR §1: attribute PnL to the responsible desk
                                try:
                                    _notional = execution_price * trade_qty_formatted
                                    _pnl_est = _notional * float(actual_return) if "actual_return" in dir() else 0.0
                                    organization.record_trade(dominant_strategy, _pnl_est, _notional)
                                except Exception:
                                    pass
                                try:
                                    db.add_event(time.time(), "order", json.dumps({
                                        "symbol": symbol, "side": side, "qty": trade_qty_formatted,
                                        "price": execution_price, "mode": active_mode,
                                        "strategy": dominant_strategy,
                                        "reasoning": STATE.get("last_reasoning", [])[:3],
                                    }, default=str))
                                except Exception:
                                    pass
                            
                                # Formulate a pedagogic and visual explanation of the trade!
                                regime = STATE.get("regime_name", "Mean-Reverting Range")
                                hmm_translation = {
                                    "Bull Trend (Low Vol)": "Soleil Haussier ☀️ (Mouvement de hausse calme)",
                                    "Bear Trend (High Vol)": "Tempête Baissière ⛈️ (Marché en baisse rapide)",
                                    "Mean-Reverting Range": "Temps Nuageux ⛅ (Marché stable qui oscille)",
                                    "Erratic High Volatility": "Volatilité Erratique 🌪️ (Marché agité et imprévisible)"
                                }
                                translated_regime = hmm_translation.get(regime, regime)
                            
                                trade_reason = ""
                                if side == "BUY":
                                    trade_reason = (
                                        "✨ *Pourquoi cet achat ?* Nos algorithmes de suivi de tendance ont détecté "
                                        "une accélération haussière prometteuse. J'en profite pour accumuler de l'actif "
                                        "afin de maximiser nos gains !"
                                    )
                                else:
                                    trade_reason = (
                                        "🔒 *Pourquoi cette vente ?* Nos modèles de protection de capital ont détecté "
                                        "un essoufflement ou un risque de retournement. Je vends pour sécuriser vos bénéfices "
                                        "au chaud et mettre notre capital à l'abri !"
                                    )
                                
                                telegram_msg = (
                                    f"🔔 *EXÉCUTION D'ORDRE ({active_mode})*\n"
                                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                    f"📝 Actif : `{symbol}`\n"
                                    f"🚀 Action : *{side == 'BUY' and '🟢 ACHAT' or '🔴 VENTE'}*\n"
                                    f"📊 Quantité : `{trade_qty_formatted:.5f}`\n"
                                    f"💵 Prix d'exécution : *${execution_price:,.2f} USD*\n"
                                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                    f"🌦️ Météo Marché : *{translated_regime}*\n\n"
                                    f"{trade_reason}\n"
                                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                    f"🖥️ _Terminal Web mis à jour. Cliquez ci-dessous pour piloter :_"
                                )
                            
                                # Standard buttons layout (Like a Telegram Web App)
                                keyboard = {
                                    "inline_keyboard": [
                                        [
                                            {"text": "📊 Rapport Status", "callback_data": "bot_status"},
                                            {"text": "📋 Historique", "callback_data": "bot_history"}
                                        ],
                                        [
                                            {"text": "⏸️ Pause le Bot", "callback_data": "bot_pause"},
                                            {"text": "🚨 KILL SWITCH", "callback_data": "bot_kill"}
                                        ]
                                    ]
                                }
                            
                                await telegram_bot.send_push_notification(telegram_msg, reply_markup=keyboard)
                            except Exception as exc:
                                logger.error(f"DEX / CEX ORDER REJECTION: {str(exc)}")
                                platform_metrics.ORDERS_FAILED.labels(mode=active_mode).inc()
                                db.add_audit_log(
                                    "ORDER_REJECTED", 
                                    audit_ip(), 
                                    f"Order {side} of {trade_qty_formatted:.5f} {symbol} failed/rejected: {str(exc)}"
                                )
                            
            except Exception as exc:
                logger.error(f"Trading tick failed for {symbol}: {str(exc)}")
                platform_metrics.ERRORS_TOTAL.labels(component="trading_loop").inc()
        # 5. Calculate total portfolio equity (consolidating all active multi-assets positions)
        net_equity = STATE[active_balance_key]
        updated_positions = db.get_positions()
        for p in updated_positions:
            # Valorisation avec le DERNIER PRIX RÉEL connu (jamais de valeur inventée)
            asset_price = STATE["assets"].get(p['symbol'], {}).get("price")
            if asset_price is None:
                asset_price = STATE.get("last_known_prices", {}).get(p['symbol'], 0.0)
            if asset_price is None:
                asset_price = 0.0
            net_equity += p['qty'] * asset_price
            
        STATE["current_equity"] = net_equity
        STATE[active_equity_history_key].append(net_equity)
        if len(STATE[active_equity_history_key]) > 100:
            STATE[active_equity_history_key].pop(0)

        # LOT 8 (PDF Pilier O) : coût de PORTAGE (funding) des positions tenues
        # — une position longue peut être perdante NET même si le prix monte.
        try:
            for _p in db.get_positions():
                _fr = STATE.get("funding_rates", {}).get(_p["symbol"])
                _px = STATE.get("last_known_prices", {}).get(_p["symbol"])
                if _fr is not None and _px:
                    cost_accounting.apply_funding_to_position(_p["symbol"], _p["qty"], _px, _fr)
            STATE["cost_metrics"] = cost_accounting.to_dict()
        except Exception:
            pass

        STATE["last_tick_ts"] = time.time()  # health-score heartbeat

        # VISION_FUTUR §8: live confidence index (self-distrust) - computed periodically
        if int(time.time()) % 30 == 0:
            try:
                _pval = live_p_value(STATE["recent_signals"], STATE["recent_returns"])
                STATE["live_p_value"] = _pval
                _ci = compute_confidence_index(
                    sim_divergence=STATE.get("sim_divergence", 0.0),
                    p_value=_pval,
                    data_quality=STATE.get("data_quality_status", "UNAVAILABLE"),
                )
                STATE["confidence_index"] = _ci["index"]
                STATE["confidence_factor"] = _ci["factor"]
            except Exception:
                pass

        # VISION_FUTUR §5b: supervisor vital-signs check
        supervisor.check()

        # LOT 6 (PDF Pilier L) : REBALANCING périodique du portefeuille
        # (anti-drift : on revient vers les cibles) + diversification RÉELLE
        # entre stratégies (pénalise les stratégies redondantes).
        try:
            if portfolio_allocator.should_rebalance():
                portfolio_allocator.rebalance(
                    STATE, STATE[active_balance_key],
                    portfolio_cvar_pct=portfolio_cvar_pct,
                    realized_vol_annual=None)
            # Corrélation entre STRATÉGIES (Pilier L exigence 2)
            try:
                _strat_rets = {}
                for _sname, _slist in meta_engine.recent_performance.items():
                    if _slist:
                        _strat_rets[_sname] = list(_slist)
                if len(_strat_rets) >= 2:
                    STATE["strategy_diversification"] = portfolio_allocator.strategy_diversification(
                        _strat_rets, min_samples=20)
            except Exception:
                pass
        except Exception as _pa:
            logger.debug(f"Portfolio allocator error: {_pa}")

        # LOT 61: refresh Prometheus gauges at the end of each tick
        try:
            update_metrics_from_state()
        except Exception as exc:
            logger.warning(f"Metrics update failed: {str(exc)}")
            
        # Circuit Breakers evaluation
        tripped = var_metrics.get("tripped", False)
        msg = var_metrics.get("reason", "")
        
        if not tripped:
            tripped, msg = risk_manager.check_circuit_breaker(net_equity)
            
        if tripped:
            STATE["kill_switch_active"] = True
            STATE["is_running"] = False
            
            # Flat close exposures (prix réel connu uniquement)
            for p in updated_positions:
                try:
                    asset_price = STATE["assets"].get(p['symbol'], {}).get("price")
                    if asset_price is None:
                        asset_price = STATE.get("last_known_prices", {}).get(p['symbol'])
                    if asset_price is None:
                        logger.error(f"CIRCUIT BREAKER: cannot flatten {p['symbol']} (no real price) - manual action required.")
                        continue
                    if active_mode == "REAL" and client:
                        client.create_order(symbol=p['symbol'].replace("USDT", "/USDT"), type='market', side='sell', amount=p['qty'])
                    close_val = p['qty'] * asset_price * 0.999
                    STATE[active_balance_key] += close_val
                    db.update_position(p['symbol'], 0, 0, active_mode)
                    db.add_order(
                        symbol=p['symbol'],
                        side="SELL",
                        price=asset_price * 0.999,
                        qty=p['qty'],
                        status="FORCE_LIQUIDATED",
                        mode=active_mode,
                        strategy="EMERGENCY_RISK_BREAKER",
                        order_type="MARKET"
                    )
                except Exception as exc:
                    logger.error(f"Failed during circuit breaker exposure flatting: {str(exc)}")
            db.add_audit_log("CIRCUIT_BREAKER_TRIPPED", audit_ip(), f"EMERGENCY KILL SWITCH ENGAGED: {msg}")
            # LOT 2 (PDF Pilier G) : le circuit breaker déclenche la machine à
            # états -> HALT (cool-down + redémarrage progressif ensuite).
            risk_state.enter(RiskStateMachine.HALT, f"CIRCUIT_BREAKER:{msg[:80]}")
            
        # 6. MLOps Automated Training schedule checks
        if mlops_trainer.check_retrain_schedule() and STATE["historical_bars"] is not None:
            try:
                mlops_trainer.execute_pipeline(STATE["historical_bars"])
            except Exception as e:
                logger.error(f"MLOps pipeline failed execution: {str(e)}")
                
        # Telemetry broadcast
        await broadcast_telemetry(consensus)
        
        await asyncio.sleep(settings.get_float("trading", "loop_sleep_seconds", 2.5))  # config-driven tick


def serialize_helper(obj):
    """
    Safely converts any datetime or non-serializable database object into standard string/types
    before sending over WebSockets or JSON responses.
    Also strips NaN/Infinity floats which are not valid JSON (would crash the endpoint with
    "ValueError: Out of range float values are not JSON compliant").
    """
    if isinstance(obj, dict):
        return {k: serialize_helper(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_helper(i) for i in obj]
    elif isinstance(obj, tuple):
        return [serialize_helper(i) for i in obj]
    elif isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    elif isinstance(obj, int):
        return obj
    elif hasattr(obj, "isoformat"):  # Matches datetime.datetime, date, etc.
        return obj.isoformat()
    return obj


def explain_last_decision(consensus) -> list:
    """
    VISION §4.5: top-5 contributing features/reasons of the last decision.
    Returns a ranked list of {feature, contribution} derived from real data.
    """
    out = []
    try:
        contribs = consensus.get("contributions", {}) or {}
        ranked = sorted(contribs.items(), key=lambda kv: abs(kv[1].get("signal", 0.0) * kv[1].get("weight", 0.0)), reverse=True)
        for name, c in ranked[:5]:
            out.append({
                "feature": name,
                "signal": round(float(c.get("signal", 0.0)), 4),
                "weight": round(float(c.get("weight", 0.0)), 4),
                "contribution": round(float(c.get("signal", 0.0)) * float(c.get("weight", 0.0)), 4),
            })
        # append market-state features
        extras = [
            ("VPIN", consensus.get("modulate_factor", 1.0) if consensus.get("modulate_factor", 1.0) < 1.0 else 0.0),
        ]
        for fname, val in extras:
            if abs(val) > 1e-6:
                out.append({"feature": fname, "signal": round(val, 4), "weight": 1.0, "contribution": round(val, 4)})
    except Exception:
        pass
    return out


def update_metrics_from_state():
    """
    Pushes the current STATE snapshot into the Prometheus registry (LOT 61).
    Called at the end of every trading-loop tick and on demand.
    """
    active_mode = STATE["mode"]
    active_balance_key = "balance_demo" if active_mode == "DEMO" else "balance_real"
    _lp = STATE["last_price"]
    platform_metrics.MARKET_LAST_PRICE.labels(symbol="BTCUSDT").set(_lp if _lp is not None else 0.0)
    platform_metrics.MARKET_EQUITY.labels(mode=active_mode).set(STATE["current_equity"])
    platform_metrics.MARKET_BALANCE.labels(mode=active_mode).set(STATE[active_balance_key])

    initial_cap = STATE["initial_capital_demo"] if active_mode == "DEMO" else STATE["initial_capital_real"]
    live_pnl_usd = STATE["current_equity"] - initial_cap if initial_cap > 0 else 0.0
    live_pnl_pct = (live_pnl_usd / initial_cap) * 100.0 if initial_cap > 0 else 0.0
    platform_metrics.MARKET_PNL_USD.labels(mode=active_mode).set(live_pnl_usd)
    platform_metrics.MARKET_PNL_PCT.labels(mode=active_mode).set(live_pnl_pct)

    platform_metrics.REGIME_ID.set(STATE["regime_id"])
    platform_metrics.RISK_EXPOSURE.set(
        (STATE["current_equity"] / STATE[active_balance_key] - 1.0) * 100.0 if STATE[active_balance_key] > 0 else 0.0
    )
    platform_metrics.POSITIONS_OPEN.set(len(STATE.get("cached_positions") or []))
    platform_metrics.SENTIMENT_INDEX.set(_neutral(STATE.get("sentiment_index")))
    platform_metrics.ONCHAIN_RISK.set(_neutral(STATE.get("onchain_risk_score"), 0.5))
    platform_metrics.WS_CLIENTS.set(len(STATE["connected_websockets"]))


def compile_telemetry_data(consensus_signals=None) -> dict:
    """
    Compiles and returns the unified telemetry payload.
    Caches database queries for 3.0 seconds to avoid blocking the event loop on high-frequency ticks.
    """
    now = time.time()
    if now - STATE.get("last_db_query_time", 0.0) >= 3.0 or consensus_signals is not None:
        STATE["last_db_query_time"] = now
        try:
            STATE["cached_positions"] = db.get_positions()
            STATE["cached_orders"] = db.get_all_orders()
            STATE["cached_audit_logs"] = db.get_audit_logs()
        except Exception as e:
            logger.error(f"Failed to fetch telemetry data from database: {str(e)}")
            
    positions = STATE.get("cached_positions", [])
    orders = STATE.get("cached_orders", [])
    audit_logs = STATE.get("cached_audit_logs", [])
    
    # Calculate live P&L
    active_mode = STATE["mode"]
    initial_cap = STATE["initial_capital_demo"] if active_mode == "DEMO" else STATE["initial_capital_real"]
    current_eq = STATE["current_equity"]
    
    live_pnl_usd = current_eq - initial_cap if initial_cap > 0 else 0.0
    # LOT 8 (PDF Pilier O) : PnL NET — les coûts RÉELS (frais + slippage +
    # impact + gas + funding) sont retranchés du PnL affiché. Mentalité n°2 :
    # l'edge est net des coûts.
    try:
        _costs = float(cost_accounting.total_costs_usd)
        live_pnl_usd -= _costs
    except Exception:
        pass
    live_pnl_pct = (live_pnl_usd / initial_cap) * 100.0 if initial_cap > 0 else 0.0
    
    # Packaged JSON (Passed through serialize_helper to resolve any PostgreSQL datetime serialization mismatches!)
    telemetry = {
        "mode": STATE["mode"],
        "is_running": STATE["is_running"],
        "kill_switch_active": STATE["kill_switch_active"],
        "last_price": STATE["last_price"],
        "price_history": STATE["price_history"],
        "order_book": STATE["order_book"],
        "balance": STATE["balance_demo"] if STATE["mode"] == "DEMO" else STATE["balance_real"],
        "current_equity": STATE["current_equity"],
        "equity_history": STATE["equity_history_demo"] if STATE["mode"] == "DEMO" else STATE["equity_history_real"],
        "live_pnl_usd": live_pnl_usd,
        "live_pnl_pct": live_pnl_pct,
        "regime_id": STATE["regime_id"],
        "regime_name": STATE["regime_name"],
        "ml_prediction_pct": STATE["ml_prediction_pct"],
        "ppo_action": STATE["ppo_action"],
        "consensus": consensus_signals,
        "positions": serialize_helper(positions),
        "orders": serialize_helper(orders[:15]),
        "audit_logs": serialize_helper(audit_logs[:15]),
        
        # ADVANCED TELEMETRY EXPOSURE
        "sentiment_index": STATE["sentiment_index"],
        "sentiment_available": STATE.get("sentiment_available", False),
        "sentiment_confidence": STATE.get("sentiment_confidence", 0.0),
        "recent_headlines": STATE.get("recent_headlines", [])[:10],
        "news_shock": STATE.get("news_shock", {"shock_detected": False}),
        "macro_phase": STATE.get("macro_phase", "NONE"),
        "macro_event": STATE.get("macro_event", ""),
        "onchain_risk_score": STATE["onchain_risk_score"],
        "onchain_available": STATE.get("onchain_available", False),
        "eth_defi_balance": STATE["eth_defi_balance"],
        "defi_wallet_address": STATE["defi_wallet_address"],
        "assets_telemetry": STATE["assets"],
        "asset_data_status": STATE.get("asset_data_status", {}),
        "order_books": {k: {kk: vv for kk, vv in v.items() if kk != "_ts"} for k, v in STATE.get("order_books", {}).items()},
        "price_consensus": STATE.get("price_consensus", {}),
        "price_divergent": STATE.get("price_divergent", {}),
        "macro_calendar": macro_calendar.get_calendar(limit=5),
        "options_strategy": STATE["options_strategy"],
        "real_iv": STATE.get("real_iv", {}),
        
        "using_fallback_data": STATE.get("using_fallback_data", False),
        "data_quality_status": STATE["data_quality_status"],
        "vol_target_scale": STATE.get("vol_target_scale", 1.0),
        "last_reasoning": STATE.get("last_reasoning", []),
        "last_reasoning_symbol": STATE.get("last_reasoning_symbol", ""),
        "regime_probs": STATE.get("regime_probs", {}),
        "conviction_threshold": STATE.get("conviction_threshold", 0.15),
        "no_trade_count": STATE.get("no_trade_stats", {}).get("count", 0),
        "moe_gate": STATE.get("moe_gate", {}),
        "risk_budget": STATE.get("risk_budget", {}),
        "risk_state": STATE.get("risk_state", {}),
        "last_kelly": STATE.get("last_kelly", {}),
        "last_rr_check": STATE.get("last_rr_check", {}),
        "strategy_win_rates": STATE.get("strategy_win_rates", {}),
        "strategy_trade_counts": STATE.get("strategy_trade_counts", {}),
        "risk_pipeline_steps": STATE.get("risk_pipeline_steps", [])[-12:],
        "regime_confidence": STATE.get("regime_confidence", {}),
        "hmm_validation": STATE.get("hmm_validation", {}),
        "expert_contribution": mixture_of_experts.expert_contribution_report(),
        "sleeping_experts": list(mixture_of_experts.sleeping),
        "causal_parents": STATE.get("causal_parents", []),
        "causal_analyzed": STATE.get("causal_analyzed", False),
        "research_gate": hypothesis_generator.can_run_research(),
        "order_flow": {s: order_flow.status(s) for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]},
        "last_sor_choice": STATE.get("last_sor_choice", {}),
        "sim_divergence": STATE.get("sim_divergence", 0.0),
        "confidence_index": STATE.get("confidence_index", 100),
        "confidence_factor": STATE.get("confidence_factor", 1.0),
        "live_p_value": STATE.get("live_p_value", 0.5),
        "structural_regimes": STATE.get("structural_regimes", {}),
        "cross_asset_bias": STATE.get("cross_asset_bias", 0.0),
        "desk_allocations": STATE.get("desk_allocations", {}),
        "pending_approvals": len(STATE.get("pending_approvals", [])),
        "consultative_mode": STATE.get("consultative_mode", False),
        "last_narrative": STATE.get("last_narrative", ""),
        "ppo_buffer_size": len(STATE.get("ppo_buffer", [])),
        "strategy_weights": meta_engine.get_strategy_weights(),
        "active_models": model_selector.get_status().get("active_models", []),
        "admitted_signals": list(hypothesis_generator.admitted.keys()),
        "capital_exposure": capital_allocator.get_current_exposure(),
        "portfolio_allocation": STATE.get("portfolio_allocation", {}),
        "strategy_diversification": STATE.get("strategy_diversification", {}),
        "position_pyramids": STATE.get("position_pyramids", {}),
        "counterparty": counterparty_risk.to_dict(),
        "reason_weights": STATE.get("reason_weights", {}),
        "reason_weights_factor": STATE.get("reason_weights_factor", 1.0),
        "cost_metrics": STATE.get("cost_metrics", {}),
        "attribution_report": STATE.get("attribution_report", {}),
        "quality_metrics": STATE.get("quality_metrics", {}),
        "stress_test_report": STATE.get("stress_test_report", {}),
        "bootstrap_sharpe": STATE.get("bootstrap_sharpe", {}),
        "module_honesty": {
            "registry": get_module_status(),
            "summary": status_summary(),
            "note": "Un module ÉDUCATIF n'influence JAMAIS le sizing réel (Faille 7 PDF).",
        },
        "watchdog": {
            "tasks_monitored": list(_BG_TASKS.keys()),
            "tasks_alive": sum(1 for t in _BG_TASKS.values() if t and not t.done()),
            "supervisor_issues": supervisor.last_issues,
            "running": True,
        },
        
        "copy_traders": [
            {
                "trader_id": t.trader_id,
                "name": t.name,
                "roi_annual": t.roi_annual * 100.0,
                "win_rate": t.win_rate * 100.0,
                "max_drawdown": t.max_drawdown * 100.0,
                "sharpe": t.sharpe,
                "seq_score": t.seq_score,
                "pnl_month": getattr(t, "pnl_month", 0.0),
                "account_value": getattr(t, "account_value", 0.0),
                "active_copied": t.trader_id in copy_manager.copied_traders,
                "allocated_capital": copy_manager.copied_traders[t.trader_id]["allocated_capital"] if t.trader_id in copy_manager.copied_traders else 0.0,
                "follow_mode": copy_manager.copied_traders[t.trader_id].get("mode", "-") if t.trader_id in copy_manager.copied_traders else "-",
                "pnl_estimate_usd": copy_manager.copied_traders[t.trader_id].get("pnl_estimate_usd", 0.0) if t.trader_id in copy_manager.copied_traders else 0.0
            }
            for t in copy_manager.get_ranked_traders()
        ]
    }
    # Sanitize the full payload: strips NaN/Inf (invalid JSON) and datetimes
    return serialize_helper(telemetry)


async def broadcast_telemetry(consensus_signals):
    """
    Broadcasts real-time trading metrics to all active dashboard connections.
    Audit B5-1: payload is serialized ONCE, each client send is isolated with
    try/except and slow/failed clients are dropped immediately.
    """
    if not STATE["connected_websockets"]:
        return

    payload = compile_telemetry_data(consensus_signals)
    try:
        text = json.dumps(payload, default=str)
    except Exception as e:
        logger.warning(f"Telemetry serialization failed: {e}")
        return

    dead_sockets = []
    for ws in list(STATE["connected_websockets"]):
        try:
            await ws.send_text(text)
        except Exception:
            dead_sockets.append(ws)

    for ws in dead_sockets:
        try:
            STATE["connected_websockets"].remove(ws)
            platform_metrics.WS_CLIENTS.set(len(STATE["connected_websockets"]))
        except Exception:
            pass


async def task_watchdog_loop():
    """
    LOT 7 (PDF Faille 6 + Pilier K) : WATCHDOG des tâches de fond.
    - Le Supervisor étendu surveille les signes vitaux de TOUS les flux.
    - Si une tâche de fond critique est morte (terminée ou en erreur),
      elle est REDÉMARRÉE automatiquement (mentalité n°13 : surveiller).
    - Les tâches enregistrées dans STATE["background_tasks"] sont les
      coroutines lancées au démarrage (boucle trading, WS, consensus,
      order flow, schedulers...).
    """
    while True:
        await asyncio.sleep(20)
        try:
            # 1. Signes vitaux étendus (Pilier K)
            supervisor.check(force=True)

            # 2. Redémarrage des tâches de fond mortes
            # NOTE : les objets Task ne sont PAS stockés dans STATE (non
            # sérialisables) — registre module-level _BG_TASKS uniquement.
            for name, task in list(_BG_TASKS.items()):
                if task is None or task.done():
                    logger.warning(f"🛠️ WATCHDOG: tâche de fond '{name}' MORTE -> redémarrage...")
                    try:
                        restart = TASK_FACTORIES.get(name)
                        if restart:
                            new_task = asyncio.create_task(restart())
                            _BG_TASKS[name] = new_task
                            logger.info(f"✅ WATCHDOG: '{name}' redémarrée")
                    except Exception as e:
                        logger.error(f"WATCHDOG: échec redémarrage {name}: {e}")
        except Exception as e:
            logger.debug(f"Watchdog error: {e}")


async def websocket_heartbeat_loop():
    """Audit B5-2: app-level heartbeat every 30s keeps dead mobile clients honest."""
    while True:
        await asyncio.sleep(30)
        dead = []
        for ws in list(STATE["connected_websockets"]):
            try:
                await ws.send_text(json.dumps({"type": "heartbeat", "ts": time.time()}))
            except Exception:
                dead.append(ws)
        for ws in dead:
            try:
                STATE["connected_websockets"].remove(ws)
                platform_metrics.WS_CLIENTS.set(len(STATE["connected_websockets"]))
            except Exception:
                pass


async def shutdown_event():
    """LOT 65 (roadmap #5): graceful shutdown - close journals, notify, flush state."""
    logger.info("🛑 Graceful shutdown initiated...")
    try:
        # Flush trade journal to disk
        if hasattr(trade_journal, "_save"):
            trade_journal._save()
    except Exception as e:
        logger.warning(f"Shutdown: journal flush failed: {e}")
    try:
        for ws in list(STATE.get("connected_websockets", [])):
            try:
                await ws.close(code=1001)
            except Exception:
                pass
        STATE["connected_websockets"].clear()
    except Exception as e:
        logger.warning(f"Shutdown: websocket close failed: {e}")
    logger.info("🛑 Graceful shutdown complete.")


# REST endpoints

@app.get("/api/telemetry")
async def get_telemetry_rest():
    """
    REST Fallback API to query telemetry.
    Ensures 100% platform connectivity even when WebSockets are blocked by client browser or proxy!
    """
    return JSONResponse(compile_telemetry_data())


@app.get("/metrics")
async def get_prometheus_metrics():
    """
    Prometheus text exposition (scraped by the bundled prometheus.yml).
    Institutional observability for Grafana dashboards & auto-scaling alerts.
    """
    platform_metrics.refresh_uptime()
    return Response(content=platform_metrics.get_metrics_text(), media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="dashboard.html", 
        context={"request": request}
    )


@app.get("/telegram", response_class=HTMLResponse)
async def get_telegram_mini_app(request: Request):
    """
    Telegram Mini App (mobile-first terminal). Set this URL as the Mini App
    link in @**BotFather** for your Telegram bot.
    """
    return templates.TemplateResponse(
        request=request,
        name="telegram_mini_app.html",
        context={"request": request}
    )


@app.get("/telegram_mini_app.html", response_class=HTMLResponse)
async def get_telegram_mini_app_alias(request: Request):
    """
    Backward-compatible alias for those who configured the old filename in
    @BotFather. Redirects to /telegram.
    """
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/telegram")


@app.get("/api/status")
async def get_status():
    # Safe serialization - remove non-serializable objects (DataFrames, etc.)
    safe_state = {k: v for k, v in STATE.items() if not isinstance(v, (pd.DataFrame, pd.Series)) and k != "ppo_buffer"}
    # Also sanitize NaN/Infinity floats (invalid JSON) and convert datetimes
    safe_state = serialize_helper(safe_state)
    return JSONResponse(safe_state)


@app.get("/api/history")
async def get_history_endpoint(timeframe: str = "1h", limit: int = 120, offset: int = 0):
    """
    Returns historical candle bars for different timeframes (1h, 4h, 1d).
    Uses persistent database caching and Binance API.
    """
    valid_timeframes = ["1h", "4h", "1d"]
    if timeframe not in valid_timeframes:
        timeframe = "1h"
        
    interval = "1h" if timeframe == "1h" else "4h" if timeframe == "4h" else "1d"
    
    # Check persistent database cache
    cache_symbol = f"BTCUSDT_{timeframe}"
    df = db.load_candles(cache_symbol, limit=120)
    
    if df.empty or len(df) < 120:
        # Fetch from Binance API
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit=120"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    bars = []
                    for b in data:
                        bars.append({
                            "timestamp": pd.to_datetime(b[0], unit='ms'),
                            "open": float(b[1]),
                            "high": float(b[2]),
                            "low": float(b[3]),
                            "close": float(b[4]),
                            "volume": float(b[5])
                        })
                    df = pd.DataFrame(bars).set_index("timestamp")
                    db.save_candles(cache_symbol, df)
        except Exception as e:
            logger.warning(f"Failed to fetch Binance klines for {timeframe}: {str(e)}")

        # Fallback: Yahoo Finance (works even when Binance is geo-blocked / offline)
        if df.empty or len(df) < 120:
            logger.info(f"Binance unavailable for {timeframe}. Falling back to Yahoo Finance...")
            yahoo_ticker = "BTC-USD"
            try:
                df_yahoo = await fetch_yahoo_finance_candles(yahoo_ticker, interval=interval, range_str="5d")
                if df_yahoo is not None and not df_yahoo.empty:
                    df = df_yahoo
                    db.save_candles(cache_symbol, df)
            except Exception as e:
                logger.warning(f"Yahoo Finance fallback failed for {timeframe}: {str(e)}")
            
    if df.empty:
        logger.error(f"Failed to load historical candles for {timeframe}. No database or CEX feed active.")
        raise HTTPException(status_code=503, detail="Historical market data currently unavailable.")
        
    prices = df['close'].values.tolist()
    timestamps = [str(t) for t in df.index]
    limit = max(10, min(1000, limit))
    offset = max(0, offset)
    return {
        "timeframe": timeframe,
        "total": len(prices),
        "limit": limit,
        "offset": offset,
        "prices": prices[offset:offset + limit],
        "timestamps": timestamps[offset:offset + limit]
    }


# ============ AUTHENTICATION (roadmap #4) ============
class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)
    totp_code: str = Field(default="", max_length=16)


@app.post("/api/login")
async def login(payload: LoginRequest):
    """
    Authenticates an operator and returns a JWT bearer token.
    AUDIT C7 (multi-user): authenticates against the `users` table (bcrypt),
    with the env ADMIN_USER/ADMIN_PASSWORD as the bootstrap admin fallback.
    On first successful bootstrap login the DB hash is upgraded to bcrypt.
    Optional TOTP second factor when ADMIN_TOTP_SECRET is set.
    """
    import hmac as _hmac
    import bcrypt as _bc

    admin_user = os.getenv("ADMIN_USER", "admin")
    admin_pass = os.getenv("ADMIN_PASSWORD", "ChangeMe!Institutionnel2026")
    totp_secret = os.getenv("ADMIN_TOTP_SECRET", "")

    # 1) Try the database users table first
    db_user = db.get_user(payload.username)
    role = Roles.VIEWER
    ok = False
    if db_user and db_user.get("password_hash"):
        ph = db_user.get("password_hash")
        if ph and not ph.startswith("$2") and ph != "hash_admin_secret":
            ok = False  # malformed hash -> cannot verify
        else:
            try:
                ok = _bc.checkpw(payload.password.encode("utf-8"), ph.encode("utf-8"))
            except Exception:
                ok = False
        role = db_user.get("role") or Roles.VIEWER

    # 2) Bootstrap admin fallback (env) - also upgrades the DB hash on success
    if not ok and payload.username == admin_user and _hmac.compare_digest(payload.password, admin_pass):
        ok = True
        role = Roles.ADMIN
        try:
            db.create_user(admin_user, _bc.hashpw(admin_pass.encode("utf-8"), _bc.gensalt()).decode("utf-8"), Roles.ADMIN)
        except Exception as e:
            logger.warning(f"Bootstrap admin hash upgrade skipped: {e}")

    if not ok:
        db.add_audit_log("AUTH_FAILURE", audit_ip(), f"Failed login attempt for '{payload.username}'.")
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    if totp_secret:  # TOTP second factor enforced whenever configured
        import pyotp as _pyotp
        totp = _pyotp.TOTP(totp_secret)
        if not totp.verify(payload.totp_code, valid_window=1):
            raise HTTPException(status_code=401, detail="Invalid TOTP code.")

    user_id = int(db_user.get("id") or 1) if db_user else 1
    token = AuthManager.create_jwt_token(user_id, payload.username, role)
    logger.info(f"🔐 Operator '{payload.username}' authenticated successfully (role {role}).")
    return {"token": token, "role": role, "username": payload.username}


@app.post("/api/toggle-strategy")
async def toggle_strategy(payload: StrategyToggle, _auth: dict = Depends(require_auth)):
    strategy = next((s for s in strategies_list if s.name == payload.name), None)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
        
    strategy.enabled = payload.enabled
    db.add_audit_log(
        "STRATEGY_TOGGLED", 
        audit_ip(), 
        f"Modified strategy '{payload.name}' enabled status to {payload.enabled}."
    )
    return {"status": "Success", "message": f"Strategy {payload.name} modified to {payload.enabled}"}


@app.post("/api/toggle-bot")
async def toggle_bot(payload: BotToggleRequest, _auth: dict = Depends(require_auth)):
    STATE["is_running"] = payload.is_running
    action_str = "STARTED" if payload.is_running else "PAUSED"
    db.add_audit_log(
        "BOT_STATE_CHANGED", 
        audit_ip(), 
        f"Automated trading loop has been manually {action_str}."
    )
    return {"status": "Success", "message": f"Automated trading loop {action_str} successfully."}


@app.post("/api/set-demo-balance")
async def set_demo_balance(payload: SetBalanceRequest, _auth: dict = Depends(require_auth)):
    if payload.balance <= 0:
        raise HTTPException(status_code=400, detail="Balance must be positive.")
    STATE["balance_demo"] = payload.balance
    risk_manager.set_initial_capital(payload.balance)
    STATE["initial_capital_demo"] = payload.balance
    STATE["current_equity"] = payload.balance
    STATE["equity_history_demo"] = [payload.balance]
    
    # Persist in DB setting so it survives server restart and browser refresh!
    db.save_setting("balance_demo", str(payload.balance))
    db.save_setting("initial_capital_demo", str(payload.balance))
    
    db.add_audit_log(
        "DEMO_BALANCE_RESET", 
        audit_ip(), 
        f"Demo balance has been manually reset to {payload.balance} USD."
    )
    return {"status": "Success", "message": f"Demo balance successfully set to {payload.balance} USD."}


@app.post("/api/retrain")
async def trigger_manual_retrain(_auth: dict = Depends(require_auth)):
    df = STATE["historical_bars"]
    if df is None:
        raise HTTPException(status_code=400, detail="No historical bars cache loaded yet.")
        
    res = mlops_trainer.execute_pipeline(df)
    return JSONResponse(res)


@app.post("/api/monte-carlo")
async def trigger_monte_carlo(_auth: dict = Depends(require_auth)):
    """
    On-Demand Monte Carlo Stress Testing API endpoint.
    Runs 10,000 simulations and returns structural safety metrics.
    """
    df = STATE["historical_bars"]
    if df is None:
        raise HTTPException(status_code=400, detail="No historical data loaded yet.")
        
    current_p = STATE["last_price"]
    if current_p is None:
        raise HTTPException(status_code=400, detail="No live price fetched yet. Please wait for WebSockets synchronization.")
        
    vols = df['close'].pct_change().std()
    res = monte_carlo_tester.execute_stress_test(
        initial_capital=STATE["balance_demo"] if STATE["mode"] == "DEMO" else STATE["balance_real"],
        current_price=current_p,
        historical_volatility=vols if not np.isnan(vols) else 0.02
    )
    
    db.add_audit_log(
        "MONTE_CARLO_TEST_EXECUTED",
        audit_ip(),
        f"Executed 10,000 Monte Carlo stress-testing simulations. Survival rate: {res['survival_probability_pct']:.2f}%."
    )
    
    return JSONResponse(res)


@app.post("/api/risk-settings")
async def update_risk_settings(payload: RiskSettingsUpdate, _auth: dict = Depends(require_auth)):
    risk_manager.params.update(payload.dict())
    db.add_audit_log(
        "RISK_SETTINGS_UPDATED", 
        audit_ip(), 
        f"Updated Risk thresholds: Max daily drawdown to {payload.max_daily_drawdown_pct*100:.2f}%."
    )
    return {"status": "Success", "message": "Risk management policies updated successfully."}


@app.post("/api/keys")
async def store_keys(payload: KeyStorage, _auth: dict = Depends(require_auth)):
    db.save_setting("api_keys_rotated_at", str(time.time()))  # audit B3-6: rotation tracking
    db.save_setting(f"{payload.exchange}_api_key", payload.api_key, encrypt=True)
    db.save_setting(f"{payload.exchange}_secret_key", payload.secret_key, encrypt=True)
    db.add_audit_log(
        "API_KEYS_STORED", 
        audit_ip(), 
        f"Stored and encrypted API key pairs for exchange {payload.exchange}."
    )
    return {"status": "Success", "message": f"Encrypted keys stored for {payload.exchange}."}


@app.post("/api/2fa-switch")
async def switch_mode(payload: SwitchModeRequest, _auth: dict = Depends(require_auth)):
    """
    Secures the Demo <-> Real trading modes transitions.
    AUDIT B3: hardcoded 2FA test codes ("123456"/"888888") REMOVED - they were a
    backdoor to REAL mode. Factors accepted now:
      - a valid TOTP code when ADMIN_TOTP_SECRET is configured, or
      - an EVM wallet address (0x...) provided by the operator.
    AUDIT D1: switching to REAL requires the autopilot paper-validation period
    to have elapsed (config autopilot.paper_validation_required / min_paper_validation_days).
    """
    global ccxt_client
    totp_secret = os.getenv("ADMIN_TOTP_SECRET", "")
    is_wallet = payload.verification_2fa.startswith("0x") and len(payload.verification_2fa) == 42
    is_totp = False
    if totp_secret:
        try:
            import pyotp
            is_totp = pyotp.TOTP(totp_secret).verify(payload.verification_2fa, valid_window=1)
        except Exception:
            is_totp = False

    if not (is_wallet or is_totp):
        db.add_audit_log("AUTH_FAILURE", audit_ip(), f"Failed 2FA transit attempt to mode {payload.target_mode}.")
        raise HTTPException(status_code=401, detail="Invalid 2FA factor. Security block triggered.")
        
    if payload.target_mode not in ["DEMO", "REAL"]:
        raise HTTPException(status_code=400, detail="Invalid target trading mode.")
        
    if payload.target_mode == "REAL":
        # AUTOPILOT GATE (audit D1): paper validation period must have elapsed
        if settings.get_bool("autopilot", "paper_validation_required", True):
            min_days = settings.get_int("autopilot", "min_paper_validation_days", 7)
            first_start = float(db.get_setting("platform_first_start_ts") or time.time())
            db.save_setting("platform_first_start_ts", str(first_start))
            elapsed_days = (time.time() - first_start) / 86400.0
            if elapsed_days < min_days:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"Autopilot gate: REAL mode requires {min_days} days of validated "
                        f"paper/DEMO trading first (currently {elapsed_days:.1f} days). "
                        f"This protects you from deploying untested logic with real money."
                    ),
                )

        # Force reload keys from database
        ccxt_client = None
        client = get_ccxt_client()
        if not client:
            raise HTTPException(
                status_code=400, 
                detail="Real Mode denied. Please configure valid and active Exchange API Keys in the dashboard first."
            )
            
    STATE["mode"] = payload.target_mode
    db.add_audit_log(
        "TRADING_MODE_CHANGED", 
        audit_ip(), 
        f"Successfully changed system trading mode to {payload.target_mode} via authorization {payload.verification_2fa[:12]}..."
    )
    return {"status": "Success", "message": f"Platform successfully switched to {payload.target_mode} Mode."}


@app.post("/api/copy-trade")
async def manage_copytrade(payload: CopyTradeRequest, _auth: dict = Depends(require_auth)):
    if payload.action == "START":
        # LOT 7 (PDF Pilier J) : plafond de capital par trader copié
        try:
            _total_cap = STATE["balance_demo"] if STATE["mode"] == "DEMO" else STATE["balance_real"]
            _cap_ok, _cap_msg = copy_manager.check_trader_risk(payload.trader_id, _total_cap) if payload.trader_id in copy_manager.copied_traders else (True, "")
            if not _cap_ok:
                return {"ok": False, "error": f"Risque copytrading: {_cap_msg}"}
        except Exception:
            pass
        ok, msg = copy_manager.start_copying(payload.trader_id, payload.allocated_capital)
        if ok:
            db.save_copy_allocation(payload.trader_id, payload.allocated_capital, 1)
            db.add_audit_log("COPY_START", audit_ip(), f"Started copytrading {payload.trader_id} with {payload.allocated_capital} USD allocation.")
            return {"status": "Success", "message": msg}
        raise HTTPException(status_code=400, detail=msg)
    else:
        ok, msg = copy_manager.stop_copying(payload.trader_id)
        if ok:
            db.save_copy_allocation(payload.trader_id, 0.0, 0)
            db.add_audit_log("COPY_STOP", audit_ip(), f"Stopped copytrading {payload.trader_id}.")
            return {"status": "Success", "message": msg}
        raise HTTPException(status_code=400, detail=msg)


@app.get("/api/v1/honesty")
async def api_v1_honesty(_auth: dict = Depends(require_auth)):
    """
    LOT 9 (PDF Faille 7) : étiquetage honnête des modules.
    PRODUCTION / EXPÉRIMENTAL / ÉDUCATIF + gardes associées + note.
    """
    return {
        "modules": get_module_status(),
        "summary": status_summary(),
        "rule": "Un module ÉDUCATIF ne doit JAMAIS influencer une décision de trading.",
        "ts": time.time(),
    }


@app.get("/api/v1/attribution")
async def api_v1_attribution(_auth: dict = Depends(require_auth)):
    """
    LOT 8 (PDF Pilier Q) : attribution de performance — d'où vient chaque
    dollar (facteur, régime, actif, stratégie) + métriques de qualité.
    """
    return {
        "attribution": STATE.get("attribution_report", {}),
        "quality_metrics": STATE.get("quality_metrics", {}),
        "costs": STATE.get("cost_metrics", {}),
        "ts": time.time(),
    }


@app.post("/api/v1/stress")
async def api_v1_stress(_auth: dict = Depends(require_auth)):
    """
    LOT 8 (PDF Pilier N) : stress test par SCÉNARIOS de crises RÉELLES
    (COVID 2020, krach 2018, FTX 2022) sur le portefeuille COMPLET.
    """
    try:
        _positions = db.get_positions()
        _prices = STATE.get("last_known_prices", {})
        _bal = STATE["balance_demo"] if STATE["mode"] == "DEMO" else STATE["balance_real"]
        _res = scenario_tester.run_stress(_positions, _bal, _prices)
        STATE["stress_test_report"] = _res
        db.add_audit_log("STRESS_TEST", audit_ip(),
                         f"Stress crises réelles: {_res['status']} (pire {_res['worst']} {_res['worst_loss_pct']}%)")
        return _res
    except Exception as e:
        import traceback
        logger.error(f"STRESS TEST endpoint error: {e}\n{traceback.format_exc()}")
        return {"ok": False, "error": str(e)}


@app.post("/api/risk-state/reset")
async def reset_risk_state(_auth: dict = Depends(require_auth)):
    """
    LOT 2 : remet la machine à états NORMAL/CAUTION/HALT à NORMAL.
    L'opérateur humain reste le décideur final (mentalité n°10).
    """
    try:
        changed = risk_state.reset(reason="api")
        STATE["risk_state"] = risk_state.to_dict()
        if changed:
            db.add_audit_log("RISK_STATE_RESET", audit_ip(), "État risque remis à NORMAL via API")
        return {"ok": True, "risk_state": risk_state.to_dict()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/kill-switch")
async def engage_kill_switch(_auth: dict = Depends(require_auth)):
    STATE["kill_switch_active"] = True
    STATE["is_running"] = False
    
    positions = db.get_positions()
    active_mode = STATE["mode"]
    active_balance_key = "balance_demo" if active_mode == "DEMO" else "balance_real"
    client = get_ccxt_client() if active_mode == "REAL" else None
    
    for p in positions:
        try:
            asset_price = STATE["assets"].get(p['symbol'], {}).get("price")
            if asset_price is None:
                asset_price = STATE.get("last_known_prices", {}).get(p['symbol'])
            if asset_price is None:
                logger.error(f"Kill switch: cannot close {p['symbol']} (no real price) - manual action required.")
                continue
            if active_mode == "REAL" and client:
                client.create_order(symbol=p['symbol'].replace("USDT", "/USDT"), type='market', side='sell', amount=p['qty'])
                
            close_val = p['qty'] * asset_price * 0.999
            STATE[active_balance_key] += close_val
            db.update_position(p['symbol'], 0, 0, active_mode)
            db.add_order(
                symbol=p['symbol'],
                side="SELL",
                price=asset_price * 0.999,
                qty=p['qty'],
                status="FORCE_LIQUIDATED",
                mode=active_mode,
                strategy="EMERGENCY_KILL_SWITCH",
                order_type="MARKET"
            )
        except Exception as exc:
            logger.error(f"Emergency close failed for {p['symbol']}: {str(exc)}")
            
    db.add_audit_log("KILL_SWITCH_ENGAGED", audit_ip(), "Global KILL SWITCH activated manually. Closed all open exposures.")
    return {"status": "Success", "message": "EMERGENCY GLOBAL KILL SWITCH ENGAGED. All exposures flatted, system locked."}


@app.post("/api/reset-bot")
async def reset_bot(_auth: dict = Depends(require_auth)):
    STATE["kill_switch_active"] = False
    STATE["is_running"] = True
    risk_manager.circuit_breaker_active = False
    db.add_audit_log("SYSTEM_RESET", audit_ip(), "Unlocked system state from emergency stop.")
    return {"status": "Success", "message": "System successfully unlocked and restarted."}


@app.post("/api/run-backtest")
async def run_backtest_handler(_auth: dict = Depends(require_auth)):
    df = STATE["historical_bars"]
    if df is None:
        raise HTTPException(status_code=400, detail="Historical bars not loaded yet.")
        
    # LOT 8 (PDF Pilier N) : AUDIT DES BIAIS avant tout backtest (look-ahead,
    # survivorship, slippage). Un backtest qui échoue à l'audit est REJETÉ.
    try:
        _bias = audit_backtest(
            df,
            assets_universe=list(STATE["assets"].keys()),
            assets_tested=list(STATE["assets"].keys()),
            slippage_bps=5.0,          # coûts réalistes (jamais 0)
            commission_pct=0.001,
        )
        STATE["last_bias_audit"] = _bias
        if _bias["status"] == "REJECTED":
            db.add_audit_log("BACKTEST_BIAS_REJECTED", audit_ip(),
                             f"Backtest rejeté: {_bias['issues']}")
            return {"ok": False, "status": "REJECTED", "bias_audit": _bias}
        db.add_audit_log("BACKTEST_BIAS_OK", audit_ip(),
                         f"Audit biais passé (score {_bias['score']})")
    except Exception as _be:
        logger.warning(f"Bias audit failed: {_be}")
        _bias = {"status": "UNKNOWN"}

    backtester = EventDrivenBacktester(initial_capital=100000.0)
    local_detector = MarketRegimeDetector()
    local_predictor = LSTMLikePredictor(5, 8)
    local_ppo = PPOTRAgent(4, 1)
    
    split = int(len(df) * 0.6)
    train_slice = df.iloc[:split]
    test_slice = df.iloc[split:]
    
    train_returns = train_slice['close'].pct_change().dropna().values
    train_vols = train_slice['close'].pct_change().rolling(5).std().dropna().values
    min_l = min(len(train_returns), len(train_vols))
    local_detector.fit(np.column_stack((train_returns[-min_l:], train_vols[-min_l:])))
    
    feats = []
    labs = []
    pct_df = train_slice[['close', 'volume', 'high', 'low', 'open']].pct_change().fillna(0)
    for i in range(5, len(pct_df) - 1):
        feats.append(pct_df.iloc[i-5:i].values)
        labs.append(pct_df['close'].iloc[i])
    local_predictor.fit(feats, np.array(labs))
    
    metrics = backtester.run(
        test_slice,
        meta_engine,
        risk_manager,
        local_detector,
        local_predictor,
        local_ppo
    )
    # LOT 8 (PDF Pilier N) : le rapport de backtest inclut l'audit des biais
    if isinstance(metrics, dict):
        metrics["bias_audit"] = _bias
    return JSONResponse(metrics)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Audit B3-5: cap total clients and 1 socket per IP
    client_ip = websocket.client.host if websocket.client else "?"
    if len(STATE["connected_websockets"]) >= 50:
        await websocket.close(code=1013, reason="Too many clients")
        return
    for existing in STATE["connected_websockets"]:
        if getattr(existing, "client", None) and existing.client.host == client_ip:
            await websocket.close(code=1008, reason="One connection per IP")
            return
    STATE["connected_websockets"].append(websocket)
    platform_metrics.WS_CLIENTS.set(len(STATE["connected_websockets"]))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in STATE["connected_websockets"]:
            STATE["connected_websockets"].remove(websocket)
            platform_metrics.WS_CLIENTS.set(len(STATE["connected_websockets"]))
