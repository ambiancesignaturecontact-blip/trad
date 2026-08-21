from dotenv import load_dotenv

load_dotenv()  # .env (secrets) before any env consumers

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import websockets
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from bot.telegram_bot import TelegramBotManager
from copytrading.manager import CopyTradingManager
from core.attribution import PerformanceAttribution
from core.confidence_index import compute_confidence_index
from core.config import settings
from core.cost_accounting import CostAccounting
from core.counterparty_risk import CounterpartyRiskManager
from core.execution_agent import ExecutionStyleBandit, StrategyExecutionAttribution, tradability_factor
from core.execution_router import ExecutionAlpha, SlippageModel, decide_style
from core.hypothesis_generator import HypothesisGenerator
from core.meta_cognition import adaptive_conviction_threshold, decide_no_trade, hedging_decision
from core.middleware import (
    IPRateLimitMiddleware,
    LoginRateLimitMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
    install_cors,
)
from core.mixture_experts import MixtureOfExperts, risk_adjusted_reward
from core.organization import Organization
from core.paper_execution import estimate_slippage_bps_from_book, min_notional_for_capital, simulate_paper_fill
from core.portfolio_allocator import PortfolioAllocator
from core.position_manager import (
    PositionProtection,
    PositionProtectionStore,
    apply_breakeven_stop,
    can_pyramid,
    evaluate_protection,
    evaluate_time_stop,
    partial_take_profit,
    position_age_hours,
)

# LOT B (F2) : autonomie stratégique — auto-adaptation bornée des paramètres
# de risque au régime HMM (facteur lissé EMA, borné [0.60, 1.25]).
from core.regime_autonomy import RegimeAutonomy
# LOT D (F4) : détection de drift par PSI (online learning) — surveille la
# distribution des features clés et accélère l'oubli du bandit au drift.
from core.drift_psi import MultiAssetDriftMonitor, run_drift_check
from core.research_discipline import live_p_value, meta_label_filter
from core.risk_committee import RiskCommittee
from core.risk_pipeline import (
    ATR_MULT_SL,
    REWARD_RISK_RATIO,
    ROUND_TRIP_COST_PCT,
    STOP_LOSS_PCT,
    RiskStateMachine,
    StrategyWinRateTracker,
    apply_risk_pipeline,
    calibrated_conviction,
    entry_rr_filter,
)
from core.robustness import (
    Supervisor,
    restore_state_snapshot,
)
from core.volatility_targeting import volatility_scale_factor
from core.world_model import (
    compute_market_state,
    compute_regime_probs,
    compute_structural_regimes,
    counterfactual_alpha,
    cross_asset_bias,
)
from database.db_manager import DBManager
from market_data.multi_source import MultiSourcePriceEngine
from market_data.order_flow import OrderFlowEngine
from models.almgren_chriss import AlmgrenChrissExecutionOptimizer, calculate_cvar_constrained_sizing
from models.defi_wallet import NonCustodialDeFiWallet
from models.dex_cex_arbitrage import DexCexArbitrageEngine
from models.execution_slicer import SmartOrderSlicer
from models.funding_arbitrage import FundingRateArbitrageEngine
from models.macro_calendar import MacroeconomicCalendarEngine
from models.microstructure_edge import MicrostructureEdgeEngine
from models.mlops_pipeline import MLOpsAutoTrainer
from models.monte_carlo import MonteCarloStressTester
from models.oms_ems import OrderManagementSystem, ReconciliationEngine
from models.onchain_tracker import OnChainTracker
from models.price_predictor import LSTMLikePredictor, PPOTRAgent

# Import our quant models
from models.regime_detector import MarketRegimeDetector
from models.risk_covariance import RiskCovarianceEngine
from models.scenario_stress import ScenarioStressTester

# NEW ADVANCED MODELS
from models.sentiment_analyzer import NewsSentimentAnalyzer
from models.volatility_arbitrage import OptionsVolatilityArbitrageEngine
from risk.risk_manager import RiskManager
from strategies.engine import (
    ArbitrageInterExchangeStrategy,
    GridTradingStrategy,
    MarketMakingStrategy,
    MeanReversionStrategy,
    MetaAllocationEngine,
    ScalpingStrategy,
    StatisticalArbitrageStrategy,
    TrendFollowingStrategy,
)

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
from strategies.institutional import CarryStrategy, CrossSectionalMomentumStrategy, MultiTimeframeWrapperStrategy
from strategies.momentum import MomentumStrategy
from strategies.regime_switching import RegimeSwitchingAllocator
from strategies.volatility_breakout import VolatilityBreakoutStrategy

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

# LOT B (F2) : autonomie stratégique — facteur d'agressivité de risque piloté
# par le régime HMM (jamais > 1.25x la config de base, drawdowns jamais
# élargis). Appliqué à risk_manager via apply_regime_factor() à chaque tick.
regime_autonomy = RegimeAutonomy()

# LOT D (F4) : moniteur de drift PSI MULTI-ACTIFS — fenêtres de
# référence/récente sur les candles RÉELLES de CHAQUE actif ; le decay du
# bandit est accéléré quand le pire PSI est sévère, fusionné avec le CUSUM.
# Les symboles sont passés à chaque appel (STATE["assets"] n'existe qu'à
# l'exécution de la boucle, pas au moment de la création de l'instance).
drift_monitor = MultiAssetDriftMonitor()

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

# === LOT 55 (module ÉDUCATIF — P2-19) ===
# rl/rlhf_reward_model.py reste dans le repo, étiqueté ÉDUCATIF dans le
# registre module_honesty. Il n'est PLUS chargé ici : un module ÉDUCATIF ne
# doit JAMAIS influencer une décision de trading (audit §2.4). L'implémentation
# reste testée unitairement (tests/test_lot9_honesty.py).

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
from fastapi.security import HTTPBearer

from core.rate_limits import bybit_limiter
from database.auth import AuthManager, Roles

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
    "regime_autonomy": {},            # LOT B (F2): autonomie stratégique (facteur + effectifs)
    "drift_psi": {},                  # LOT D (F4): drift distribution (PSI) + decay bandit
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
        # Axe 2 (mission intelligence) : le RegimeSwitchingAllocator apprend
        # la performance RÉELLE par (régime, stratégie) — les poids de régime
        # s'adaptent en ligne, bornés (jamais de sur-réaction au bruit).
        try:
            regime_allocator.update_regime_performance(
                STATE.get("regime_id", 2), strategy, pnl_pct)
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



def _is_remote_deployment() -> bool:
    """
    P0-2 (audit indépendant §2.9): détecte un déploiement non-local.
    Considéré non-local si une variable RAILWAY_* est présente (Railway) ou si
    PORT est défini dans l'environnement (convention PaaS/Railway) — une URL
    publique est alors exposée et l'authentification devient OBLIGATOIRE.
    """
    if any(k.startswith("RAILWAY_") for k in os.environ):
        return True
    port = os.getenv("PORT", "").strip()
    return port.isdigit() and int(port) > 0


def auth_enforced() -> bool:
    """
    L'authentification est-elle exigée sur les routes d'action ?
     - AUTH_ENABLED=true explicite -> OUI (priorité absolue)
     - AUTH_ENABLED=false explicite -> NON (choix assumé de l'opérateur,
       ex. sandbox de dev/preview ; prioritaire sur la détection d'environnement)
     - sinon : mode REAL (argent réel : jamais contrôlable sans session) OU
       déploiement non-local (PORT/RAILWAY_* -> URL publique, P0-2) -> OUI.
    """
    explicit = os.getenv("AUTH_ENABLED", "").strip().lower()
    if explicit in ("1", "true", "yes", "on"):
        return True
    if STATE["mode"] == "REAL":
        return True
    if explicit in ("0", "false", "no", "off"):
        return False
    return _is_remote_deployment()


def require_auth(credentials=Depends(auth_security_optional)):
    """
    Protects state-changing endpoints.
    Enforced when AUTH_ENABLED=true, OR in REAL mode, OR on any non-local
    deployment (P0-2: PORT/RAILWAY_* -> an exposed URL must never be open
    without authentication, even in DEMO mode).
    """
    if not auth_enforced():
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


# ============ P0-4 (audit indépendant §2.1) : distribution réelle de final_scale ============
# 17 facteurs multiplicatifs en chaîne peuvent s'auto-amplifier silencieusement
# (0.8^15 ≈ 3.5% de la taille de base). On accumule les final_scale observés en
# live et on loggue p10/p50/p90 sur la fenêtre glissante de 48h — si p50 < 20%,
# le problème est la chaîne de facteurs, pas le seuil de signal.

FINAL_SCALE_DOWNSAMPLE_SEC = 60.0   # 1 échantillon / min / symbole max
FINAL_SCALE_WINDOW_HOURS = 48.0     # fenêtre glissante de référence
FINAL_SCALE_MAX_SAMPLES = 25000     # borne dure mémoire


# ============ P0-6 (audit §5-P0-6) : suivi du paper-trading DATÉ ============
# L'historique de paper-trading validé avant REAL doit être réel, daté et
# CONTINU (4-8 semaines). Ce tracker marque chaque jour où le bot tourne
# (persisté en DB) et expose la série + le statut de validation — un chiffre
# de vérité que le code ne peut pas truquer : si le bot ne tourne pas, le
# jour n'est pas compté.

execution_bandit = ExecutionStyleBandit()
strategy_exec_attr = StrategyExecutionAttribution()

# VISION_FUTUR instances
organization = Organization(STATE)
supervisor = Supervisor(STATE)

# CCXT Exchange Client Cache


# AUDIT B6-1: short TTL cache for Yahoo chart calls (rate-limit friendly)
_yahoo_cache: dict = {}


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


async def pick_best_venue_net(symbol: str, side: str, qty: float = None) -> dict:
    """
    LOT 3 (PDF Pilier H-3) : SOR multi-venue — compare les venues sur le COÛT
    NET TOTAL (prix + frais + slippage attendu), pas seulement le prix.
    P1-13 (audit §4.6) : quand la taille du trade est fournie, le slippage de
    chaque venue est estimé par BOOK-WALKING de son carnet réel (si dispo),
    sinon par le modèle des fills réels, sinon fallback prudent 5 bps.

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
                # P1-13 : book-walking du carnet réel de CETTE venue quand dispo
                if qty is not None and qty > 0:
                    try:
                        _ex_key = q.exchange.lower()
                        _book = STATE.get("exchange_order_books", {}).get(_ex_key, {}).get(symbol)
                        if _book is None:  # consolidation BBO toutes venues
                            _book = STATE.get("order_books", {}).get(symbol)
                        _bps = estimate_slippage_bps_from_book(side, qty, _book, q.ask)
                        if _bps is not None:
                            slip = _bps / 1e4
                    except Exception:
                        pass
                net = gross * (1.0 + slip)
            else:
                gross = q.bid * (1.0 - q.fee_rate)
                slip = slippage_model.expected_slippage_bps(q.exchange.capitalize(), symbol, fallback=5.0) / 1e4
                if qty is not None and qty > 0:
                    try:
                        _ex_key = q.exchange.lower()
                        _book = STATE.get("exchange_order_books", {}).get(_ex_key, {}).get(symbol)
                        if _book is None:
                            _book = STATE.get("order_books", {}).get(symbol)
                        _bps = estimate_slippage_bps_from_book(side, qty, _book, q.bid)
                        if _bps is not None:
                            slip = _bps / 1e4
                    except Exception:
                        pass
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
    "final_scale_stats": lambda: final_scale_stats_loop(),
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


def _telegram_send_sync(token: str, chat_id: str, text: str) -> bool:
    """Envoi Telegram best-effort synchrone (canal dédié pour un secret au boot).
    Ne lève jamais : retourne False en cas d'échec."""
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text},
            )
            return resp.status_code == 200
    except Exception:
        return False


def _deliver_admin_password_once(admin_pass: str, creds_path: str = None) -> str:
    """
    P0-3 (audit indépendant §2.10): livre le mot de passe admin auto-généré par
    un canal dédié — JAMAIS dans les logs applicatifs (les logs sont souvent
    centralisés/exportés vers des tiers).
    Ordre : 1) Telegram DM si TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID configurés,
            2) fichier .admin_credentials en mode 0600, 3) aucun canal.
    Retourne "telegram" | "file" | "none".
    """
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if tg_token and tg_chat:
        try:
            ok = _telegram_send_sync(
                tg_token,
                tg_chat,
                "🔐 QUANT-PORTAL : mot de passe admin auto-généré "
                "(secret — ne pas partager, ne pas logger).\n"
                "username=admin_quant\n"
                f"password={admin_pass}\n"
                "Définissez ADMIN_PASSWORD (env) puis redémarrez pour le remplacer.",
            )
            if ok:
                return "telegram"
        except Exception:
            pass
    path = Path(creds_path) if creds_path else Path(os.path.dirname(os.path.abspath(__file__))) / ".admin_credentials"
    try:
        path.write_text(
            "# QUANT-PORTAL — mot de passe admin auto-généré (SECRET).\n"
            "# Ne jamais committer, logger ni partager. Supprimez ce fichier après\n"
            "# la première connexion et définissez ADMIN_PASSWORD (env) à la place.\n"
            f"username=admin_quant\npassword={admin_pass}\n",
            encoding="utf-8",
        )
        os.chmod(path, 0o600)
        return "file"
    except Exception:
        return "none"


def _ensure_auth_secrets(creds_path: str = None) -> None:
    """
    Audit B3-3/B3-4 + P0-2/P0-3 : auto-génère des secrets forts au premier boot
    (jamais de défaut prévisible), les persiste chiffrés en DB et injecte dans
    l'environnement. Ne JAMAIS logguer un secret en clair : le mot de passe
    admin auto-généré est livré via _deliver_admin_password_once() uniquement.
    """
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
            # P0-3: le secret n'apparaît JAMAIS dans les logs — canal dédié.
            delivered = _deliver_admin_password_once(admin_pass, creds_path)
            if delivered == "telegram":
                logger.warning(
                    "🔐 AUTH: auto-generated admin password — livré par Telegram "
                    "(jamais loggé). Définissez ADMIN_PASSWORD env pour le remplacer."
                )
            elif delivered == "file":
                logger.warning(
                    "🔐 AUTH: auto-generated admin password — écrit dans "
                    ".admin_credentials (0600, jamais loggé). Supprimez ce fichier "
                    "après la première connexion et définissez ADMIN_PASSWORD env."
                )
            else:
                logger.warning(
                    "🔐 AUTH: auto-generated admin password (jamais loggé, aucun "
                    "canal de livraison configuré). Définissez ADMIN_PASSWORD env "
                    "pour garder la main sur l'accès."
                )
    else:
        # env password provided: make sure the DB hash is in sync
        try:
            hashed = _bcrypt.hashpw(admin_pass.encode(), _bcrypt.gensalt()).decode()
            db.upsert_admin(hashed, Roles.ADMIN)
        except Exception:
            pass


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

    # 5. Authentication (P0-2 audit indépendant §2.9)
    production_auth = auth_enforced()
    if _is_remote_deployment():
        checks.append(("Authentication", "FORCED on (non-local deployment detected)"))
    elif production_auth:
        checks.append(("Authentication", "on (AUTH_ENABLED=true or REAL mode)"))
    else:
        checks.append(("Authentication", "off (local DEMO only)"))

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

    # Audit B3-3/B3-4 + P0-2/P0-3 (audit indépendant §2.9/§2.10): jamais de
    # secrets par défaut prévisibles ; AUTH forcée sur tout déploiement
    # non-local (PORT/RAILWAY_* -> URL publique) ; aucun secret loggé en clair.
    if production_auth:
        _ensure_auth_secrets()


class MacroOverrideRequest(BaseModel):
    """Pilotage humain du risque macro (LOT 5, PDF Pilier I)."""
    action: str = Field(pattern="^(reduce|halt|reset)$")
    factor: float = Field(default=0.5, ge=0.05, le=1.0)


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


# ===== AUDIT C3: custom price alerts =====
class PriceAlertCreate(BaseModel):
    symbol: str = Field(min_length=3, max_length=20)
    direction: str = Field(pattern="^(above|below)$")
    target_price: float = Field(gt=0.0)
    note: str = Field(default="", max_length=200)


# ===== AUDIT C10: market replay =====
class ReplayRequest(BaseModel):
    symbol: str = Field(min_length=3, max_length=20)
    interval: str = Field(default="1h", pattern="^(1m|5m|15m|1h|4h|1d)$")
    limit: int = Field(default=300, ge=60, le=2000)


# ===== VISION §2.1/§2.2: signal admission gate + experiment registry =====
class SignalEvalRequest(BaseModel):
    symbol: str = Field(min_length=3, max_length=20)
    limit: int = Field(default=300, ge=100, le=2000)


class ExperimentCreate(BaseModel):
    hypothesis: str = Field(min_length=3, max_length=500)


# ===== VISION §7.1 replayable event journal + §6 factor model =====


# ===== VISION endpoints =====


# ===== VISION_FUTUR endpoints =====


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


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


async def startup_event():
    # LOT 62: institutional configuration checklist
    validate_startup_config()

    # P0-6 : premier marquage du jour de paper-trading (le bot démarre)
    try:
        _mark_paper_validation_day()
    except Exception:
        pass

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
        # LOT D (F4, corrigé) : seuil de profondeur 250 au lieu de 10 — le
        # PSI (drift distribution) exige ~550 barres pour un calcul fiable ;
        # le fetch profond (700 barres) met le cache à niveau au premier boot
        # du nouveau code. Les vieux caches à 120 barres sont refetchés une
        # fois, puis l'historique s'accumule normalement.
        if _df.empty or len(_df) < 250:
            logger.info(f"Database cache incomplete for {_sym} ({len(_df)} barres). Fetching from real APIs (Binance/Bybit/Yahoo)...")
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

    # P0-4 (audit §2.1) : distribution de final_scale (p10/p50/p90) toutes les 60 min
    launch_named(final_scale_stats_loop(), "final_scale_stats")
    logger.info("✅ P0-4: final_scale distribution stats loop started (every 60 min)")

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
                                "🔴 *HALT MÉDIA* — choc systémique détecté. Nouveaux ordres bloqués."))
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
                                    # P0-1 (audit §2.3): message honnête — la tx est
                                    # broadcastée mais AUCUNE protection anti-sandwich
                                    # on-chain n'est active (contrat non déployé).
                                    await telegram_bot.send_push_notification(
                                        f"✅ *ARBITRAGE DEX-CEX EXÉCUTÉ* (broadcast on-chain)\n"
                                        f"-----------------------------------------\n"
                                        f"📈 Actif : `{symbol}`\n"
                                        f"⚖️ Route : *{route}*\n"
                                        f"📊 Écart : *{spread*100:.2f}%* — gain net estimé : *+{profit_pct*100:.2f}%*\n"
                                        f"🔗 Tx : `{bcast.get('tx_hash', 'voir audit log')}`\n"
                                        f"⚠️ Aucune protection anti-sandwich on-chain active."
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
                                f"🔎 *ARBITRAGE DEX-CEX DÉTECTÉ — NON EXÉCUTÉ (signal-only)*\n"
                                f"-----------------------------------------\n"
                                f"📈 Actif : `{symbol}`\n"
                                f"⚖️ Route : *{route}* (écart {spread*100:.2f}%)\n"
                                f"💵 Gain net estimé si exécuté : *+{profit_pct*100:.2f}% (net de gaz)*\n"
                                f"ℹ️ Exécution désactivée : mode signal-only "
                                f"(ARBITRAGE_EXECUTION={arb_exec_mode}). Réglez "
                                f"ARBITRAGE_EXECUTION=auto + EVM_PRIVATE_KEY pour une "
                                f"vraie exécution."
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

                    # LOT B (F2) : AUTONOMIE STRATÉGIQUE — auto-adaptation BORNÉE
                    # des paramètres de risque au régime HMM. Facteur lissé EMA,
                    # borné [FACTOR_MIN, FACTOR_MAX] (jamais > 1.25x la config de
                    # base) ; les drawdowns ne sont JAMAIS élargis. Même chemin
                    # de code en DEMO et en REAL (aucun flag de mode ici).
                    try:
                        _regime_conf = STATE.get("regime_confidence", {})
                        _rconf = float(_regime_conf.get("confidence", 0.5)) \
                            if isinstance(_regime_conf, dict) else 0.5
                        regime_autonomy.update(STATE["regime_id"], _rconf)
                        risk_manager.apply_regime_factor(regime_autonomy.factor)
                        STATE["regime_autonomy"] = dict(regime_autonomy.to_dict())
                        STATE["regime_autonomy"]["regime_name"] = STATE.get("regime_name", "")
                        STATE["regime_autonomy"]["effective"] = risk_manager.effective_params()
                    except Exception as _ra_e:
                        # Jamais bloquant : sans autonomie, on retombe sur la
                        # config de base (facteur 1.0 = comportement pré-LOT B).
                        logger.warning(f"Regime autonomy failed ({_ra_e}) — facteur 1.0 (config de base).")
                        STATE["regime_autonomy"] = {"enabled": False, "error": str(_ra_e)}

                    # LOT D (F4, corrigé) : DÉTECTION DE DRIFT — PSI MULTI-ACTIFS
                    # (distribution des features par actif, pire cas global)
                    # FUSIONNÉ au CUSUM (erreur de prédiction du modèle). Drift
                    # sévère -> oubli du bandit ACCÉLÉRÉ. Jamais bloquant.
                    run_drift_check(STATE, db, drift_monitor, meta_engine, audit_ip,
                                    logger=logger, symbols=list(STATE["assets"].keys()))

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
                        # LOT D (F4, corrigé) : FUSION CUSUM + PSI — un drift de
                        # prédiction est aussi un signal d'oubli accéléré pour le
                        # bandit (l'edge des stratégies peut avoir changé). Le flag
                        # reste actif DRIFT_CUSUM_HOLD_SECONDS puis retombe.
                        try:
                            STATE["drift_cusum"] = {"detected": True,
                                                    "ts": time.time(),
                                                    "detail": "erreur de prédiction LSTM (CUSUM)"}
                            # accélération immédiate (le prochain tick PSI
                            # confirmera via la fusion)
                            from core.drift_psi import BANDIT_DECAY_DRIFT
                            meta_engine.set_bandit_decay(
                                min(meta_engine.bandit_decay, BANDIT_DECAY_DRIFT))
                            db.add_audit_log(
                                "DRIFT_CUSUM_DETECTED", audit_ip(),
                                f"Concept drift CUSUM détecté — retraining + oubli bandit "
                                f"{meta_engine.bandit_decay:.4f} (fusion PSI/CUSUM).")
                        except Exception as _cu_e:
                            logger.warning(f"CUSUM drift flag failed ({_cu_e})")

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

                    # 2. MAX PER-ASSET CAP 25 % (plafond dur, étape 2 du pipeline).
                    # LOT B (F2) : plafond EFFECTIF — config de base × facteur de
                    # régime (borné [0.10, 0.30], jamais > 1.25x la base). La
                    # même valeur effective est appliquée par RiskManager au
                    # sizing et affichée en télémétrie (une seule source).
                    try:
                        max_asset_pct = risk_manager.effective_params()["max_exposure_per_asset_pct"]
                    except Exception:
                        max_asset_pct = settings.get_float("risk", "max_per_asset_pct", 0.25)
                    try:
                        _user_max = float(os.getenv("USER_MAX_EXPOSURE_PCT", "0"))
                        if 0 < _user_max < 1:
                            max_asset_pct = min(max_asset_pct, _user_max)
                    except Exception:
                        pass
                    max_asset_qty = (STATE[active_balance_key] * max_asset_pct) / current_price

                    # VISION_FUTUR §2d: meta-label filter — filtre les faux signaux avant
                    # exécution (López de Prado).
                    # FIX (mini-app 50$) : warm-up porté à 20 trades clôturés et,
                    # en DEMO, un win rate faible RÉDUIT la taille au lieu de
                    # BLOQUER totalement — sinon le bot arrête de trader après 5
                    # trades perdants et ne peut plus JAMAIS apprendre (piège
                    # observé en production : 'wr 0.12 (n=5)' -> aucun trade).
                    # En REAL, le filtre reste bloquant (prudence maximale).
                    try:
                        _ml_count = win_tracker.samples(_dom_early)
                        _ml_ok = meta_label_filter(_dom_early,
                                                   STATE.get("strategy_win_rates", {}),
                                                   counts=STATE.get("strategy_trade_counts", {}),
                                                   min_samples=20)
                        if not _ml_ok:
                            if active_mode == "REAL":
                                decide_no_trade(symbol, final_signal, 0.999,
                                                [f"meta-label REAL: wr {STATE.get('strategy_win_rates', {}).get(_dom_early, 0.0):.2f} (n={_ml_count})"],
                                                STATE["no_trade_stats"], db)
                                continue
                            # DEMO : pas de blocage total — le win rate faible
                            # réduit la taille (le bot continue d'apprendre)
                            target_qty *= max(0.25, STATE.get("strategy_win_rates", {}).get(_dom_early, 0.0) / 0.52)
                            STATE["meta_label_scale"] = max(0.25, STATE.get("strategy_win_rates", {}).get(_dom_early, 0.0) / 0.52)
                            logger.info(f"META-LABEL DEMO {symbol}: wr {STATE.get('strategy_win_rates', {}).get(_dom_early, 0.0):.2f} (n={_ml_count}) -> taille x{STATE['meta_label_scale']:.2f} (apprentissage continu)")
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
                        if _cash_s < 0.99:
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
                    # 12. RLHF — P2-19 (audit §2.4) : module ÉDUCATIF, JAMAIS
                    # dans le sizing. Le facteur du pipeline est 1.0 constant
                    # (voir rlhf_scale dans l'appel ci-dessous).
                    # 13. Volatilité cible
                    _vol_scale = 1.0
                    try:
                        _vol_scale = volatility_scale_factor(STATE[active_equity_history_key])
                        STATE["vol_target_scale"] = _vol_scale
                    except Exception:
                        pass
                    # 14. Tradabilité / slippage attendu
                    # P1-13 (audit §4.6) : le slippage est d'abord estimé par
                    # BOOK-WALKING du carnet consolidé réel (profondeur réelle,
                    # taille du trade), puis par le modèle des fills réels,
                    # puis fallback prudent 5 bps. Fini le slippage fixe seul.
                    _trad_s = 1.0
                    try:
                        _slip_avg = None
                        try:
                            _book = STATE.get("order_books", {}).get(symbol)
                            _side = "BUY" if final_signal > 0 else "SELL"
                            _slip_avg = estimate_slippage_bps_from_book(
                                _side, target_qty, _book, current_price)
                            if _slip_avg is not None:
                                STATE["book_slippage_bps"] = {
                                    "symbol": symbol, "bps": round(float(_slip_avg), 2),
                                    "method": "book_walk", "ts": time.time()}
                        except Exception:
                            _slip_avg = None
                        if _slip_avg is None:
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
                            # Axe 1 (mission intelligence) : conviction CALIBRÉE par le
                            # win rate réel de la stratégie dominante (meta-labeling —
                            # la taille reflète la probabilité calibrée de succès, pas
                            # seulement l'intensité du signal). Le win rate vient du
                            # tracker réel (lissé EMA, borné 0.45-0.65).
                            conviction=calibrated_conviction(
                                final_signal,
                                win_tracker.get(_dom_kelly) if _dom_kelly else None,
                            ),
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
                            rlhf_scale=1.0,  # P2-19 : rlhf ÉDUCATIF -> JAMAIS d'influence sizing
                            vol_scale=_vol_scale,
                            tradability_scale=_trad_s,
                        )
                        target_qty = _pipe["qty"]
                        STATE["risk_pipeline_steps"] = _pipe["steps"]
                        risk_pipeline_last.update({"symbol": symbol, "final_scale": _pipe["final_scale"],
                                                   "n_steps": len(_pipe["steps"]),
                                                   "active_factors": _pipe.get("active_factors", 0)})
                        # P0-4 (audit §2.1) : accumulation de la distribution réelle
                        # + facteur limitant (idée n°1) à partir des steps du pipeline
                        _record_final_scale(symbol, _pipe["final_scale"], len(_pipe["steps"]),
                                            steps=_pipe.get("steps"))
                        # FIX (trading micro 50$) : avec un signal faible, la taille
                        # post-pipeline tombe SOUS le min notional (ex. 1,50$ < 3$)
                        # -> l'exécution REJETTE -> le bot ne trade JAMAIS sur petit
                        # compte. Comportement UNIQUE DEMO == REAL (fidélité) :
                        # arrondi d'exécution réaliste — en REAL aussi, un ordre sous
                        # le min serait rejeté par l'exchange, et le pipeline a déjà
                        # validé le trade (target_qty > 0). La remontée est bornée :
                        #   - au min notional (3$ < 200$, 5$ < 1000$, 10$ sinon)
                        #   - à 80 % du capital (jamais plus, cohérent avec
                        #     calculate_position_size)
                        #   - aucun effet si le capital ne permet pas le min
                        # Le risque reste donc plafonné au min notional réel.
                        try:
                            if target_qty > 0:
                                _mn = min_notional_for_capital(STATE[active_balance_key])
                                _notional = target_qty * current_price
                                if _notional < _mn:
                                    _cap_qty = (STATE[active_balance_key] * 0.80) / current_price
                                    _bumped = min(_mn / current_price, _cap_qty)
                                    if _bumped > target_qty:
                                        logger.info(
                                            f"micro-budget {symbol} ({active_mode}): notional {_notional:.2f}$ "
                                            f"< min {_mn:.2f}$ -> arrondi exécution à {_bumped*current_price:.2f}$ "
                                            f"(plafond 80% capital).")
                                        target_qty = _bumped
                        except Exception:
                            pass
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
                            # LOT A (F1) : base ADAPTATIVE (p25 des |signaux|
                            # réels) au lieu de la constante 0.08 qui bornait
                            # le seuil à [0.08, 0.14] — le seuil suit la
                            # conviction réellement produite par le marché.
                            base_threshold=None)
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
                        # FIX (ruff F821) : dominant_strategy était utilisé dans le
                        # bloc consultative_mode AVANT sa définition (NameError au
                        # premier passage). Initialisée ici, dès l'entrée.
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

                            # PER-MODEL ATTRIBUTION : dominant_strategy est déjà
                            # initialisée à l'entrée du bloc d'exécution (FIX F821).

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
                                        _sor = await pick_best_venue_net(symbol, side, qty=trade_qty_formatted)
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
                                    # LOT 7 (fidélité DEMO == REAL) : le SOR multi-venue est
                                    # exécuté en DEMO aussi (venue choisie au coût net, prix +
                                    # frais + slippage book-walké — comme en REAL), et le fill
                                    # paper passe TOUJOURS par simulate_paper_fill : book-walking
                                    # réel du carnet de la venue, partial fills, rejets liquidité.
                                    # (Avant : le fill était au prix fixe ±3 bps quand un carnet
                                    # était présent — simulate_paper_fill n'était appelé que si
                                    # le carnet était ABSENT. L'inverse de la haute fidélité.)
                                    _sor_venue = None
                                    try:
                                        _sor = await pick_best_venue_net(symbol, side, qty=trade_qty_formatted)
                                        STATE["last_sor_choice"] = _sor
                                        if _sor.get("venue"):
                                            _sor_venue = _sor["venue"]
                                            logger.info(
                                                f"SOR-DEMO {symbol} {side}: venue {_sor_venue} "
                                                f"(net {_sor['net_price']:.6f}) | {_sor['reason']}")
                                            db.add_audit_log(
                                                "SOR_CHOICE_DEMO", audit_ip(),
                                                f"{symbol} {side}: venue {_sor_venue} (net {_sor['net_price']:.6f})")
                                    except Exception:
                                        _sor_venue = None
                                    _paper_book = None
                                    if _sor_venue:
                                        _paper_book = STATE.get("exchange_order_books", {}).get(
                                            _sor_venue.lower(), {}).get(symbol)
                                    if _paper_book is None:
                                        _paper_book = STATE.get("order_books", {}).get(symbol)
                                    _paper_venue = _sor_venue if _sor_venue else (
                                        "Binance" if symbol in ("BTCUSDT","ETHUSDT","SOLUSDT") else "Bybit")
                                    _paper = simulate_paper_fill(
                                        symbol=symbol, side=side, qty=trade_qty_formatted,
                                        arrival_price=current_price,
                                        order_book=_paper_book,
                                        venue=_paper_venue,
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
                                    slippage_model.update(_paper_venue, symbol, _paper["slippage_bps"])
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
                    realized_vol_annual=None,
                    regime_id=STATE.get("regime_id"))
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


# ============ AUTHENTICATION (roadmap #4) ============
class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)
    totp_code: str = Field(default="", max_length=16)


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

# ============ LOT C (F3) : gros blocs extraits ============
# Les définitions ont été déplacées vers des modules dédiés ; on
# ré-exporte les noms pour préserver l'espace de noms de main (tests,
# TASK_FACTORIES, api/routes.py, schedulers.py, telemetry.py). Ordre
# de dépendances respecté (autonomous_ai dépend de fetch_* ré-exporté).
# CE BLOC VIENT AVANT les imports de schedulers/telemetry/routes : ceux-ci
# consomment les ré-exports (ex. schedulers importe _final_scale_report).
from core.observability import _final_scale_report, _final_scale_stats, _limiting_factor_stats, _load_final_scale_samples, _mark_paper_validation_day, _paper_validation_stats, _persist_final_scale_samples, _purge_final_scale_samples, _record_final_scale, _signal_stats  # noqa: F401,E402
from core.ccxt_client import format_exchange_size, get_ccxt_client  # noqa: F401,E402
from market_data.historical_fetch import _klines_to_df, fetch_bybit_klines, fetch_historical_market_data, fetch_yahoo_finance_candles  # noqa: F401,E402
from core.autonomous_ai import autonomous_ai_scheduler  # noqa: F401,E402
from core.decision_explain import explain_last_decision, update_metrics_from_state  # noqa: F401,E402

# ============ étape 2 du découpage (LOT 7) : télémétrie ============
# ============ LOT 7 (P1-7 audit §4.1) : modules extraits ============
# Les définitions ont été déplacées vers api/routes.py et schedulers.py ;
# on ré-exporte les noms pour préserver l'espace de noms de main (les
# tests et TASK_FACTORIES y accèdent) et on monte le router des routes API.
from api.routes import router as _api_router  # noqa: E402
from schedulers import (  # noqa: F401,E402
    concierge_scheduler,
    copy_mirror_scheduler,
    copy_trading_refresh_scheduler,
    db_backup_scheduler,
    final_scale_stats_loop,
    reconciliation_scheduler,
)
from telemetry import broadcast_telemetry, compile_telemetry_data, serialize_helper  # noqa: F401,E402

app.include_router(_api_router)
