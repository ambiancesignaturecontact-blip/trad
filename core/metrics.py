"""
Institutional Prometheus metrics registry.

Exposes a /metrics endpoint (Prometheus text format) so the bundled
prometheus.yml + Grafana stack (docker-compose.monitoring.yml) can actually
scrape the platform. Without this, the monitoring stack was dead code.

All gauges/counters are updated from the trading loop via update_* helpers.
"""
import time

from prometheus_client import Counter, Gauge, Histogram, generate_latest, REGISTRY

# --- Core runtime gauges ---
UPTIME_SECONDS = Gauge("quant_uptime_seconds", "Seconds since platform start")
LAST_LOOP_DURATION = Gauge("quant_last_loop_duration_seconds", "Duration of the last trading-loop tick")
WS_CLIENTS = Gauge("quant_websocket_clients", "Currently connected dashboard WebSockets")

MARKET_LAST_PRICE = Gauge("quant_market_price", "Last known price per symbol", ["symbol"])
MARKET_EQUITY = Gauge("quant_equity", "Current equity (active mode)", ["mode"])
MARKET_BALANCE = Gauge("quant_balance", "Available balance (active mode)", ["mode"])
MARKET_PNL_USD = Gauge("quant_pnl_usd", "Live PnL in USD", ["mode"])
MARKET_PNL_PCT = Gauge("quant_pnl_pct", "Live PnL in percent", ["mode"])

# --- Activity counters ---
ORDERS_TOTAL = Counter("quant_orders_total", "Orders placed (mode=DEMO/REAL, side=BUY/SELL)", ["mode", "side"])
ORDERS_FAILED = Counter("quant_orders_failed_total", "Rejected/failed orders", ["mode"])
ERRORS_TOTAL = Counter("quant_errors_total", "Caught exceptions in the trading loop", ["component"])
WS_MESSAGES = Counter("quant_ws_messages_total", "Telemetry frames broadcast", ["channel"])

# --- Risk & regime ---
REGIME_ID = Gauge("quant_regime_id", "Current HMM regime id")
REGIME_NAME = Gauge("quant_regime_name", "Current regime (hashed for label safety)", ["name"])
RISK_EXPOSURE = Gauge("quant_risk_exposure_pct", "Current portfolio exposure percent")
RISK_CVAR = Gauge("quant_risk_cvar_pct", "Latest portfolio CVaR percent")
POSITIONS_OPEN = Gauge("quant_positions_open", "Number of open positions")
SENTIMENT_INDEX = Gauge("quant_sentiment_index", "Latest live sentiment index")
ONCHAIN_RISK = Gauge("quant_onchain_risk_score", "Latest on-chain risk score")

# --- Data quality ---
DATA_QUALITY = Gauge("quant_data_quality", "Data quality status (0=UNAVAILABLE..4=LIVE)", ["source"])

# --- AI / autonomy (audit B9-3) ---
AI_OOS_SHARPE = Gauge("quant_ai_oos_sharpe", "Latest walk-forward out-of-sample Sharpe")
AI_PPO_BUFFER = Gauge("quant_ai_ppo_buffer", "RL experiences collected for PPO training")
AI_LAST_CYCLE = Gauge("quant_ai_last_cycle_ts", "Timestamp of the last autonomous AI cycle")
AI_MODEL_ERROR = Gauge("quant_ai_model_error", "Rolling prediction error (|pred - actual|)", ["model"])

# --- Latency histogram for outbound API calls ---
API_LATENCY = Histogram(
    "quant_api_latency_seconds",
    "Outbound exchange/data API call latency",
    ["endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

_START_TIME = time.time()


def get_metrics_text() -> bytes:
    """Returns the Prometheus text exposition for the default registry."""
    return generate_latest(REGISTRY)


def record_api_latency(endpoint: str, duration: float) -> None:
    API_LATENCY.labels(endpoint=endpoint).observe(duration)


def mark_startup() -> None:
    UPTIME_SECONDS.set(0)


def refresh_uptime() -> None:
    UPTIME_SECONDS.set(time.time() - _START_TIME)
