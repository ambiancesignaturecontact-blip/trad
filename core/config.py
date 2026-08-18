"""
Centralized runtime configuration (LOT 68 - audit B1-2/B1-3).

Loads config.yaml (institutional parameters) + .env (secrets) once at startup
so the trading loop, risk engine and strategies read tunable values instead of
hard-coded constants. Every key falls back to a safe default if missing.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any

import yaml

try:
    from dotenv import load_dotenv
    _dotenv_loaded = load_dotenv()  # loads .env from CWD if present
except Exception:
    _dotenv_loaded = False


_DEFAULTS: Dict[str, Any] = {
    "trading": {
        "min_notional_usd_micro": 3.0,
        "min_notional_usd_small": 5.0,
        "min_notional_usd_normal": 10.0,
        "signal_threshold": 0.08,
        "loop_sleep_seconds": 2.5,
        "order_cooldown_demo_seconds": 10.0,
        "order_cooldown_real_seconds": 60.0,
        "max_assets": 20,
    },
    "risk": {
        "max_exposure_micro": 0.85,
        "max_exposure_normal": 0.25,
        "daily_drawdown_micro": 0.18,
        "daily_drawdown_normal": 0.025,
        "max_per_asset_pct": 0.25,
        # LOT 2 (PDF Pilier F/G) : source unique de vérité du RR + machine à états
        "reward_risk_ratio": 1.8,
        "min_reward_risk": 1.5,
        "min_reward_risk_high_vol": 2.0,
        "high_vol_return_std": 0.02,
        "round_trip_cost_pct": 0.002,
        "kelly_fraction": 0.15,
        "stop_loss_pct": 0.03,
        "atr_mult_sl": 2.0,
        "halt_cooldown_minutes": 15.0,
    },
    "data": {
        "use_real_data_only": True,
        "yahoo_fallback": True,
        "bybit_fallback": True,
        "yahoo_cache_ttl_seconds": 20.0,
    },
    "strategies": {
        "all_enabled": True,
        "enable_stat_arb": True,
        "enable_inter_exchange_arb": True,
        "enable_scalping": True,
    },
    "autopilot": {
        "min_paper_validation_days": 7,   # gates before REAL (audit D1)
        "paper_validation_required": True,
    },
    "alerts": {
        "daily_digest_enabled": True,
        "daily_digest_hour_utc": 18,
    },
    "logging": {
        "level": "INFO",
        "structured": False,
    },
}


@dataclass
class Settings:
    raw: Dict[str, Any] = field(default_factory=dict)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        sec = self.raw.get(section, {})
        if isinstance(sec, dict) and key in sec:
            return sec[key]
        return default

    def get_float(self, section: str, key: str, default: float) -> float:
        try:
            return float(self.get(section, key, default))
        except (TypeError, ValueError):
            return default

    def get_bool(self, section: str, key: str, default: bool) -> bool:
        val = self.get(section, key, default)
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in ("1", "true", "yes", "on")

    def get_int(self, section: str, key: str, default: int) -> int:
        try:
            return int(self.get(section, key, default))
        except (TypeError, ValueError):
            return default


def _deep_merge(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings() -> Settings:
    """Loads config.yaml (if present) merged over safe defaults."""
    data = dict(_DEFAULTS)
    cfg_path = os.getenv("CONFIG_PATH", os.path.join(os.getcwd(), "config.yaml"))
    try:
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                user_cfg = yaml.safe_load(f) or {}
            data = _deep_merge(data, user_cfg)
    except Exception as e:  # never block startup on a bad config file
        import logging
        logging.getLogger("Config").warning(f"config.yaml ignored ({e})")
    return Settings(raw=data)


settings = load_settings()
