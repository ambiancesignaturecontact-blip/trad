"""
Centralized runtime configuration (LOT 68 - audit B1-2/B1-3).

Loads config.yaml (institutional parameters) + .env (secrets) once at startup
so the trading loop, risk engine and strategies read tunable values instead of
hard-coded constants. Every key falls back to a safe default if missing.
"""
import os
from dataclasses import dataclass, field
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
    _dotenv_loaded = load_dotenv()  # loads .env from CWD if present
except Exception:
    _dotenv_loaded = False


_DEFAULTS: dict[str, Any] = {
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
        # P1-14 (audit §2.7) : plafond d'exposition TOTALE du portefeuille —
        # nettement supérieur au plafond PAR ACTIF (max_per_asset_pct=0.25)
        # pour permettre 3-4 positions diversifiées simultanées (75 % total
        # / 25 % par actif, exactement la fourchette 70-80 % recommandée).
        "max_exposure_normal": 0.75,
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
        # P0-4 (audit §2.1) : plancher anti-empilement de la réduction
        # cumulative des overlays (0.8^15 ≈ 3,5 % sans plancher).
        "final_scale_floor": 0.15,
        # P1-8 (audit §3) : drawdowns par taille de compte (RiskManager)
        "daily_drawdown_small": 0.10,
        "max_total_drawdown_micro": 0.35,
        "max_total_drawdown_small": 0.20,
        "max_total_drawdown_normal": 0.08,
        "kelly_multiplier_default": 0.15,
    },
    "data": {
        "use_real_data_only": True,
        "yahoo_fallback": True,
        "bybit_fallback": True,
        "yahoo_cache_ttl_seconds": 20.0,
        # P1-8 (audit §3) : seuil de divergence multi-sources (futures vs spot)
        "divergence_threshold_pct": 1.00,
    },
    # P1-8 (audit §3) : branchement de core/config.py — les constantes en dur
    # des modules cœur passent par settings (défauts = valeurs historiques).
    "portfolio": {
        "cash_reserve_pct": 0.15,
        "target_vol_annual": 0.10,
        "rebalance_hours": 24.0,
        "max_participation_pct": 0.01,   # 1 % du volume 24h
        "redundant_corr": 0.85,
    },
    "counterparty": {
        "max_capital_per_exchange_pct": 0.40,
        "alert_spread_bps": 25.0,
        "alert_volume_drop_pct": 0.30,
    },
    "costs": {
        "default_fee_rate": 0.001,       # 0.1 % taker
        "default_slippage_bps": 5.0,     # 5 bps moyen
    },
    "orderflow": {
        "delta_window_seconds": 60.0,
        "cascade_window_seconds": 30.0,
        "absorption_window_seconds": 30.0,
        "toxic_delta_ratio": 0.35,
        "min_toxic_volume_usd": 100000.0,
        "cascade_min_events": 3,
        "cascade_min_notional_usd": 200000.0,
        "absorption_volume_usd": 300000.0,
    },
    "vol_targeting": {
        "target_tick_vol": 0.0004,
        "realized_window": 40,
        "min_scale": 0.25,
        "max_scale": 2.0,
    },
    "execution": {
        "cvar_limit_pct": 0.025,
    },
    "strategies": {
        "all_enabled": True,
        "enable_stat_arb": True,
        "enable_inter_exchange_arb": True,
        "enable_scalping": True,
        # P1-12 (audit §2.6) : oubli du bandit Thompson (non-stationnarité)
        "bandit_decay": 0.98,
        # P1-12 (audit §2.2) : durée de vie d'un tirage Thompson (cycle de décision)
        "bandit_sample_refresh_seconds": 60.0,
        # P1-11 (audit §2.5) : plancher d'échantillon + borne d'ajustement PnL
        "pnl_min_samples_full": 20,
        "pnl_max_adjustment": 0.20,
    },
    "autopilot": {
        # P0-6 (audit §5-P0-6) : l'audit exige 4-8 semaines de paper-trading
        # daté et CONTINU avant REAL. 28 jours = 4 semaines.
        "min_paper_validation_days": 28,
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
    raw: dict[str, Any] = field(default_factory=dict)

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


def _deep_merge(base: dict, override: dict) -> dict:
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
            with open(cfg_path) as f:
                user_cfg = yaml.safe_load(f) or {}
            data = _deep_merge(data, user_cfg)
    except Exception as e:  # never block startup on a bad config file
        import logging
        logging.getLogger("Config").warning(f"config.yaml ignored ({e})")
    return Settings(raw=data)


settings = load_settings()
