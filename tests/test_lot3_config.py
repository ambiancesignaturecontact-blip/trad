"""
LOT 3 — P1-8 (audit indépendant §3) : branchement de core/config.py.

L'audit : « settings.get(...) n'est utilisé que dans 3 fichiers sur 171.
L'immense majorité du code garde ses constantes en dur. »

Ce test vérifie MÉCANIQUEMENT que les constantes du cœur qui touche l'argent
(portefeuille, contrepartie, coûts, order flow, vol targeting, CVaR,
drawdowns, multi-sources) sont branchées sur settings avec des défauts
STRICTEMENT identiques aux valeurs historiques — zéro changement de
comportement, mais une source unique de vérité désormais surchargeable via
config.yaml.
"""
from pathlib import Path

import pytest
import yaml

from core.config import settings, load_settings

# --------------------------------------------------------------------------- #
# 1. Le config.yaml du repo est bien chargé et cohérent
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent.parent


def test_config_yaml_parses_and_sections_present():
    """config.yaml existe, se parse, et contient les nouvelles sections."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for section in ("portfolio", "counterparty", "costs", "orderflow",
                    "vol_targeting", "execution", "autopilot"):
        assert section in cfg, f"section {section} manquante dans config.yaml"


def test_settings_loaded_from_yaml():
    """Les valeurs du config.yaml réel sont chargées dans settings."""
    assert settings.get_float("portfolio", "cash_reserve_pct", -1.0) == 0.15
    assert settings.get_float("portfolio", "redundant_corr", -1.0) == 0.85
    assert settings.get_float("counterparty", "max_capital_per_exchange_pct", -1.0) == 0.40
    assert settings.get_float("costs", "default_fee_rate", -1.0) == 0.001
    assert settings.get_int("orderflow", "cascade_min_events", -1) == 3
    assert settings.get_int("autopilot", "min_paper_validation_days", -1) == 28


# --------------------------------------------------------------------------- #
# 2. Les modules cœur lisent settings (défauts identiques -> non-régression)
# --------------------------------------------------------------------------- #

def test_portfolio_allocator_branched():
    import core.portfolio_allocator as m
    assert m.CASH_RESERVE_PCT == settings.get_float("portfolio", "cash_reserve_pct", 0.15)
    assert m.TARGET_VOL_ANNUAL == settings.get_float("portfolio", "target_vol_annual", 0.10)
    assert m.REBALANCE_HOURS == settings.get_float("portfolio", "rebalance_hours", 24.0)
    assert m.MAX_PARTICIPATION_PCT == settings.get_float("portfolio", "max_participation_pct", 0.01)
    assert m.REDUNDANT_CORR == settings.get_float("portfolio", "redundant_corr", 0.85)
    # valeurs historiques préservées
    assert m.CASH_RESERVE_PCT == 0.15 and m.REDUNDANT_CORR == 0.85


def test_counterparty_branched():
    import core.counterparty_risk as m
    assert m.MAX_CAPITAL_PER_EXCHANGE_PCT == settings.get_float("counterparty", "max_capital_per_exchange_pct", 0.40)
    assert m.ALERT_SPREAD_BPS == settings.get_float("counterparty", "alert_spread_bps", 25.0)
    assert m.ALERT_VOLUME_DROP_PCT == settings.get_float("counterparty", "alert_volume_drop_pct", 0.30)


def test_cost_accounting_branched():
    import core.cost_accounting as m
    assert m.DEFAULT_FEE_RATE == settings.get_float("costs", "default_fee_rate", 0.001)
    assert m.DEFAULT_SLIPPAGE_BPS == settings.get_float("costs", "default_slippage_bps", 5.0)


def test_order_flow_branched():
    import market_data.order_flow as m
    assert m.DELTA_WINDOW == settings.get_float("orderflow", "delta_window_seconds", 60.0)
    assert m.CASCADE_WINDOW == settings.get_float("orderflow", "cascade_window_seconds", 30.0)
    assert m.ABSORPTION_WINDOW == settings.get_float("orderflow", "absorption_window_seconds", 30.0)
    assert m.TOXIC_DELTA_RATIO == settings.get_float("orderflow", "toxic_delta_ratio", 0.35)
    assert m.MIN_TOXIC_VOLUME_USD == settings.get_float("orderflow", "min_toxic_volume_usd", 100000.0)
    assert m.CASCADE_MIN_EVENTS == settings.get_int("orderflow", "cascade_min_events", 3)
    assert m.CASCADE_MIN_NOTIONAL == settings.get_float("orderflow", "cascade_min_notional_usd", 200000.0)
    assert m.ABSORPTION_VOLUME_USD == settings.get_float("orderflow", "absorption_volume_usd", 300000.0)


def test_vol_targeting_branched():
    import core.volatility_targeting as m
    # les défauts de la fonction sont issus de settings
    import inspect
    sig = inspect.signature(m.volatility_scale_factor)
    assert sig.parameters["target_tick_vol"].default == settings.get_float("vol_targeting", "target_tick_vol", 0.0004)
    assert sig.parameters["realized_window"].default == settings.get_int("vol_targeting", "realized_window", 40)
    assert sig.parameters["min_scale"].default == settings.get_float("vol_targeting", "min_scale", 0.25)
    assert sig.parameters["max_scale"].default == settings.get_float("vol_targeting", "max_scale", 2.0)
    # comportement préservé
    assert m.volatility_scale_factor([]) == 1.0


def test_cvar_optimizer_branched():
    import core.cvar_optimizer as m
    class FakeCov:
        pass
    opt = m.CVaRPortfolioOptimizer(FakeCov())
    assert opt.cvar_limit_pct == settings.get_float("execution", "cvar_limit_pct", 0.025)
    # le paramètre explicite reste prioritaire
    opt2 = m.CVaRPortfolioOptimizer(FakeCov(), cvar_limit_pct=0.05)
    assert opt2.cvar_limit_pct == 0.05


def test_risk_manager_drawdowns_branched():
    from risk.risk_manager import RiskManager
    rm = RiskManager()
    rm.set_initial_capital(50.0)
    assert rm.params["max_daily_drawdown_pct"] == settings.get_float("risk", "daily_drawdown_micro", 0.18)
    assert rm.params["max_total_drawdown_pct"] == settings.get_float("risk", "max_total_drawdown_micro", 0.35)
    rm.set_initial_capital(50000.0)
    assert rm.params["max_daily_drawdown_pct"] == settings.get_float("risk", "daily_drawdown_small", 0.10)
    assert rm.params["max_total_drawdown_pct"] == settings.get_float("risk", "max_total_drawdown_small", 0.20)
    rm.set_initial_capital(1_000_000.0)
    assert rm.params["max_daily_drawdown_pct"] == settings.get_float("risk", "daily_drawdown_normal", 0.025)
    assert rm.params["max_total_drawdown_pct"] == settings.get_float("risk", "max_total_drawdown_normal", 0.08)
    # Kelly par défaut branché
    assert rm.params["fractional_kelly_multiplier"] == settings.get_float("risk", "kelly_multiplier_default", 0.15)


def test_multi_source_branched():
    import market_data.multi_source as m
    assert m.DEFAULT_THRESHOLD_PCT == settings.get_float("data", "divergence_threshold_pct", 1.00)


def test_paper_execution_branched():
    import core.paper_execution as m
    assert m.MIN_NOTIONAL_USD == settings.get_float("trading", "min_notional_usd_normal", 10.0)


# --------------------------------------------------------------------------- #
# 3. L'override via config.yaml fonctionne réellement
# --------------------------------------------------------------------------- #

def test_load_settings_with_custom_config(tmp_path, monkeypatch):
    """Un CONFIG_PATH personnalisé doit surcharger les défauts (la preuve que
    la config n'est pas décorative)."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "portfolio:\n  cash_reserve_pct: 0.25\n  redundant_corr: 0.95\n"
        "orderflow:\n  toxic_delta_ratio: 0.50\n"
        "risk:\n  daily_drawdown_micro: 0.22\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONFIG_PATH", str(cfg))
    s = load_settings()
    assert s.get_float("portfolio", "cash_reserve_pct", 0.0) == 0.25
    assert s.get_float("portfolio", "redundant_corr", 0.0) == 0.95
    assert s.get_float("orderflow", "toxic_delta_ratio", 0.0) == 0.50
    assert s.get_float("risk", "daily_drawdown_micro", 0.0) == 0.22
    # une clé absente du YAML reste au défaut
    assert s.get_float("portfolio", "target_vol_annual", 0.10) == 0.10


def test_load_settings_ignores_bad_yaml(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("::::: pas du yaml [", encoding="utf-8")
    monkeypatch.setenv("CONFIG_PATH", str(cfg))
    s = load_settings()  # ne doit pas lever
    assert s.get_float("portfolio", "cash_reserve_pct", 0.15) == 0.15


# --------------------------------------------------------------------------- #
# 4. Périmètre du branchement (le compteur de l'audit)
# --------------------------------------------------------------------------- #

def test_config_branching_coverage():
    """Le cœur qui touche l'argent est branché : le nombre de fichiers
    importants utilisant core.config passe de 3 à >= 10."""
    branched = [
        # core/config.py est la SOURCE, pas un consommateur
        "core/risk_pipeline.py",
        "core/paper_execution.py",
        "core/portfolio_allocator.py",
        "core/counterparty_risk.py",
        "core/cost_accounting.py",
        "core/volatility_targeting.py",
        "core/cvar_optimizer.py",
        "market_data/order_flow.py",
        "market_data/multi_source.py",
        "risk/risk_manager.py",
        "main.py",
    ]
    missing = []
    for rel in branched:
        p = ROOT / rel
        src = p.read_text(encoding="utf-8")
        if "from core.config import" not in src:
            missing.append(rel)
    assert missing == [], f"modules non branchés sur core.config : {missing}"
