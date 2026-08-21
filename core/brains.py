"""
ARCHITECTURE MULTI-CERVEaux (LOT 8 du mandat, §17).

Documente et vérifie la séparation en 8 cerveaux spécialisés, découplés et
observables. Chaque cerveau regroupe des MODULES existants (rien n'est
réécrit : le registre est la carte de l'architecture réelle) :

  MARKET      : marché, régime, macro, volatilité, données multi-sources
  ALPHA       : stratégies, prédictions, experts (bandit, MoE, regret)
  CONVICTION  : calibration, meta-labeling, incertitude, TRADE/WAIT
  DECISION    : trade/no-trade, allocation, contexte global, journal
  RISK        : limites, CVaR, drawdown, concentration, kill switch, adversarial
  EXECUTION   : SOR, timing, venue, coût, execution intel
  LEARNING    : post-trade, edge decay, drift, online learning
  ANALYST     : explication, rapports, observabilité

Exigences (mandat) : chaque cerveau doit être testable, observable,
documenté, découplé, remplaçable. Le registre + les tests de
test_lot8_architecture.py vérifient ces propriétés.
"""
import logging

logger = logging.getLogger("InstitutionalTradingBot")

BRAINS: dict[str, dict] = {
    "MARKET": {
        "role": "marché, régime, macro, volatilité, données",
        "modules": [
            "market_data.multi_source", "market_data.order_flow",
            "market_data.historical_fetch", "market_data.order_book",
            "models.regime_detector", "models.macro_calendar",
            "core.volatility_targeting", "core.factor_model",
            "core.drift_psi",              # drift distribution (input)
        ],
    },
    "ALPHA": {
        "role": "stratégies, prédictions, experts",
        "modules": [
            "strategies.engine", "strategies.regime_switching",
            "core.hierarchical_allocator", "core.mixture_experts",
            "core.execution_agent",        # bandit de style
            "models.price_predictor", "models.mlops_pipeline",
        ],
    },
    "CONVICTION": {
        "role": "calibration, meta-labeling, incertitude",
        "modules": [
            "core.conviction_engine", "core.meta_cognition",
            "core.risk_pipeline",          # calibrated_conviction, kelly
        ],
    },
    "DECISION": {
        "role": "trade/no-trade, allocation, contexte, journal",
        "modules": [
            "core.decision_journal", "core.ops_alerts",
            "core.portfolio_allocator", "core.dynamic_capital_allocator",
        ],
    },
    "RISK": {
        "role": "limites, CVaR, drawdown, concentration, kill switch",
        "modules": [
            "risk.risk_manager", "core.risk_pipeline",
            "core.regime_autonomy", "core.counterparty_risk",
            "core.cvar_optimizer", "core.cost_accounting",
            "core.adversarial_engine",     # robustesse sous stress
            "core.risk_committee", "core.robustness",
        ],
    },
    "EXECUTION": {
        "role": "SOR, timing, venue, coût",
        "modules": [
            "core.execution_router", "core.multi_exchange_sor",
            "core.execution_intel", "core.execution_simulator",
            "core.almgren_chriss_advanced", "core.paper_execution",
        ],
    },
    "LEARNING": {
        "role": "post-trade, edge decay, drift, online learning",
        "modules": [
            "core.edge_decay", "core.drift_psi",   # PSI/CUSUM fusion
            "models.mlops_pipeline", "models.lopez_de_prado",
            "core.attribution",
        ],
    },
    "ANALYST": {
        "role": "explication, rapports, observabilité",
        "modules": [
            "core.observability", "core.decision_explain",
            "core.reporting", "core.self_assessment", "core.llm_narrative",
            "core.module_honesty", "telemetry", "core.state_helpers",
        ],
    },
}

# Tous les modules du registre (vérification : importables)
ALL_MODULES: list[str] = sorted({m for b in BRAINS.values() for m in b["modules"]})


def brain_of(module: str) -> str | None:
    """Cerveau d'un module (None si non répertorié)."""
    for name, info in BRAINS.items():
        if module in info["modules"]:
            return name
    return None


def verify_importable() -> list[str]:
    """
    Vérifie que chaque module du registre est IMPORTABLE. Retourne la liste
    des modules qui échouent (vide = architecture saine).
    """
    import importlib
    failures = []
    for mod in ALL_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failures.append(f"{mod}: {type(e).__name__}")
    return failures


def verify_decoupled() -> list[str]:
    """
    Vérifie que les modules du registre n'utilisent PAS `from main import *`
    (découplage). Retourne les violations (vide = propre).
    """
    import ast
    from pathlib import Path
    violations = []
    for mod in ALL_MODULES:
        path = Path(*mod.split("."))
        py = path.with_suffix(".py")
        if not py.exists():
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "main" \
                    and any(a.name == "*" for a in node.names):
                violations.append(mod)
    return violations


def report() -> dict:
    """Registre complet + vérifications pour télémétrie / API."""
    importable_failures = verify_importable()
    decoupling_violations = verify_decoupled()
    return {
        "brains": {name: {"role": info["role"], "modules": info["modules"]}
                   for name, info in BRAINS.items()},
        "n_brains": len(BRAINS),
        "n_modules": len(ALL_MODULES),
        "checks": {
            "all_importable": not importable_failures,
            "importable_failures": importable_failures,
            "no_star_import_from_main": not decoupling_violations,
            "star_import_violations": decoupling_violations,
        },
        "note": "Carte de l'architecture réelle : chaque cerveau regroupe les "
                "modules existants (aucun ré-écrit) ; les vérifications sont "
                "exécutées par tests/test_lot8_architecture.py.",
        "ts": __import__("time").time(),
    }
