"""
ÉTIQUETAGE HONNÊTE DES MODULES (PROMPT MAÎTRE, Faille 7).

Un n°1 mondial est brutalement honnête sur son système (mentalité n°20 :
zéro ego). Le projet se vendait « institutionnel » alors qu'une partie était
éducative/simulée. Ce registre étiquette CHAQUE module de la plateforme :

  PRODUCTION    : module réel, testé, utilisé dans la décision (sizing/exécution)
  EXPÉRIMENTAL  : module branché mais à surveiller (données limitées, non validé
                  statistiquement) — influence le sizing UNIQUEMENT via des
                  gardes bornées (facteur borné, neutre si indisponible)
  ÉDUCATIF      : module de démonstration — NE DOIT JAMAIS influencer une
                  décision de trading réelle (affichage/analyse uniquement)

Règle absolue (PDF) : « ne jamais laisser un module expérimental influencer
le sizing réel » — chaque module EXPÉRIMENTAL est associé à une GARDE qui le
borne (facteur neutre 1.0 si indisponible, bornes serrées sinon).
"""

# --------------------------------------------------------------------------- #
# REGISTRE CENTRAL — source unique de vérité de l'étiquetage
# --------------------------------------------------------------------------- #
MODULE_STATUS: dict[str, dict] = {
    # --- PRODUCTION (utilisés dans la décision, données réelles) ---
    "multi_source_price": {"status": "PRODUCTION",
                           "detail": "Consensus prix multi-exchange (médiane, divergence -> gel)."},
    "risk_pipeline": {"status": "PRODUCTION",
                      "detail": "Pipeline de risque unifié ordonné (14+ étapes tracées)."},
    "risk_state_machine": {"status": "PRODUCTION",
                           "detail": "NORMAL/CAUTION/HALT avec cool-down et redémarrage progressif."},
    "order_flow": {"status": "PRODUCTION",
                   "detail": "Delta/CVD/OFI/absorption/liquidations sur flux réels (seuils de volume)."},
    "execution_sor": {"status": "PRODUCTION",
                      "detail": "SOR net (prix+frais+slippage), paper execution haute fidélité."},
    "position_lifecycle": {"status": "PRODUCTION",
                           "detail": "SL/TP/trailing/time-stop/breakeven/scaling-out/pyramiding contrôlé."},
    "portfolio_allocator": {"status": "PRODUCTION",
                            "detail": "Top-down : budget risque + réserve cash + capacité + rebalancing."},
    "macro_calendar": {"status": "PRODUCTION",
                       "detail": "Calendrier macro RÉEL daté UTC + phases avant/pendant/après."},
    "sentiment": {"status": "PRODUCTION",
                  "detail": "Sentiment NLP sources réelles pondérées ; UNAVAILABLE si hors ligne."},
    "counterparty_risk": {"status": "PRODUCTION",
                          "detail": "Limite par exchange, signaux d'alerte, hot/cold custody."},
    "reconciliation": {"status": "PRODUCTION",
                       "detail": "Soldes+positions+ordres ; HALT auto en REAL sur écart."},
    "watchdog": {"status": "PRODUCTION",
                 "detail": "11 tâches de fond surveillées + auto-restart."},
    "cost_accounting": {"status": "PRODUCTION",
                        "detail": "Coût réel par trade + portage + PnL net affiché."},
    "attribution": {"status": "PRODUCTION",
                    "detail": "Attribution par facteur/régime/actif/stratégie + métriques de qualité."},
    "scenario_stress": {"status": "PRODUCTION",
                        "detail": "Stress par crises RÉELLES (COVID/2018/FTX) sur le portefeuille."},
    "bias_audit": {"status": "PRODUCTION",
                   "detail": "Audit look-ahead/survivorship/slippage avant tout backtest."},

    # --- EXPÉRIMENTAL (branchés, gardés, à surveiller) ---
    "regime_confidence": {"status": "EXPÉRIMENTAL",
                          "detail": "Confiance du régime HMM (prob x stabilité) — facteur borné 0.5-1.0, neutre si indisponible.",
                          "guard": "facteur borné [0.5, 1.0], jamais de régime sur données insuffisantes"},
    "mlops_challenger": {"status": "EXPÉRIMENTAL",
                         "detail": "Challenger vs champion hors-échantillon — promotion uniquement si battu OOS.",
                         "guard": "aucune promotion sans preuve OOS (Sharpe déflaté)"},
    "causal_discovery": {"status": "EXPÉRIMENTAL",
                         "detail": "Parents causaux (PC-lite) — désactive les signaux si aucun parent.",
                         "guard": "facteur 0.5 max si analyse faite sans parent, neutre sinon"},
    "meta_attribution": {"status": "EXPÉRIMENTAL",
                         "detail": "Poids des raisons (win rate réel) — réduction bornée.",
                         "guard": "facteur borné [0.5, 1.1], aucun impact sans >= 5 échantillons"},

    # --- ÉDUCATIF (ne doivent JAMAIS influencer le sizing) ---
    "rlhf": {"status": "ÉDUCATIF",
             "detail": "RLHF reward model (sans torch = neutre None -> facteur 1.0).",
             "guard": "NEUTRE si indisponible ; borné [0.25, 1.0] sinon ; sans torch jamais actif"},
    "gan_scenarios": {"status": "ÉDUCATIF",
                      "detail": "GAN scénarios extrêmes (fallback bruit si pas de torch) — stress/éducation uniquement.",
                      "guard": "JAMAIS dans le sizing live (affichage/analyse uniquement)"},
    "options_volatility": {"status": "ÉDUCATIF",
                           "detail": "Stratégies d'options (IV Deribit réelle si dispo, sinon UNAVAILABLE).",
                           "guard": "UNAVAILABLE sans IV réelle ; aucune exécution d'options"},
    "llm_narrative": {"status": "ÉDUCATIF",
                      "detail": "Narratif LLM quotidien (OpenRouter optionnel) — informatif uniquement.",
                      "guard": "aucune influence sur les trades"},
}


# Modules ÉDUCATIFS — liste canonique (LOT E / F5). C'est la référence des
# verrous : aucun de ces modules ne doit apparaître dans RISK_PIPELINE_ORDER
# ni être importé par le cœur décisionnel (vérifié mécaniquement par
# tests/test_lot_e_educational.py — spy AST sur tout le repo).
EDUCATIONAL_MODULES: list[str] = [
    name for name, info in MODULE_STATUS.items()
    if info.get("status") == "ÉDUCATIF"
]

# Imports racines des modules éducatifs (tels qu'ils apparaissent dans les
# fichiers du repo) — utilisés par le spy AST de test_lot_e_educational.
EDUCATIONAL_IMPORT_ROOTS: list[str] = [
    "rl.rlhf_reward_model",
    "ai.generative_extreme_scenarios",
    "models.volatility_arbitrage",
    "core.llm_narrative",
]


def get_module_status() -> dict:
    """Registre complet pour la télémétrie et l'UI."""
    return {name: dict(info) for name, info in MODULE_STATUS.items()}


def is_experimental(name: str) -> bool:
    return MODULE_STATUS.get(name, {}).get("status") == "EXPÉRIMENTAL"


def is_educational(name: str) -> bool:
    return MODULE_STATUS.get(name, {}).get("status") == "ÉDUCATIF"


def educational_modules() -> list[str]:
    """Les modules ÉDUCATIFS du registre (source unique des verrous)."""
    return list(EDUCATIONAL_MODULES)


def status_summary() -> dict:
    """Comptage par statut (pour l'UI : X modules PRODUCTION, Y EXPÉRIMENTAL, Z ÉDUCATIF)."""
    counts = {"PRODUCTION": 0, "EXPÉRIMENTAL": 0, "ÉDUCATIF": 0}
    for info in MODULE_STATUS.values():
        counts[info["status"]] = counts.get(info["status"], 0) + 1
    return counts


def assert_registry_coherent() -> list[str]:
    """
    Cohérence du registre (LOT E / F5) : statuts valides, les 4 modules
    ÉDUCATIFS canoniques présents. Retourne la liste des anomalies (vide si
    le registre est sain). Appelé par les tests ET par le safety check.
    """
    errors: list[str] = []
    valid = ("PRODUCTION", "EXPÉRIMENTAL", "ÉDUCATIF")
    for name, info in MODULE_STATUS.items():
        if info.get("status") not in valid:
            errors.append(f"{name}: statut invalide {info.get('status')!r}")
        if not info.get("detail"):
            errors.append(f"{name}: détail manquant")
    for name in ("rlhf", "gan_scenarios", "options_volatility", "llm_narrative"):
        if not is_educational(name):
            errors.append(f"{name}: doit être ÉDUCATIF (verrou F5)")
    return errors
