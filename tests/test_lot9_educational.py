"""
LOT 9 — P2-19 (audit indépendant §2.4) : aucun module ÉDUCATIF dans le pipeline
de sizing.

L'audit : le registre module_honesty était « déclaratif, pas appliqué » — aucun
test ne vérifiait mécaniquement que les modules ÉDUCATIF (rlhf, gan_scenarios,
options_volatility, llm_narrative) n'atteignent jamais apply_risk_pipeline().

VIOLATION RÉELLE trouvée et corrigée : `rlhf_scale=_rlhf_s` alimentait un
facteur multiplicatif du pipeline depuis rlhf_reward_model (module ÉDUCATIF).
Correction : l'appel du pipeline passe rlhf_scale=1.0 CONSTANT (le paramètre
reste dans la signature pour compat ; le bloc de calcul est conservé pour
l'affichage mais n'alimente plus le sizing).

Ce fichier VERROUILLE mécaniquement la règle, exécuté à chaque CI.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# 1. Mapping facteur du pipeline -> module du registre
# --------------------------------------------------------------------------- #

# Facteur du pipeline -> module source dans le registre module_honesty.
# None = pas de module du registre (signal, override humain, confiance...).
FACTOR_MODULE_MAP = {
    "cvar_cap": None,
    "max_asset_cap": "portfolio_allocator",
    "conviction": None,                 # |signal| — pas un module
    "risk_state": "risk_state_machine",
    "news_shock": "sentiment",
    "macro_event": "macro_calendar",
    "macro_tactile": None,              # override humain (boutons Telegram)
    "onchain": None,                    # onchain_tracker (hors registre) — réel
    "correlation": None,                # correlation_risk (hors registre) — réel
    "regime_confidence": "regime_confidence",   # EXPÉRIMENTAL (gardé)
    "capacity": "portfolio_allocator",
    "cash_reserve": "portfolio_allocator",
    "reason_attribution": "meta_attribution",   # EXPÉRIMENTAL (gardé)
    "order_flow": "order_flow",
    "confidence": None,                 # méta-cognition (hors registre) — réel
    "organization": None,               # organization (hors registre) — réel
    "rlhf": "rlhf",                     # ÉDUCATIF -> doit être NEUTRALISÉ
    "vol_targeting": None,              # vol_targeting (hors registre) — réel
    "tradability": "execution_sor",
}

# Facteurs dont la source est un module ÉDUCATIF mais qui sont NEUTRALISÉS
# (le code de main.py passe une constante, preuve ci-dessous).
NEUTRALIZED_EDUCATIONAL_FACTORS = {"rlhf"}


def test_registry_flags_educational_modules():
    """Les 4 modules ÉDUCATIF du registre sont bien étiquetés."""
    from core.module_honesty import is_educational
    for name in ("rlhf", "gan_scenarios", "options_volatility", "llm_narrative"):
        assert is_educational(name) is True, f"{name} doit être ÉDUCATIF"


def test_no_educational_module_feeds_pipeline_unless_neutralized():
    """AUCUN facteur du pipeline ne provient d'un module ÉDUCATIF — sauf ceux
    explicitement neutralisés (rlhf), vérifiés plus bas."""
    from core.module_honesty import is_educational
    from core.risk_pipeline import RISK_PIPELINE_ORDER

    for factor in RISK_PIPELINE_ORDER:
        module = FACTOR_MODULE_MAP.get(factor)
        if module is None:
            continue
        if is_educational(module):
            assert factor in NEUTRALIZED_EDUCATIONAL_FACTORS, \
                f"facteur {factor} alimenté par le module ÉDUCATIF {module} (non neutralisé) !"


def test_pipeline_has_no_unknown_educational_leak():
    """Tout facteur du pipeline est couvert par le mapping (aucun oubli)."""
    from core.risk_pipeline import RISK_PIPELINE_ORDER
    for factor in RISK_PIPELINE_ORDER:
        assert factor in FACTOR_MODULE_MAP, f"facteur {factor} absent du mapping de contrôle"


# --------------------------------------------------------------------------- #
# 2. Spy statique (AST) de l'appel réel dans main.py
# --------------------------------------------------------------------------- #

def _main_call_kwargs() -> dict:
    """Retourne {kwarg: valeur_littérale} de l'appel apply_risk_pipeline
    dans main.py (dernier appel = celui de la boucle live)."""
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "apply_risk_pipeline":
            kwargs = {}
            for kw in node.keywords:
                if isinstance(kw.value, ast.Constant):
                    kwargs[kw.arg] = kw.value.value
                elif isinstance(kw.value, ast.Name):
                    kwargs[kw.arg] = ("VAR", kw.value.id)
                elif isinstance(kw.value, ast.UnaryOp) and isinstance(kw.value.operand, ast.Constant):
                    kwargs[kw.arg] = -kw.value.operand.value
                else:
                    kwargs[kw.arg] = ("EXPR", ast.dump(kw.value)[:40])
            calls.append(kwargs)
    assert calls, "aucun appel à apply_risk_pipeline trouvé dans main.py"
    return calls[-1]  # l'appel de la boucle live (le dernier du fichier)


def test_rlhf_scale_is_constant_one_in_pipeline_call():
    """L'appel réel du pipeline passe rlhf_scale=1.0 CONSTANT — jamais la
    variable _rlhf_s (qui pourrait être alimentée par le module ÉDUCATIF)."""
    kwargs = _main_call_kwargs()
    assert "rlhf_scale" in kwargs
    assert kwargs["rlhf_scale"] == 1.0, \
        f"rlhf_scale={kwargs['rlhf_scale']} — doit être la constante 1.0"


def test_no_educational_variable_feeds_pipeline_call():
    """Aucun kwarg de l'appel du pipeline n'est une variable issue d'un
    module ÉDUCATIF (rlhf_reward_model, generative_engine, etc.)."""
    kwargs = _main_call_kwargs()
    for arg, val in kwargs.items():
        if isinstance(val, tuple) and val[0] == "VAR":
            assert val[1] != "_rlhf_s", \
                f"{arg} reçoit encore _rlhf_s (module ÉDUCATIF) !"


def test_factor_collection_block_avoids_educational_engines():
    """Le bloc de collecte des facteurs (entre le commentaire de collecte et
    l'appel du pipeline) ne référence aucun moteur ÉDUCATIF."""
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    start = src.find("# ---- Collecte de TOUS les facteurs")
    end = src.find("_pipe = apply_risk_pipeline(")
    assert start != -1 and end != -1 and start < end, "bornes du bloc introuvables"
    block = src[start:end]
    for forbidden in ("rlhf_reward_model.predict_reward",
                      "generative_engine.generate_extreme_scenarios",
                      "volatility_arb_engine.evaluate_optimal_options_strategy",
                      "llm_narrative", "daily_market_narrative"):
        assert forbidden not in block, \
            f"moteur ÉDUCATIF utilisé dans le bloc de collecte des facteurs : {forbidden}"


# --------------------------------------------------------------------------- #
# 3. Intégration : le registre est exposé et cohérent dans la télémétrie
# --------------------------------------------------------------------------- #

def test_telemetry_exposes_educational_rule():
    from fastapi.testclient import TestClient

    import main
    with TestClient(main.app) as c:
        r = c.get("/api/telemetry")
        assert r.status_code == 200
        body = r.json()
        honesty = body.get("module_honesty", {})
        assert "registry" in honesty
        reg = honesty["registry"]
        assert reg.get("rlhf", {}).get("status") == "ÉDUCATIF"
        assert reg.get("gan_scenarios", {}).get("status") == "ÉDUCATIF"


def test_apply_risk_pipeline_still_accepts_rlhf_scale_param():
    """Le paramètre rlhf_scale reste dans la signature (compat) mais main
    passe 1.0 — la mécanique du pipeline est inchangée."""
    import inspect

    from core.risk_pipeline import apply_risk_pipeline
    sig = inspect.signature(apply_risk_pipeline)
    assert "rlhf_scale" in sig.parameters


# --------------------------------------------------------------------------- #
# 4. Régression : le warning MLOps (int('') sur get_setting) ne revient pas
# --------------------------------------------------------------------------- #

class _RealisticDB:
    """Fake qui imite la VRAIE signature de DBManager.get_setting :
    (key, user_id=1, decrypt=False) avec int(user_id) — c'est ce qui
    déclenchait 'invalid literal for int() with base 10: \'\' quand un
    appel passait un 'défaut' non numérique en 2e position."""

    def __init__(self):
        self.s = {}

    def get_setting(self, key, user_id=1, decrypt=False):
        return self.s.get(key, "")

    def save_setting(self, key, value, user_id=1, encrypt=False):
        self.s[key] = value

    def add_audit_log(self, *a, **k):
        pass


def test_deploy_mlops_no_int_error_with_realistic_db(caplog):
    """P2-limite : deploy_challenger_if_beats_champion ne doit plus lever
    'invalid literal for int() with base 10' quand la DB n'a pas de valeur
    (le 2e argument de get_setting était passé comme user_id)."""
    import logging

    import pandas as pd

    from models.mlops_pipeline import MLOpsAutoTrainer
    from models.price_predictor import LSTMLikePredictor
    from models.regime_detector import MarketRegimeDetector

    caplog.set_level(logging.WARNING, logger="MLOpsPipeline")
    trainer = MLOpsAutoTrainer(MarketRegimeDetector(), LSTMLikePredictor(),
                               _RealisticDB())
    df = pd.DataFrame({"close": [1.0] * 60, "high": [1.0] * 60,
                       "low": [1.0] * 60, "open": [1.0] * 60,
                       "volume": [1.0] * 60})
    # ne doit PAS logger le warning (ni lever)
    res = trainer.deploy_challenger_if_beats_champion(df, "lstm", 5.78)
    assert res in (True, False)
    assert "Challenger/champion comparison failed" not in caplog.text


def test_mlops_does_not_pass_default_to_get_setting():
    """Les appels get_setting de deploy ne passent plus de 'défaut' en 2e
    position (le bug int(''))."""
    src = (ROOT / "models" / "mlops_pipeline.py").read_text(encoding="utf-8")
    assert 'get_setting("mlops_n_trials")' in src
    assert 'get_setting(champion_key)' in src
    assert 'get_setting(champion_key, ""' not in src
