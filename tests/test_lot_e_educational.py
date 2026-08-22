"""
LOT E (F5 — modules éducatifs) : verrouillage RENFORCÉ.

LOT 9 (P2-19) avait neutralisé `rlhf_scale=1.0` dans l'appel du pipeline.
LOT E étend le verrou à TOUT le dépôt :

  1. REGISTRE : `core/module_honesty.py` est la source unique — 4 modules
     ÉDUCATIFS canoniques (rlhf, gan_scenarios, options_volatility,
     llm_narrative), cohérence vérifiée (assert_registry_coherent).
  2. PIPELINE : aucun facteur de RISK_PIPELINE_ORDER ne provient d'un
     module ÉDUCATIF (le mapping LOT 9 + la liste canonique).
  3. SPY AST REPO ENTIER : les modules éducatifs ne sont importés QUE par
     les fichiers autorisés (instances/affichage/télémétrie) et JAMAIS par
     le cœur décisionnel (pipeline, sizing, exécution, allocation).
  4. DATA-FLOW : les sorties des moteurs éducatifs (gan_stress_ratio,
     options_strategy, narratives) n'atteignent jamais le bloc du pipeline
     dans main.py.
  5. API/UI : le registre est exposé (/api/v1/honesty) et l'UI l'affiche.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

EDUCATIONAL = ("rlhf", "gan_scenarios", "options_volatility", "llm_narrative")
EDUCATIONAL_IMPORT_ROOTS = (
    "rl.rlhf_reward_model",
    "ai.generative_extreme_scenarios",
    "models.volatility_arbitrage",
    "core.llm_narrative",
)

# Fichiers AUTORISÉS à importer les modules éducatifs (instances, affichage,
# télémétrie, API narrative) — hors tests.
ALLOWED_FILES = {
    "main.py",
    "core/autonomous_ai.py",          # GAN stress -> jauge Prometheus + STATE (jamais pipeline)
    "core/module_honesty.py",         # registre lui-même
    "api/routes.py",                  # assistant LLM / narrative
    "schedulers.py",                  # narrative quotidienne
    "telemetry.py",
    "core/llm_narrative.py",          # le module éducatif lui-même
    "ai/generative_extreme_scenarios.py",
    "rl/rlhf_reward_model.py",
    "models/volatility_arbitrage.py",
    "strategies/volatility_arbitrage.py",
    "models/macro_calendar.py",       # shim de dépréciation (cf. market_data/macro_calendar)
    "market_data/macro_calendar.py",
}

# Fichiers du CŒUR DÉCISIONNEL : AUCUN import éducatif n'y est toléré.
# PHASE 3 Cycle 2 (§23 maintenance active) : core/kelly_sizing.py retiré —
# module jamais importé (DynamicKellySizer sans appelant), purgé du dépôt.
DECISION_CORE = {
    "core/risk_pipeline.py",
    "core/portfolio_allocator.py",
    "core/paper_execution.py",
    "core/execution_agent.py",
    "core/execution_router.py",
    "core/execution_simulator.py",
    "core/cvar_optimizer.py",
    "core/volatility_targeting.py",
    "core/cost_accounting.py",
    "core/counterparty_risk.py",
    "core/regime_autonomy.py",
    "core/drift_psi.py",
    "core/position_manager.py",
    "core/risk_committee.py",
    "core/meta_cognition.py",
    "risk/risk_manager.py",
    "strategies/engine.py",
    "strategies/regime_switching.py",
    "strategies/momentum.py",
    "strategies/mean_reversion.py",
    "strategies/scalping.py",
    "strategies/grid.py",
    "strategies/swing.py",
}


def _imports_from(tree: ast.AST) -> set:
    """Modules importés (racine) dans un AST."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module.split(".")[0])
    return out


# --------------------------------------------------------------------------- #
# 1. Registre
# --------------------------------------------------------------------------- #
class TestRegistry:
    def test_educational_modules_canonical(self):
        from core.module_honesty import educational_modules
        assert set(educational_modules()) == set(EDUCATIONAL)

    def test_registry_coherent(self):
        from core.module_honesty import MODULE_STATUS, assert_registry_coherent
        errors = assert_registry_coherent()
        assert not errors, f"registre incohérent : {errors}"
        # 16 PRODUCTION + 4 EXPÉRIMENTAL + 4 ÉDUCATIF
        assert len(MODULE_STATUS) == 24
        from core.module_honesty import status_summary
        s = status_summary()
        assert s["ÉDUCATIF"] == 4

    def test_no_educational_in_pipeline_order(self):
        """Aucun facteur du pipeline ne provient d'un module ÉDUCATIF — SAUF
        rlhf, qui est NEUTRALISÉ (constante 1.0 vérifiée par LOT 9)."""
        from core.module_honesty import educational_modules
        from core.risk_pipeline import RISK_PIPELINE_ORDER
        neutralized = {"rlhf"}   # seul facteur éducatif toléré, neutralisé
        for factor in RISK_PIPELINE_ORDER:
            if factor in educational_modules():
                assert factor in neutralized, \
                    f"facteur {factor} = module ÉDUCATIF non neutralisé dans le pipeline !"


# --------------------------------------------------------------------------- #
# 2. Spy AST repo entier : imports éducatifs uniquement dans la liste blanche
# --------------------------------------------------------------------------- #
class TestRepoWideSpy:
    @pytest.mark.parametrize("root", EDUCATIONAL_IMPORT_ROOTS)
    def test_educational_imports_restricted_to_allowlist(self, root):
        """Le module racine {root} n'est importé que par les fichiers autorisés."""
        offenders = []
        for py in ROOT.rglob("*.py"):
            if "__pycache__" in str(py) or str(py).startswith(str(ROOT / "tests")):
                continue
            rel = py.relative_to(ROOT).as_posix()
            if rel in ALLOWED_FILES:
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            imports = _imports_from(tree)
            top = root.split(".")[0]
            if top in imports or root in {f"{top}.{p}" for p in ("rlhf_reward_model",
                                                                  "generative_extreme_scenarios",
                                                                  "volatility_arbitrage",
                                                                  "llm_narrative")}:
                # vérification précise : le module ciblé (pas juste un homonyme)
                precise = False
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module and \
                            node.module == root:
                        precise = True
                    elif isinstance(node, ast.ImportFrom) and node.module and \
                            node.module.startswith(root):
                        precise = True
                if precise:
                    offenders.append(rel)
        assert not offenders, \
            f"{root} importé par des fichiers non autorisés : {offenders}"

    def test_decision_core_never_imports_educational(self):
        """Le cœur décisionnel n'importe AUCUN module éducatif (aucun filet)."""
        for rel in DECISION_CORE:
            py = ROOT / rel
            if not py.exists():
                continue
            tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for root in EDUCATIONAL_IMPORT_ROOTS:
                        assert not node.module.startswith(root), \
                            f"{rel} importe {node.module} (module ÉDUCATIF) !"


# --------------------------------------------------------------------------- #
# 3. Data-flow : sorties éducatives jamais dans le bloc du pipeline
# --------------------------------------------------------------------------- #
class TestDataFlow:
    def test_pipeline_block_free_of_educational_outputs(self):
        """Le bloc de collecte des facteurs ne référence aucune sortie éducative."""
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        start = src.find("# ---- Collecte de TOUS les facteurs")
        end = src.find("_pipe = apply_risk_pipeline(")
        assert start != -1 and end != -1 and start < end, "bornes du bloc introuvables"
        block = src[start:end]
        for forbidden in ("gan_stress_ratio", "options_strategy", "real_iv",
                          "generative_engine", "volatility_arb_engine",
                          "rlhf_reward_model", "llm_narrative"):
            assert forbidden not in block, \
                f"sortie éducative {forbidden} référencée dans le bloc du pipeline !"

    def test_gan_stress_is_telemetry_only(self):
        """gan_stress_ratio n'existe QUE dans autonomous_ai (STATE) — jamais lu ailleurs."""
        import subprocess
        res = subprocess.run(
            ["grep", "-rn", "gan_stress_ratio", "--include=*.py", "."],
            capture_output=True, text=True, cwd=str(ROOT))
        hits = [line for line in res.stdout.splitlines()
                if "test" not in line and "autonomous_ai.py" not in line
                and "STATE[" not in line and "gan_stress_ratio" in line]
        # le seul usage hors autonomous_ai/tests doit être inexistant
        for line in hits:
            assert "STATE[" in line, f"gan_stress_ratio utilisé hors télémétrie : {line}"

    def test_options_strategy_is_display_only(self):
        """options_strategy (module ÉDUCATIF options) n'est lu que pour l'affichage."""
        import subprocess
        res = subprocess.run(
            ["grep", "-rn", "options_strategy", "--include=*.py", "."],
            capture_output=True, text=True, cwd=str(ROOT))
        for line in res.stdout.splitlines():
            if "test" in line or "STATE[" in line or "templates" in line or "telemetry" in line:
                continue
            # définition par défaut dans le dict STATE (ligne "options_strategy": {...})
            # ou définition de la méthode dans le module éducatif lui-même
            if '"options_strategy":' in line or "volatility_arbitrage.py" in line:
                continue
            assert "STATE[" in line, f"options_strategy hors télémétrie : {line}"


# --------------------------------------------------------------------------- #
# 4. API / UI
# --------------------------------------------------------------------------- #
class TestExposure:
    def test_api_v1_honesty_exposes_registry(self):
        from fastapi.testclient import TestClient

        import main
        with TestClient(main.app) as c:
            r = c.get("/api/v1/honesty")
            assert r.status_code == 200
            body = r.json()
            assert "modules" in body and "summary" in body
            assert body["summary"]["ÉDUCATIF"] == 4
            for name in EDUCATIONAL:
                assert name in body["modules"]
                assert body["modules"][name]["status"] == "ÉDUCATIF"

    def test_dashboard_and_miniapp_display_lock(self):
        """L'UI affiche le verrou des modules éducatifs."""
        dash = (ROOT / "templates/dashboard.html").read_text(encoding="utf-8")
        assert "HONNÊTETÉ DES MODULES" in dash
        assert "ÉDUCATIF" in dash
        mini = (ROOT / "templates/telegram_mini_app.html").read_text(encoding="utf-8")
        assert "Honnêteté modules" in mini
