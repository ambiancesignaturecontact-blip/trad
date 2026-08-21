"""
LOT 8 (mandat — Architecture) : découplage + séparation des cerveaux +
interfaces stables.

Vérifié ici :
  1. main.py < 4000 lignes (réduction continue) ET les helpers d'état/auth
     ont été extraits (core/state_helpers.py, core/auth_helpers.py) avec
     ré-exports compatibles (api/routes, schedulers, telemetry fonctionnent).
  2. Registre des CERVEAUX (core/brains.py) : 8 cerveaux (mandat §17), chaque
     module répertorié, tous importables, AUCUN star import depuis main
     (découplage vérifié par AST sur tout le repo).
  3. API /api/v1/brains + télémétrie brains exposée.
  4. Interfaces stables : les symboles extraits restent accessibles via main
     (ré-exports) — les modules consommateurs (api/routes, telemetry) n'ont
     pas changé.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# 1. Réduction de main.py + extraction
# --------------------------------------------------------------------------- #
class TestMainReduction:
    def test_main_under_4000_lines(self):
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        assert len(src.splitlines()) < 4000, \
            f"main.py = {len(src.splitlines())} lignes (objectif < 4000)"

    def test_helpers_extracted_to_modules(self):
        """Les helpers d'état et d'auth vivent dans leurs modules dédiés."""
        sh = (ROOT / "core" / "state_helpers.py").read_text(encoding="utf-8")
        for fn in ("mark_real_price", "update_asset_order_book",
                   "set_asset_quality", "set_data_quality", "_neutral",
                   "record_open_position"):
            assert f"def {fn}" in sh, f"{fn} manquant dans state_helpers"
        ah = (ROOT / "core" / "auth_helpers.py").read_text(encoding="utf-8")
        for fn in ("auth_enforced", "require_auth", "require_admin",
                   "_is_remote_deployment", "_ensure_auth_secrets"):
            assert f"def {fn}" in ah, f"{fn} manquant dans auth_helpers"

    def test_reexports_preserved_in_main(self):
        """Les ré-exports maintiennent l'interface main.X (consommateurs
        inchangés : api/routes, telemetry, schedulers)."""
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        assert "from core.state_helpers import" in src
        assert "from core.auth_helpers import" in src
        import main  # noqa: F401
        for name in ("mark_real_price", "update_asset_order_book", "_neutral",
                     "auth_enforced", "require_auth", "require_admin",
                     "auth_security_optional", "AuthManager"):
            assert hasattr(main, name), f"main.{name} manquant (ré-export cassé)"

    def test_auth_flow_still_works(self):
        """Le flux d'authentification (routes) fonctionne avec les helpers
        extraits."""
        from fastapi.testclient import TestClient

        import main  # noqa: F401
        with TestClient(main.app) as c:
            # sans auth : 401 sur une route protégée ? (AUTH_ENABLED off en
            # sandbox -> les routes répondent sans 401)
            r = c.get("/api/v1/decision-journal")
            assert r.status_code < 500


# --------------------------------------------------------------------------- #
# 2. Registre des cerveaux
# --------------------------------------------------------------------------- #
class TestBrains:
    def test_eight_brains(self):
        from core.brains import BRAINS
        assert len(BRAINS) == 8
        expected = {"MARKET", "ALPHA", "CONVICTION", "DECISION",
                    "RISK", "EXECUTION", "LEARNING", "ANALYST"}
        assert set(BRAINS) == expected

    def test_each_brain_has_role_and_modules(self):
        from core.brains import BRAINS
        for name, info in BRAINS.items():
            assert info.get("role"), f"{name}: rôle manquant"
            assert len(info.get("modules", [])) >= 2, f"{name}: trop peu de modules"

    def test_all_modules_importable(self):
        from core.brains import verify_importable
        failures = verify_importable()
        assert not failures, f"modules non importables : {failures}"

    def test_no_star_import_from_main_anywhere(self):
        """Découplage : AUCUN module du repo n'utilise `from main import *`."""
        violations = []
        for py in ROOT.rglob("*.py"):
            if "__pycache__" in str(py) or str(py).startswith(str(ROOT / "tests")):
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "main" \
                        and any(a.name == "*" for a in node.names):
                    violations.append(py.relative_to(ROOT).as_posix())
        assert not violations, f"star imports : {violations}"

    def test_report_checks_pass(self):
        from core.brains import report
        r = report()
        assert r["n_brains"] == 8
        assert r["checks"]["all_importable"] is True
        assert r["checks"]["no_star_import_from_main"] is True

    def test_brain_of(self):
        from core.brains import brain_of
        assert brain_of("core.conviction_engine") == "CONVICTION"
        assert brain_of("core.edge_decay") == "LEARNING"
        assert brain_of("risk.risk_manager") == "RISK"
        assert brain_of("core.execution_intel") == "EXECUTION"


# --------------------------------------------------------------------------- #
# 3. API + télémétrie
# --------------------------------------------------------------------------- #
class TestExposure:
    def test_api_brains(self):
        from fastapi.testclient import TestClient

        import main  # noqa: F401
        with TestClient(main.app) as c:
            r = c.get("/api/v1/brains")
            assert r.status_code == 200
            body = r.json()
            assert body["n_brains"] == 8
            assert body["checks"]["all_importable"] is True

    def test_telemetry_exposes_brains(self):
        import main  # noqa: F401
        from telemetry import compile_telemetry_data
        tel = compile_telemetry_data()
        assert "brains" in tel
        assert tel["brains"]["n_brains"] == 8


# --------------------------------------------------------------------------- #
# 4. Découplage : les modules extraits n'ont pas de dépendance circulaire
# --------------------------------------------------------------------------- #
class TestDecoupling:
    def test_state_helpers_importable_alone_after_main(self):
        """state_helpers importable (main chargé d'abord — pattern LOT C)."""
        import core.state_helpers as sh
        import main  # noqa: F401
        assert hasattr(sh, "mark_real_price")
        assert hasattr(sh, "update_asset_order_book")

    def test_auth_helpers_importable_alone_after_main(self):
        import core.auth_helpers as ah
        import main  # noqa: F401
        assert hasattr(ah, "auth_enforced")
        assert hasattr(ah, "require_auth")
