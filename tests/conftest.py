import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ===========================================================================
# DB DE TEST ISOLÉE (contention SQLite réglée) :
# les tests écrivent dans UNE COPIE de la base réelle (ou une base vide), PAS
# dans trading_platform.db. Avant : lancer pytest pendant que le serveur
# uvicorn écrit dans la même DB rendait la suite 15x plus lente (500s+ vs 30s)
# et polluait la vraie base. SQLITE_DB_PATH est lu par database/db_manager.py
# au moment de l'import — on le définit ICI, avant tout import de main.
# ===========================================================================
_TEST_DB = os.path.join(os.path.dirname(__file__), "test_trading.db")
os.environ["SQLITE_DB_PATH"] = _TEST_DB
if os.path.exists(_TEST_DB):
    try:
        os.remove(_TEST_DB)  # base fraîche à chaque session de test
    except OSError:
        pass

@pytest.fixture
def sample_returns():
    import numpy as np
    return {
        "BTCUSDT": np.random.randn(100) * 0.02,
        "ETHUSDT": np.random.randn(100) * 0.025
    }


def all_api_paths(app) -> list:
    """
    Liste récursive des chemins de routes d'une app FastAPI — déroule les
    wrappers _IncludedRouter (FastAPI >= 0.141 monte les routers inclus dans
    un wrapper privé : les chemins ne sont plus visibles par r.path direct).
    """
    paths = []

    def walk(routes):
        for r in routes:
            if hasattr(r, "path"):
                paths.append(r.path)
            # wrapper FastAPI 0.141+ : routes réelles dans original_router
            orig = getattr(r, "original_router", None)
            if orig is not None and hasattr(orig, "routes"):
                walk(orig.routes)
            elif hasattr(r, "routes"):
                walk(r.routes)

    walk(app.router.routes)
    return paths
