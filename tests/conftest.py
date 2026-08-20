import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
