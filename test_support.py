"""Helpers partagés par les tests (importables depuis la racine du repo)."""


def all_api_paths(app) -> list:
    """Liste récursive des chemins de routes d'une app FastAPI — déroule les
    wrappers _IncludedRouter (FastAPI >= 0.141 monte les routers inclus dans
    un wrapper privé : les chemins ne sont plus visibles par r.path direct)."""
    paths = []

    def walk(routes):
        for r in routes:
            if hasattr(r, "path"):
                paths.append(r.path)
            orig = getattr(r, "original_router", None)
            if orig is not None and hasattr(orig, "routes"):
                walk(orig.routes)
            elif hasattr(r, "routes"):
                walk(r.routes)

    walk(app.router.routes)
    return paths
