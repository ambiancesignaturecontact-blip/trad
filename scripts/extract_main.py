"""
LOT 7 (P1-7 audit §4.1) : extraction automatisée des routes API et des
schedulers hors de main.py, par AST (précis au nœud près, corps inchangés).

- Les fonctions décorées @app.get/post/put/delete/patch (SAUF "/", "/telegram")
  sont déplacées vers api/routes.py avec @router.X.
- Les schedulers listés sont déplacés vers schedulers.py.
- main.py ré-exporte les noms déplacés et fait include_router en fin de fichier.
- Les modules extraits accèdent aux symboles de main via `from main import *`
  (main est COMPLET au moment de l'import, qui se fait en fin de main.py) —
  étape 1 du découpage : sortir le code, pas encore refactorer les dépendances.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"

SCHEDULERS = [
    "final_scale_stats_loop",
    "reconciliation_scheduler",
    "concierge_scheduler",
    "db_backup_scheduler",
    "copy_trading_refresh_scheduler",
    "copy_mirror_scheduler",
]

KEEP_ROUTES = {"/", "/telegram"}  # routes HTML servies par main (templates)


def collect_route_nodes(tree):
    """Fonctions décorées par une route API (hors routes HTML)."""
    out = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            # @app.get(...) est un ast.Call dont func est Attribute(value=Name('app'))
            if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                    and isinstance(dec.func.value, ast.Name) and dec.func.value.id == "app"
                    and dec.func.attr in ("get", "post", "put", "delete", "patch")):
                # chemin de la route (1er argument string)
                if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
                    if dec.args[0].value in KEEP_ROUTES:
                        break
                out.append(node)
                break
    return out


def collect_scheduler_nodes(tree):
    out = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in SCHEDULERS:
            out.append(node)
    return out


def extract(src, nodes):
    """Supprime les segments des nœuds (du plus tardif au plus précoce) et
    retourne (nouveau_src, {nom: segment}). La suppression part du PREMIER
    DÉCORATEUR (sinon les @app.X orphelins s'attachent à la fonction suivante)."""
    segments = {}
    for node in sorted(nodes, key=lambda n: n.lineno, reverse=True):
        # ligne de départ = premier décorateur (ou def s'il n'y en a pas)
        if node.decorator_list:
            start = min(d.lineno for d in node.decorator_list) - 1
        else:
            start = node.lineno - 1
        lines = src.splitlines(keepends=True)
        end = node.end_lineno
        # segment construit manuellement (get_source_segment omet les décorateurs)
        seg = "".join(lines[start:end])
        del lines[start:end]
        # retire les lignes vides qui précédaient (jusqu'à 3)
        removed = 0
        while removed < 3 and start - 1 - removed >= 0 and lines[start - 1 - removed].strip() == "":
            removed += 1
        if removed:
            del lines[start - removed:start]
        src = "".join(lines)
        segments[node.name] = seg
    return src, segments


def private_names_used(segments, module_globals):
    """Noms _privés utilisés par les segments extraits ET définis au niveau
    module de main (exclut les variables locales des fonctions)."""
    used = set()
    for seg in segments.values():
        tree = ast.parse(seg)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id.startswith("_") and isinstance(node.ctx, ast.Load):
                if node.id in module_globals:
                    used.add(node.id)
    return sorted(used)


def main():
    src = MAIN.read_text(encoding="utf-8")
    tree = ast.parse(src)

    route_nodes = collect_route_nodes(tree)
    sched_nodes = collect_scheduler_nodes(tree)
    print(f"routes extraites : {len(route_nodes)} | schedulers : {len(sched_nodes)}")

    new_src, segments = extract(src, route_nodes + sched_nodes)
    # globals définis au niveau module de main (pour filtrer les _privés)
    module_globals = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_globals.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id.startswith("_"):
                    module_globals.add(t.id)
    priv = private_names_used(segments, module_globals)

    # ---- api/routes.py ----
    api_dir = ROOT / "api"
    api_dir.mkdir(exist_ok=True)
    (api_dir / "__init__.py").write_text('"""Module des routes API (extrait de main.py, LOT 7)."""\n', encoding="utf-8")
    route_segs = {n.name: segments[n.name] for n in route_nodes}
    routes_hdr = (
        '"""\nRoutes API extraites de main.py (LOT 7, P1-7 audit §4.1).\n'
        "Corps des fonctions STRICTEMENT inchangés ; les décorateurs @app.X\n"
        "deviennent @router.X. Les symboles partagés viennent de main via\n"
        "`from main import *` (main est complet quand ce module est importé,\n"
        "en fin de main.py). Étape 1 du découpage : sortir le code, pas encore\n"
        "refactorer les dépendances.\n\"\"\"\n"
        "from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect\n"
        "from fastapi.responses import HTMLResponse, JSONResponse, Response\n"
        "from fastapi.staticfiles import StaticFiles\n"
        "from fastapi.templating import Jinja2Templates\n"
        "from pydantic import BaseModel, Field, field_validator\n"
        "from typing import List, Dict\n"
        "import json\nimport os\nimport time\nimport asyncio\nimport numpy as np\nimport pandas as pd\n\n"
        "import main  # noqa: F401  (le module est complet au moment de cet import)\n"
        "from main import *  # noqa: F401,F403  (symboles partagés, étape 1 du découpage)\n"
    )
    if priv:
        routes_hdr += f"from main import {', '.join(priv)}  # noqa: F401\n"
    routes_hdr += "\nrouter = APIRouter()\n\n\n"
    routes_body = []
    for n in route_nodes:
        seg = route_segs[n.name]
        # réécrire le décorateur @app.X -> @router.X
        seg = seg.replace("@app.get", "@router.get").replace("@app.post", "@router.post") \
                 .replace("@app.put", "@router.put").replace("@app.delete", "@router.delete") \
                 .replace("@app.patch", "@router.patch")
        routes_body.append(seg)
    (api_dir / "routes.py").write_text(routes_hdr + "\n\n".join(routes_body) + "\n", encoding="utf-8")

    # ---- schedulers.py ----
    sched_segs = {n.name: segments[n.name] for n in sched_nodes}
    sched_hdr = (
        '"""\nSchedulers / boucles de fond extraits de main.py (LOT 7, P1-7 audit §4.1).\n'
        "Corps STRICTEMENT inchangés ; les symboles partagés viennent de main via\n"
        "`from main import *` (main est complet quand ce module est importé, en fin\n"
        "de main.py).\n\"\"\"\n"
        "import json\nimport os\nimport time\nimport asyncio\n\n"
        "import main  # noqa: F401\n"
        "from main import *  # noqa: F401,F403\n"
    )
    if priv:
        sched_hdr += f"from main import {', '.join(priv)}  # noqa: F401\n"
    (ROOT / "schedulers.py").write_text(sched_hdr + "\n\n\n".join(sched_segs[n.name] for n in sched_nodes) + "\n", encoding="utf-8")

    # ---- main.py : ré-exports + include_router en fin de fichier ----
    footer = (
        "\n\n# ============ LOT 7 (P1-7 audit §4.1) : modules extraits ============\n"
        "# Les définitions ont été déplacées vers api/routes.py et schedulers.py ;\n"
        "# on ré-exporte les noms pour préserver l'espace de noms de main (les\n"
        "# tests et TASK_FACTORIES y accèdent) et on monte le router des routes API.\n"
        "from schedulers import (" + ", ".join(SCHEDULERS) + ")  # noqa: F401,E402\n"
        "from api.routes import router as _api_router  # noqa: E402\n"
        "app.include_router(_api_router)\n"
    )
    new_src = new_src.rstrip() + footer
    MAIN.write_text(new_src, encoding="utf-8")

    print(f"main.py : {len(src.splitlines())} -> {len(new_src.splitlines())} lignes")
    print(f"helpers privés référencés : {priv}")


if __name__ == "__main__":
    sys.exit(main())
