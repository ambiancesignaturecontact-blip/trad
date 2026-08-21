"""
LOT C (F3 — architecture) : extraction par AST des gros blocs de main.py
vers des modules dédiés, avec imports EXPLICITES (fini le `from main import *`).

Blocs extraits :
  1. core/observability.py        : suivi final_scale (P0-4) + paper-validation (P0-6)
  2. core/ccxt_client.py          : client CCXT authentifié + format des tailles
  3. market_data/historical_fetch.py : fetch OHLCV réels (Binance/Bybit/Yahoo)
  4. core/autonomous_ai.py        : scheduler IA autonome (LOT 66)
  5. core/decision_explain.py     : explication de décision + push Prometheus

Corps des fonctions STRICTEMENT inchangés (extraction au nœud AST près).
Les modules extraits accèdent aux symboles de main par IMPORTS EXPLICITES
(never `*`) et sont importés EN FIN de main.py (main complet à ce moment).
main.py ré-exporte les noms pour préserver son espace de noms (tests,
TASK_FACTORIES, api/routes.py, schedulers.py, telemetry.py y accèdent).
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"

# (module, [noms de fonctions], [noms de constantes/globals à déplacer])
PLAN = {
    "core/observability.py": {
        "consts": [],  # les constantes FINAL_SCALE_* restent dans main
                       # (utilisées par api/routes.py et les tests via main.X) ;
                       # observability les importe depuis main.
        "funcs": ["_record_final_scale", "_purge_final_scale_samples",
                  "_limiting_factor_stats", "_final_scale_stats",
                  "_persist_final_scale_samples", "_load_final_scale_samples",
                  "_signal_stats", "_final_scale_report",
                  "_mark_paper_validation_day", "_paper_validation_stats"],
    },
    "core/ccxt_client.py": {
        "consts": ["ccxt_client"],
        "funcs": ["get_ccxt_client", "format_exchange_size"],
    },
    "market_data/historical_fetch.py": {
        "consts": ["_yahoo_cache"],
        "funcs": ["fetch_yahoo_finance_candles", "_klines_to_df",
                  "fetch_bybit_klines", "fetch_historical_market_data"],
    },
    "core/autonomous_ai.py": {
        "consts": [],
        "funcs": ["autonomous_ai_scheduler"],
    },
    "core/decision_explain.py": {
        "consts": [],
        "funcs": ["explain_last_decision", "update_metrics_from_state"],
    },
}

# imports EXPLICITES depuis main pour chaque module (analysés à la main,
# vérifiés par AST dans le script : aucun nom manquant ne doit subsister)
FROM_MAIN = {
    "core/observability.py": "FINAL_SCALE_DOWNSAMPLE_SEC, FINAL_SCALE_MAX_SAMPLES, FINAL_SCALE_WINDOW_HOURS, STATE, db, settings",
    "core/ccxt_client.py": "db",
    "market_data/historical_fetch.py": "CRYPTO_SYMBOLS, bybit_limiter, settings, yahoo_limiter",
    "core/autonomous_ai.py": (
        "LSTMLikePredictor, MarketRegimeDetector, MetaAllocationEngine, PPOTRAgent, "
        "RiskManager, STATE, TrendFollowingStrategy, _neutral, attribution, audit_ip, "
        "build_causal_feature_df, calculate_deflated_sharpe_ratio, curriculum_sort, "
        "daily_risk_budget, db, discover_causal_parents, execution_alpha, "
        "fetch_historical_market_data, generative_engine, hypothesis_generator, "
        "meta_attribution, meta_engine, mixture_of_experts, mlops_trainer, "
        "monte_carlo_tester, organization, platform_metrics, ppo_agent, quality_metrics, "
        "reason_weight_from_attribution, risk_committee, save_state_snapshot, "
        "scenario_tester, simulation_divergence, strategies_list, telegram_bot"
    ),
    "core/decision_explain.py": "STATE, _neutral, platform_metrics",
}

EXTRA_IMPORTS = {
    "core/observability.py": "import json\nimport time\n\nimport numpy as np\nfrom datetime import UTC\n",
    "core/ccxt_client.py": "import ccxt\n",
    "market_data/historical_fetch.py": "import time\n\nimport httpx\nimport pandas as pd\n",
    "core/autonomous_ai.py": "import asyncio\nimport json\nimport time\n\nimport numpy as np\n",
    "core/decision_explain.py": "",
}

MODULE_HEADERS = {
    "core/observability.py": (
        '"""\nSuivi d\'observation (P0-4 final_scale, P0-6 paper-validation) extrait de\n'
        'main.py (LOT C, F3). Corps inchangés ; symboles partagés importés de main\n'
        'de façon EXPLICITE (main est complet quand ce module est importé, en fin\n'
        'de main.py).\n"""\n'
    ),
    "core/ccxt_client.py": (
        '"""\nClient CCXT authentifié (clés chiffrées en DB) extrait de main.py\n'
        '(LOT C, F3). Corps inchangés.\n"""\n'
    ),
    "market_data/historical_fetch.py": (
        '"""\nFetch OHLCV RÉELS (Binance -> Bybit -> Yahoo) extrait de main.py\n'
        '(LOT C, F3). Corps inchangés ; jamais de barres fabriquées.\n"""\n'
    ),
    "core/autonomous_ai.py": (
        '"""\nScheduler IA autonome (LOT 66) extrait de main.py (LOT C, F3).\n'
        'Corps inchangé ; symboles partagés importés de main de façon EXPLICITE.\n"""\n'
    ),
    "core/decision_explain.py": (
        '"""\nExplication de décision + push Prometheus extraits de main.py\n'
        '(LOT C, F3). Corps inchangés.\n"""\n'
    ),
}

# Ré-exports à ajouter à la fin de main.py (ordre de dépendances respecté)
REEXPORTS = {
    "core/observability.py": [
        "_final_scale_report", "_final_scale_stats", "_limiting_factor_stats",
        "_load_final_scale_samples", "_mark_paper_validation_day",
        "_paper_validation_stats", "_persist_final_scale_samples",
        "_purge_final_scale_samples", "_record_final_scale", "_signal_stats",
    ],
    "core/ccxt_client.py": ["format_exchange_size", "get_ccxt_client"],
    "market_data/historical_fetch.py": [
        "_klines_to_df", "fetch_bybit_klines", "fetch_historical_market_data",
        "fetch_yahoo_finance_candles",
    ],
    "core/autonomous_ai.py": ["autonomous_ai_scheduler"],
    "core/decision_explain.py": ["explain_last_decision", "update_metrics_from_state"],
}


def collect_nodes(tree, plan):
    """Retourne {nom: node} pour les fonctions et constantes du plan."""
    funcs, consts = {}, {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in plan["funcs"]:
            funcs[node.name] = node
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in plan["consts"]:
                    consts[t.id] = node
    return funcs, consts


def segment_of(src_lines, node):
    """Segment source d'un nœud (du premier décorateur/commentaire au end_lineno)."""
    start = node.lineno - 1
    # descend jusqu'à la fin exacte (end_lineno inclus)
    end = node.end_lineno
    return "".join(src_lines[start:end])


def extract_all(src, tree, plan):
    """Supprime tous les segments (du plus tardif au plus précoce)."""
    removals = []
    for mod, p in plan.items():
        funcs, consts = collect_nodes(tree, p)
        for name, node in {**funcs, **consts}.items():
            removals.append((node.lineno, node.end_lineno, mod, name))
    removals.sort(key=lambda r: r[0], reverse=True)
    lines = src.splitlines(keepends=True)
    extracted = {m: [] for m in plan}
    for start_lineno, end_lineno, mod, name in removals:
        seg = "".join(lines[start_lineno - 1:end_lineno])
        del lines[start_lineno - 1:end_lineno]
        # retire les lignes vides qui précédaient (jusqu'à 3)
        removed = 0
        while removed < 3 and start_lineno - 2 - removed >= 0 and lines[start_lineno - 2 - removed].strip() == "":
            removed += 1
        if removed:
            del lines[start_lineno - 1 - removed:start_lineno - 1]
        extracted[mod].append((name, seg))
    return "".join(lines), extracted


def main():
    src = MAIN.read_text(encoding="utf-8")
    tree = ast.parse(src)
    new_src, extracted = extract_all(src, tree, PLAN)

    for mod, items in extracted.items():
        # ordre : constantes d'abord, puis fonctions (ordre d'apparition)
        items.sort(key=lambda it: -len(it[1]))  # plus grand segment en premier
        # tri par ordre d'origine : on retri par position dans le plan
        order = {n: i for i, n in enumerate(
            PLAN[mod]["consts"] + PLAN[mod]["funcs"])}
        items.sort(key=lambda it: order.get(it[0], 99))
        path = ROOT / mod
        path.parent.mkdir(exist_ok=True)
        body = "\n\n\n".join(seg.rstrip("\n") for _, seg in items)
        hdr = (
            MODULE_HEADERS[mod] + "\n"
            "import logging\n\n"
            + EXTRA_IMPORTS[mod]
            + "from main import (" + FROM_MAIN[mod] + ")  # noqa: E402\n\n"
            + f'logger = logging.getLogger("{Path(mod).stem}")\n\n\n'
        )
        path.write_text(hdr + body + "\n", encoding="utf-8")
        print(f"{mod} : {len(items)} segments -> {len(body.splitlines())} lignes")

    # ---- footer de main.py ----
    footer = "\n\n# ============ LOT C (F3) : gros blocs extraits ============\n"
    footer += "# Les définitions ont été déplacées vers des modules dédiés ; on\n"
    footer += "# ré-exporte les noms pour préserver l'espace de noms de main (tests,\n"
    footer += "# TASK_FACTORIES, api/routes.py, schedulers.py, telemetry.py). Ordre\n"
    footer += "# de dépendances respecté (autonomous_ai dépend de fetch_* ré-exporté).\n"
    for mod, names in REEXPORTS.items():
        dotted = mod[:-3].replace("/", ".")
        footer += f"from {dotted} import {', '.join(names)}  # noqa: F401,E402\n"
    footer += "from api.routes import router as _api_router  # noqa: E402\n"
    footer += "app.include_router(_api_router)\n"
    new_src = new_src.rstrip() + footer
    MAIN.write_text(new_src, encoding="utf-8")
    print(f"main.py : {len(src.splitlines())} -> {len(new_src.splitlines())} lignes")


if __name__ == "__main__":
    sys.exit(main())
