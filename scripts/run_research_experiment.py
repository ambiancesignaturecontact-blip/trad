"""
PHASE 3 Cycle 2 — Exécution d'une expérience de recherche (item 2).

Usage :
    python scripts/run_research_experiment.py [experiment_id]

Exécute le pipeline de validation (baseline vs traitement isolé, walk-forward
70/30, stress haute vol réelle) pour l'expérience donnée (défaut : 1) et
enregistre la décision REJECT/KEEP dans la mémoire de recherche (table
`experiments`). Ne modifie JAMAIS la production.
"""
import json
import sys

sys.path.insert(0, ".")

from core.research_experiments import run_experiment  # noqa: E402
from database.db_manager import DBManager  # noqa: E402


def main():
    experiment_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    db = DBManager()
    print(f"🔬 EXPÉRIENCE #{experiment_id} — pipeline de validation\n"
          f"   Données : cache DB market_candles (réelles) · coûts AR "
          f"0,213 % (frais réels 0,1 %/side + slippage médian 6,6 bps)")
    results = run_experiment(db, experiment_id)
    print(json.dumps(results, ensure_ascii=False, indent=1, default=str))
    oos = results.get("oos", {})
    print("\n=== DÉCISION ===")
    print(f"  {results.get('decision')} · n trades OOS : "
          f"{oos.get('n_oos_trades')} · enregistré : {results.get('recorded')}")
    return 0 if results.get("recorded") else 1


if __name__ == "__main__":
    sys.exit(main())
