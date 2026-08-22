"""
PHASE 3 Cycle 2-4 — Exécution d'une expérience de recherche.

Usage :
    python scripts/run_research_experiment.py [experiment_id] [signal|position] [1h|4h|1d]

Exécute le pipeline de validation (baseline vs traitement isolé, walk-forward
70/30, stress haute vol réelle) pour l'expérience donnée (défaut : 1) et
enregistre la décision REJECT/KEEP dans la mémoire de recherche (table
`experiments`). `filter_mode` : "signal" (Cycle 2) ou "position" (Cycle 3,
affinement : filtre sur le poids final). `timeframe` : "1h" (production),
"4h"/"1d" (Cycle 4, Expérience #3 — horizon long). Ne modifie JAMAIS la
production.
"""
import json
import sys

sys.path.insert(0, ".")

from core.research_experiments import run_experiment  # noqa: E402
from database.db_manager import DBManager  # noqa: E402


def main():
    experiment_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    filter_mode = sys.argv[2] if len(sys.argv) > 2 else "signal"
    timeframe = sys.argv[3] if len(sys.argv) > 3 else "1h"
    signal_family = sys.argv[4] if len(sys.argv) > 4 else "momentum"
    symbols = tuple(s.strip() for s in sys.argv[5].split(",")) \
        if len(sys.argv) > 5 and sys.argv[5] else None
    cost_ar = float(sys.argv[6]) if len(sys.argv) > 6 else None
    assert filter_mode in ("signal", "position")
    assert timeframe in ("1h", "4h", "1d")
    assert signal_family in ("momentum", "contrarian", "flow")
    db = DBManager()
    print(f"🔬 EXPÉRIENCE #{experiment_id} (mode filtre : {filter_mode}, "
          f"timeframe : {timeframe}, famille : {signal_family}, "
          f"symboles : {symbols or 'défaut'}) — pipeline de validation\n"
          f"   Données : cache DB market_candles (réelles) · coûts AR "
          f"{cost_ar if cost_ar else 0.213} % (frais réels 0,1 %/side + "
          f"slippage médian mesuré)")
    results = run_experiment(db, experiment_id, filter_mode=filter_mode,
                             timeframe=timeframe,
                             signal_family=signal_family,
                             symbols=symbols,
                             cost_ar_pct=cost_ar)
    print(json.dumps(results, ensure_ascii=False, indent=1, default=str))
    oos = results.get("oos", {})
    print("\n=== DÉCISION ===")
    print(f"  {results.get('decision')} · n trades OOS : "
          f"{oos.get('n_oos_trades')} · enregistré : {results.get('recorded')}")
    return 0 if results.get("recorded") else 1


if __name__ == "__main__":
    sys.exit(main())
