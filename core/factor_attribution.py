"""
FACTOR ATTRIBUTION (PHASE 4 — P4-B, axe 3 du mandat).

« Quelle partie de notre P&L vient réellement de nos décisions et quelle
partie vient simplement d'une exposition factorielle (mouvement du marché) ? »

Pour chaque trade clôturé :
    pnl_pct          : rendement NET du trade (décision + marché + coûts)
    marginal_alpha   : alpha contrefactuel = ret_trade − benchmark_return
                       (déjà persisté à la clôture, core/world_model.py)
  => benchmark implicite : bench = pnl_pct − marginal_alpha
     (par construction : alpha = ret − bench et pnl ≈ ret net)

Le rapport décompose le PnL agrégé en :
    part_beta_pct  : % du PnL expliqué par le mouvement du marché (bêta)
    part_alpha_pct : % du PnL expliqué par les décisions (alpha résiduel)
avec moyenne ET médiane (robustesse aux outliers — mesuré : 192/230 trades
alpha > 0 mais moyenne tirée par quelques gros négatifs) + quantiles.

Sources :
  - ACTUELLE  : decision_journal (pnl_pct NOT NULL, versionné) — la seule qui
                compte pour le calibrage actuel ; 0 clôture -> « insuffisant ».
  - HISTORIQUE: events closed_trade_alpha (230 clôtures, ANCIEN calibrage) —
                exposée avec RÉSERVE explicite (calibrage différent, pas de
                dates de détention -> benchmark reconstruit par différence).

Principes : jamais de chiffre inventé ; sans échantillon -> dict honnête.
"""
import json
import logging

logger = logging.getLogger("InstitutionalTradingBot")

MIN_SAMPLES = 10


# --------------------------------------------------------------------------- #
# Chargement des clôtures
# --------------------------------------------------------------------------- #
def _load_journal_closes(db) -> list[dict]:
    """Clôtures du calibrage ACTUEL (decision_journal, pnl_pct NOT NULL)."""
    out = []
    try:
        if db is None or not hasattr(db, "get_connection"):
            return out
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT symbol, strategy, pnl_pct, exit_reason, "
                "system_version FROM decision_journal "
                "WHERE pnl_pct IS NOT NULL ORDER BY ts ASC")
            for r in cur.fetchall():
                try:
                    out.append({
                        "symbol": str(r[0]), "strategy": str(r[1] or "?"),
                        "pnl_pct": float(r[2]),
                        "exit_reason": str(r[3] or ""),
                        "system_version": str(r[4] or ""),
                    })
                except Exception:
                    continue
    except Exception as e:
        logger.debug(f"_load_journal_closes failed: {e}")
    return out


def _load_historical_closes(db, limit: int = 5000) -> list[dict]:
    """Clôtures HISTORIQUES (events closed_trade_alpha : pnl_pct +
    marginal_alpha — le benchmark est reconstruit par différence)."""
    out = []
    try:
        if db is None or not hasattr(db, "list_events"):
            return out
        evs = db.list_events(event_type="closed_trade_alpha", limit=limit)
        for e in evs:
            try:
                d = json.loads(e.get("payload", "{}"))
                pnl = float(d.get("pnl_pct"))
                alpha = float(d.get("marginal_alpha"))
                if not all(map(lambda x: x == x, (pnl, alpha))):  # NaN
                    continue
                out.append({
                    "symbol": str(d.get("symbol", "?")),
                    "strategy": str(d.get("strategy", "?")),
                    "pnl_pct": pnl,
                    "marginal_alpha": alpha,
                    "bench_pct": round(pnl - alpha, 6),
                })
            except Exception:
                continue
    except Exception as e:
        logger.debug(f"_load_historical_closes failed: {e}")
    return out


# --------------------------------------------------------------------------- #
# Statistiques robustes
# --------------------------------------------------------------------------- #
def _stats(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean_pct": None, "median_pct": None,
                "p10_pct": None, "p90_pct": None}
    import numpy as np
    a = np.asarray(values, dtype=float)
    return {
        "n": n,
        "mean_pct": round(float(a.mean()) * 100.0, 4),
        "median_pct": round(float(np.median(a)) * 100.0, 4),
        "p10_pct": round(float(np.percentile(a, 10)) * 100.0, 4),
        "p90_pct": round(float(np.percentile(a, 90)) * 100.0, 4),
    }


def _attribution(trades: list[dict]) -> dict:
    """Décompose le PnL agrégé en part alpha vs part bêta (moyenne et
    médiane) + distribution. Retourne dict honnête."""
    if len(trades) < MIN_SAMPLES:
        return {"n_trades": len(trades), "status": "INSUFFICIENT",
                "note": (f"{len(trades)} clôture(s) — il en faut ≥ "
                         f"{MIN_SAMPLES} pour une attribution significative")}
    pnls = [t["pnl_pct"] for t in trades]
    alphas = [t["marginal_alpha"] for t in trades]
    benchs = [t.get("bench_pct", t["pnl_pct"] - t["marginal_alpha"])
              for t in trades]
    # moyenne pondérée également par trade ; la part alpha/bêta est calculée
    # sur les TOTAUX (somme) — la somme des alphas et la somme des bêta
    total_pnl = float(sum(pnls))
    total_alpha = float(sum(alphas))
    total_bench = float(sum(benchs))
    denom = abs(total_pnl) if total_pnl != 0 else 1e-9
    return {
        "n_trades": len(trades),
        "status": "OK",
        "pnl": _stats(pnls),
        "alpha": _stats(alphas),
        "beta": _stats(benchs),
        "sum_pnl_pct": round(total_pnl * 100.0, 4),
        "sum_alpha_pct": round(total_alpha * 100.0, 4),
        "sum_beta_pct": round(total_bench * 100.0, 4),
        "share_alpha_pct": round(total_alpha / denom * 100.0, 2),
        "share_beta_pct": round(total_bench / denom * 100.0, 2),
        "pct_trades_alpha_positive": round(
            sum(1 for a in alphas if a > 0) / len(alphas) * 100.0, 1),
        "note": ("share_alpha/share_beta = parts du PnL TOTAL expliquées par "
                 "les décisions (alpha) vs le mouvement du marché (bêta) ; "
                 "les sommes peuvent dépasser 100 % en sens opposés"),
    }


def alpha_beta_report(db) -> dict:
    """Rapport complet : calibrage ACTUEL (decision_journal) + HISTORIQUE
    (events closed_trade_alpha, avec réserve)."""
    current = _load_journal_closes(db)
    historical = _load_historical_closes(db)
    out = {
        "current": {
            "n_closes": len(current),
            "attribution": _attribution(
                [{"pnl_pct": t["pnl_pct"], "marginal_alpha": 0.0,
                  "bench_pct": 0.0} for t in current])
            if len(current) >= MIN_SAMPLES else
            {"n_trades": len(current), "status": "INSUFFICIENT",
             "note": (f"{len(current)} clôture(s) du calibrage actuel — il en "
                      f"faut ≥ {MIN_SAMPLES} (condition du CONDITIONAL GO)")},
            "note": "Source : decision_journal (calibrage ACTUEL, versionné)",
        },
        "historical": None,
    }
    if historical:
        out["historical"] = {
            "n_closes": len(historical),
            "attribution": _attribution(historical),
            "reserve": ("ANCIEN calibrage (avant purge PHASE 2) ; dates de "
                        "détention absentes -> benchmark RECONSTRUIT par "
                        "différence (bench = pnl − alpha). Indicatif, "
                        "non décisionnel."),
        }
    else:
        out["historical"] = {"n_closes": 0,
                             "attribution": None,
                             "reserve": "aucune clôture historique"}
    out["note"] = ("L'attribution factorielle sépare les DÉCISIONS (alpha) "
                   "du MOUVEMENT DU MARCHÉ (bêta). Seul le calibrage ACTUEL "
                   "est décisionnel.")
    return out
