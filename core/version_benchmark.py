"""
VERSION BENCHMARK — LE SYSTÈME EST SON PROPRE BENCHMARK (PHASE 4, P4-E, axe 10).

« QUANT-PORTAL v1, v2, v3… Chaque nouvelle version doit prouver qu'elle est
MEILLEURE que la précédente APRÈS COÛTS ET RISQUE. Sinon elle n'est pas
promue. »

Chaque décision du journal est versionnée (system_version, LOT 9) : on peut
donc comparer les versions entre elles sur les clôtures RÉELLES.

Ce module fournit :
  1. `_closes_per_version`   : clôtures (pnl_pct NOT NULL) groupées par
     system_version, ordonnées par ts ;
  2. `version_performance`   : métriques d'une version (n, win rate,
     expectancy moyenne, pnl cumulé, drawdown de la courbe cumulée) ;
  3. `version_gate`          : décision PURE — la nouvelle version est-elle
     meilleure que la précédente après coûts et risque ?
       - PROMOTED   : n >= MIN_CLOSES des deux côtés ET expectancy_new >
                      expectancy_old ET drawdown non dégradé ;
       - NOT_PROMOTED : la nouvelle version ne bat pas l'ancienne ;
       - INSUFFICIENT : échantillon insuffisant d'un côté (défaut honnête).
  4. `version_benchmark_report` : tableau des versions + verdicts + note.

Principes :
  - PnL net : pnl_pct est le rendement de la clôture (brut de frais
    d'exécution — même base pour TOUTES les versions -> comparaison relative
    valide ; le gate de friction couvre les coûts séparément).
  - Drawdown : calculé sur la courbe cumulée des clôtures (approximation
    honnête, documentée) — le mandat exige « après coûts ET RISQUE ».
  - JAMAIS de promotion sans échantillon : INSUFFICIENT par défaut.
  - Le gate est un AVIS enregistré — il ne modifie aucun statut.
"""
import logging
from collections import defaultdict

logger = logging.getLogger("InstitutionalTradingBot")

MIN_CLOSES_GATE = 30            # clôtures minimales par version pour comparer
DD_DEGRADE_TOL = 0.20           # tolérance drawdown (20 % relatif)


# --------------------------------------------------------------------------- #
# Chargement des clôtures par version
# --------------------------------------------------------------------------- #
def _closes_per_version(db) -> dict:
    """{system_version: [clôtures triées par ts]} — clôtures réelles
    (pnl_pct NOT NULL) du decision_journal. Toutes les versions présentes
    dans le journal sont listées (avec liste vide si 0 clôture)."""
    out = defaultdict(list)
    try:
        if db is None or not hasattr(db, "get_connection"):
            return out
        with db.get_connection() as conn:
            cur = conn.cursor()
            # toutes les versions distinctes (transparence, même sans clôture)
            cur.execute(
                "SELECT DISTINCT COALESCE(system_version, 'legacy_sans_version') "
                "FROM decision_journal")
            for r in cur.fetchall():
                out.setdefault(str(r[0]), [])
            cur.execute(
                "SELECT ts, COALESCE(system_version, 'legacy_sans_version'), "
                "pnl_pct, duration_sec, symbol "
                "FROM decision_journal WHERE pnl_pct IS NOT NULL "
                "ORDER BY ts ASC")
            for r in cur.fetchall():
                try:
                    v = str(r[1])
                    out[v].append({"ts": float(r[0]), "pnl_pct": float(r[2]),
                                   "duration_sec": (float(r[3])
                                                    if r[3] is not None
                                                    else None),
                                   "symbol": str(r[4] or "?")})
                except Exception:
                    continue
    except Exception as e:
        logger.debug(f"_closes_per_version failed: {e}")
    return out


# --------------------------------------------------------------------------- #
# Métriques d'une version
# --------------------------------------------------------------------------- #
def version_performance(closes: list[dict]) -> dict:
    """Métriques de performance d'une version (moyenne, médiane, win rate,
    pnl cumulé, drawdown de la courbe cumulée, durée moyenne)."""
    n = len(closes)
    out = {"n_closes": n, "win_rate": None, "expectancy_pct": None,
           "median_pct": None, "cumulative_pnl_pct": None,
           "max_drawdown_pct": None, "avg_duration_sec": None,
           "status": "INSUFFICIENT" if n < MIN_CLOSES_GATE else "OK"}
    if n == 0:
        return out
    pnls = [c["pnl_pct"] for c in closes]
    wins = sum(1 for p in pnls if p > 0)
    out["win_rate"] = round(wins / n, 4)
    out["expectancy_pct"] = round(sum(pnls) / n * 100.0, 4)
    out["cumulative_pnl_pct"] = round(sum(pnls) * 100.0, 4)
    # médiane
    sp = sorted(pnls)
    out["median_pct"] = round(
        (sp[n // 2] if n % 2 else (sp[n // 2 - 1] + sp[n // 2]) / 2) * 100.0, 4)
    # drawdown de la courbe cumulée (en % cumulé — approximation documentée)
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p * 100.0
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
    out["max_drawdown_pct"] = round(max_dd, 4)
    # durée moyenne
    durs = [c["duration_sec"] for c in closes if c.get("duration_sec")]
    if durs:
        out["avg_duration_sec"] = round(sum(durs) / len(durs), 1)
    return out


# --------------------------------------------------------------------------- #
# Gate PURE : la nouvelle version bat-elle l'ancienne ?
# --------------------------------------------------------------------------- #
def version_gate(new_perf: dict, old_perf: dict) -> dict:
    """
    (decision, reasons) — PURE et testable.
      PROMOTED     : les DEUX versions ont >= MIN_CLOSES_GATE clôtures ET
                     expectancy_new > expectancy_old ET drawdown non dégradé
                     (max_drawdown_new >= max_drawdown_old - tol).
      NOT_PROMOTED : la comparaison est possible mais la nouvelle version ne
                     bat pas l'ancienne (edge OU risque).
      INSUFFICIENT : échantillon insuffisant d'un côté — JAMAIS de promotion.
    """
    n_new = int(new_perf.get("n_closes", 0))
    n_old = int(old_perf.get("n_closes", 0))
    if n_new < MIN_CLOSES_GATE or n_old < MIN_CLOSES_GATE:
        return {
            "decision": "INSUFFICIENT",
            "reasons": [f"échantillon insuffisant (nouvelle {n_new}, "
                        f"ancienne {n_old} clôtures — minimum "
                        f"{MIN_CLOSES_GATE})"],
        }
    exp_new = new_perf.get("expectancy_pct")
    exp_old = old_perf.get("expectancy_pct")
    dd_new = new_perf.get("max_drawdown_pct")
    dd_old = old_perf.get("max_drawdown_pct")
    reasons = []
    edge_better = exp_new is not None and exp_old is not None \
        and exp_new > exp_old
    reasons.append(f"expectancy {exp_new} % vs {exp_old} % "
                   f"({'meilleure' if edge_better else 'pas meilleure'})")
    dd_ok = dd_new is not None and dd_old is not None \
        and dd_new >= dd_old - DD_DEGRADE_TOL * max(abs(dd_old), 1e-9)
    reasons.append(f"drawdown {dd_new} % vs {dd_old} % "
                   f"({'non dégradé' if dd_ok else 'dégradé'})")
    if edge_better and dd_ok:
        return {"decision": "PROMOTED", "reasons": reasons}
    return {"decision": "NOT_PROMOTED", "reasons": reasons}


# --------------------------------------------------------------------------- #
# Rapport complet (tableau des versions + verdicts)
# --------------------------------------------------------------------------- #
def version_benchmark_report(db, current_version: str = "") -> dict:
    """
    Vue d'ensemble : performances par version + gate version-vs-version
    (chaque version vs la précédente, ordre chronologique). La version
    courante est marquée. Honnête : sans clôtures -> tout INSUFFICIENT.
    """
    by_ver = _closes_per_version(db)
    # tri chronologique (première clôture de chaque version)
    versions = sorted(
        by_ver.keys(),
        key=lambda v: (by_ver[v][0]["ts"] if by_ver[v] else 0.0))
    perfs = {v: version_performance(closes) for v, closes in by_ver.items()}
    table = []
    prev = None
    for v in versions:
        row = {"version": v, "n_closes": perfs[v]["n_closes"],
               "expectancy_pct": perfs[v]["expectancy_pct"],
               "win_rate": perfs[v]["win_rate"],
               "max_drawdown_pct": perfs[v]["max_drawdown_pct"],
               "is_current": bool(current_version) and v == current_version,
               "gate_vs_previous": None}
        if prev is not None:
            row["gate_vs_previous"] = version_gate(perfs[v], perfs[prev])
        table.append(row)
        prev = v
    return {
        "versions": table,
        "n_versions": len(table),
        "min_closes_gate": MIN_CLOSES_GATE,
        "current_version": current_version or None,
        "note": ("Le système est son propre benchmark : chaque version doit "
                 "battre la précédente (expectancy > ET drawdown non dégradé, "
                 "≥ 30 clôtures de chaque côté) pour être promue. pnl_pct = "
                 "brut de frais d'exécution (même base pour toutes les "
                 "versions -> comparaison relative valide) ; le gate de "
                 "friction couvre les coûts séparément. Drawdown = courbe "
                 "cumulée des clôtures (approximation documentée)."),
    }
