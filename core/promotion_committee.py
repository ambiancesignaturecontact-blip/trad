"""
PROMOTION COMMITTEE (PHASE 4 — P4-D, axe 6 du mandat).

« Le Promotion Committee est composé de règles quantitatives INDÉPENDANTES
qui peuvent dire : REJECT / KEEP / PROMOTE. »

Chaque règle est une fonction PURE (testable, réversible) qui vote sur un
aspect DIFFÉRENT d'une expérience :
  1. rule_sample    : échantillon OOS (n_round_trips >= 30 pour promouvoir)
  2. rule_edge      : expectancy pondérée par RT > 0 APRÈS coûts réels
  3. rule_stress    : le traitement ne dégrade pas le PnL de stress
  4. rule_drawdown  : le drawdown max du traitement n'est pas plus profond
  5. rule_friction  : le coût AR attendu passe sous le gate de friction

Vote agrégé (conservateur) :
  - REJECT  : VETO — au moins une règle vote REJECT (hypothèse invalidée
              ou fragile sous stress) ;
  - PROMOTE : toutes les règles votent PROMOTE (candidat à la phase
              suivante : PAPER / LIMITED — JAMAIS production directe) ;
  - KEEP    : tout le reste (preuve incomplète, amélioration partielle).

Règles absolues :
  - Le committee est un AVIS enregistré (params JSON de l'expérience) : il ne
    modifie JAMAIS la production ni le statut de l'expérience automatiquement.
  - PROMOTE != production : le candidat doit ensuite passer le shadow/paper.
  - Sans métrique mesurée (None) -> la règle vote KEEP (pas de preuve, pas
    de promotion) — jamais de chiffre inventé.
"""
import json
import logging

logger = logging.getLogger("InstitutionalTradingBot")

MIN_TRADES_PROMOTE = 30           # échantillon OOS minimal pour promouvoir
EDGE_DEGRADE_TOL = 0.20           # tolérance dégradation edge (20 % relatif)
STRESS_DEGRADE_TOL = 0.10         # tolérance dégradation stress (10 % relatif)
DD_DEGRADE_TOL = 0.20             # tolérance drawdown (20 % relatif)
FRICTION_THRESHOLD_PCT = 1.0      # coût AR max admissible (gate friction)

VOTE_REJECT = "REJECT"
VOTE_KEEP = "KEEP"
VOTE_PROMOTE = "PROMOTE"


# --------------------------------------------------------------------------- #
# Règles individuelles (PURES)
# --------------------------------------------------------------------------- #
def rule_sample(n_round_trips) -> tuple:
    """Échantillon : >= 30 RT OOS requis pour promouvoir ; sinon KEEP
    (preuve insuffisante — jamais REJECT : ce n'est pas une invalidation)."""
    n = n_round_trips if n_round_trips is not None else 0
    if n >= MIN_TRADES_PROMOTE:
        return VOTE_PROMOTE, f"{n} RT OOS >= {MIN_TRADES_PROMOTE}"
    return VOTE_KEEP, (f"{n} RT OOS < {MIN_TRADES_PROMOTE} — "
                       f"échantillon insuffisant pour promouvoir")


def rule_edge(exp_treatment, exp_baseline) -> tuple:
    """Edge : expectancy pondérée par RT du traitement > 0 après coûts.
    Dégradation réelle = traitement PLUS NÉGATIF que baseline de plus de
    20 % relatif -> REJECT ; amélioration ou neutralité -> KEEP/PROMOTE
    selon le signe. (Formule correcte pour les expectancies négatives :
    on compare la distance relative à baseline, pas un produit signé.)"""
    if exp_treatment is None:
        return VOTE_KEEP, "aucune expectancy mesurée — pas de preuve d'edge"
    if exp_treatment > 0.0:
        return VOTE_PROMOTE, f"expectancy {exp_treatment} %/RT > 0 après coûts"
    if exp_baseline is not None:
        tol = EDGE_DEGRADE_TOL * max(abs(exp_baseline), 1e-9)
        if exp_treatment < exp_baseline - tol:
            return VOTE_REJECT, (f"expectancy {exp_treatment} %/RT négative "
                                 f"ET dégradée de > {EDGE_DEGRADE_TOL*100:.0f} % "
                                 f"vs baseline {exp_baseline} %/RT")
    return VOTE_KEEP, (f"expectancy {exp_treatment} %/RT <= 0 (pas d'edge "
                       f"net) mais pas de dégradation marquée")


def rule_stress(stress_treatment, stress_baseline) -> tuple:
    """Stress : PnL cumulé de stress du traitement >= baseline (pas pire).
    Dégradation de plus de 10 % relatif -> REJECT (fragile sous stress)."""
    if stress_treatment is None or stress_baseline is None:
        return VOTE_KEEP, "aucun stress mesuré"
    if stress_treatment >= stress_baseline:
        return VOTE_PROMOTE, (f"stress {stress_treatment} % >= baseline "
                              f"{stress_baseline} % (non dégradé)")
    tol = STRESS_DEGRADE_TOL * abs(stress_baseline)
    if stress_treatment < stress_baseline - tol:
        return VOTE_REJECT, (f"stress dégradé de > {STRESS_DEGRADE_TOL*100:.0f} % "
                             f"({stress_treatment} vs {stress_baseline} %)")
    return VOTE_KEEP, f"stress légèrement dégradé ({stress_treatment} % vs {stress_baseline} %)"


def rule_drawdown(dd_treatment, dd_baseline) -> tuple:
    """Drawdown : traitement pas plus profond que 20 % relatif vs baseline.
    (les drawdowns sont négatifs : plus grand = moins profond = mieux)."""
    if dd_treatment is None or dd_baseline is None:
        return VOTE_KEEP, "aucun drawdown mesuré"
    if dd_treatment >= dd_baseline:
        return VOTE_PROMOTE, (f"drawdown {dd_treatment} % >= baseline "
                              f"{dd_baseline} % (non dégradé)")
    if dd_treatment < dd_baseline * (1.0 + DD_DEGRADE_TOL):
        return VOTE_REJECT, (f"drawdown plus profond de > {DD_DEGRADE_TOL*100:.0f} % "
                             f"({dd_treatment} vs {dd_baseline} %)")
    return VOTE_KEEP, f"drawdown légèrement plus profond ({dd_treatment} vs {dd_baseline})"


def rule_friction(max_cost_ar_pct, threshold_pct=FRICTION_THRESHOLD_PCT) -> tuple:
    """Friction : si le coût AR du pire symbole dépasse le seuil du gate,
    la production BLOQUERAIT le trade -> KEEP (contrainte d'exécution, pas
    une invalidation de l'edge)."""
    if max_cost_ar_pct is None:
        return VOTE_KEEP, "friction non mesurée"
    if max_cost_ar_pct <= threshold_pct:
        return VOTE_PROMOTE, (f"coût AR {max_cost_ar_pct} % <= seuil "
                              f"{threshold_pct} % (passerait le gate)")
    return VOTE_KEEP, (f"coût AR {max_cost_ar_pct} % > seuil {threshold_pct} % "
                       f"— le gate friction bloquerait en production")


# --------------------------------------------------------------------------- #
# Vote agrégé (fonction PURE)
# --------------------------------------------------------------------------- #
def committee_vote(oos: dict, stress: dict, max_cost_ar_pct=None) -> dict:
    """
    Applique les 5 règles indépendantes à une expérience (format identique à
    celui retourné par run_experiment / run_vol_filter_experiment) et agrège.
    Retourne {vote, votes (par règle), note}.
    """
    base = (oos or {}).get("baseline") or {}
    treat = (oos or {}).get("treatment") or {}
    st_base = (stress or {}).get("baseline") or {}
    st_treat = (stress or {}).get("treatment") or {}

    votes = {
        "sample": rule_sample(treat.get("n_round_trips")),
        "edge": rule_edge(treat.get("expectancy_pct"),
                          base.get("expectancy_pct")),
        "stress": rule_stress(st_treat.get("cumulative_pnl_pct"),
                              st_base.get("cumulative_pnl_pct")),
        "drawdown": rule_drawdown(treat.get("max_drawdown_pct"),
                                  base.get("max_drawdown_pct")),
        "friction": rule_friction(max_cost_ar_pct),
    }
    statuses = [v[0] for v in votes.values()]
    if VOTE_REJECT in statuses:
        vote = VOTE_REJECT
    elif all(s == VOTE_PROMOTE for s in statuses):
        vote = VOTE_PROMOTE
    else:
        vote = VOTE_KEEP

    # justifications
    reasons = {name: {"vote": v[0], "reason": v[1]}
               for name, v in votes.items()}
    note = {
        "REJECT": "VETO : au moins une règle invalide l'hypothèse — archivée "
                  "(kill list) ou fragile sous stress.",
        "PROMOTE": "TOUTES les règles votent PROMOTE : candidat à la phase "
                   "suivante (PAPER/LIMITED) — JAMAIS production directe.",
        "KEEP": "Preuve incomplète ou amélioration partielle : l'hypothèse "
                "est conservée, ré-évaluation après plus de données ou "
                "affinement.",
    }[vote]
    return {"vote": vote, "votes": reasons, "note": note}


# --------------------------------------------------------------------------- #
# Application sur une expérience enregistrée (DB)
# --------------------------------------------------------------------------- #
def evaluate_experiment(db, experiment_id: int,
                        friction_threshold_pct: float = FRICTION_THRESHOLD_PCT
                        ) -> dict:
    """Charge l'expérience (result JSON), applique le committee, enregistre
    le vote dans `params` (traçabilité, JAMAIS de changement de statut auto)
    et retourne {experiment_id, hypothesis, vote, votes, note}."""
    out = {"experiment_id": experiment_id, "vote": None, "votes": {},
           "note": None, "status": "ERROR"}
    try:
        if db is None or not hasattr(db, "get_connection"):
            out["note"] = "db indisponible"
            return out
        with db.get_connection() as conn:
            cur = conn.cursor()
            ph = "%s" if getattr(db, "is_postgres", False) else "?"
            cur.execute(
                f"SELECT hypothesis, result, oos_results, stress_results, "
                f"status, killed FROM experiments WHERE id = {ph}",
                (int(experiment_id),))
            row = cur.fetchone()
        if row is None:
            out["note"] = f"expérience #{experiment_id} introuvable"
            return out
        hypothesis = str(row[0])
        result_raw = str(row[2] or row[1] or "{}")  # oos_results d'abord
        stress_raw = str(row[3] or "{}")            # stress_results
        try:
            oos = json.loads(result_raw) if result_raw else {}
            stress = json.loads(stress_raw) if stress_raw else {}
        except Exception:
            oos, stress = {}, {}
        # coût AR max : cherché dans result (cost_ar_pct) s'il existe
        max_cost = None
        try:
            res = json.loads(str(row[1] or "{}"))
            max_cost = res.get("cost_ar_pct")
        except Exception:
            pass
        verdict = committee_vote(oos, stress, max_cost_ar_pct=max_cost)
        out.update({"hypothesis": hypothesis, "status": str(row[4]),
                    "killed": bool(row[5]), **verdict})
        # enregistrement du vote (params JSON) — jamais de statut auto
        try:
            params = json.dumps({"promotion_committee": verdict},
                                default=str)
            db.update_experiment(int(experiment_id), {"params": params})
        except Exception as e:
            logger.debug(f"committee params save failed: {e}")
    except Exception as e:
        out["note"] = f"indisponible ({e})"
        logger.debug(f"evaluate_experiment failed: {e}")
    return out


def committee_overview(db, limit: int = 8) -> dict:
    """Réévalue les dernières expériences (hors killed) — vue Daily Report."""
    out = {"votes": [], "n_reject": 0, "n_keep": 0, "n_promote": 0,
           "note": "Le committee est un AVIS multi-règles indépendantes — "
                   "il ne modifie jamais la production."}
    try:
        if db is None or not hasattr(db, "list_experiments"):
            return out
        exps = db.list_experiments(limit=limit)
        for e in exps:
            eid = int(e.get("id", 0))
            if not eid or e.get("killed"):
                continue
            try:
                v = evaluate_experiment(db, eid)
                out["votes"].append({
                    "id": eid,
                    "hypothesis": (e.get("hypothesis") or "")[:70],
                    "vote": v.get("vote"),
                })
                out[f"n_{str(v.get('vote', '')).lower()}"] = \
                    out.get(f"n_{str(v.get('vote', '')).lower()}", 0) + 1
            except Exception:
                continue
    except Exception as e:
        out["note"] = f"indisponible ({e})"
    return out
