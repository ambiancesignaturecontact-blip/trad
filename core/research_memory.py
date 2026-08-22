"""
RESEARCH MEMORY & KILL LIST (PHASE 3 — §5/§6/§7).

Couche de recherche quantitative : transforme une OBSERVATION en HYPOTHÈSE
falsifiable, enregistre chaque EXPÉRIENCE (modification, données, période,
régimes, paramètres, résultats OOS/stress, conclusion), et maintient une
KILL LIST : une hypothèse invalidée ne doit JAMAIS être re-proposée
automatiquement sans nouvelle preuve (is_hypothesis_killed).

Pipeline (mandat) :
    OBSERVATION -> HYPOTHÈSE -> EXPÉRIENCE -> VALIDATION (OOS/stress)
        -> DÉCISION REJECT / KEEP / PROMOTE

Principes :
  1. Jamais bloquant ; persistance via la table `experiments` (db_manager).
  2. Le Research Brain PROPOSE — il ne modifie jamais la production.
  3. Une expérience n'est promue que par un processus séparé, avec preuves.
  4. Aucun flag de mode (DÉMO == RÉAL).
"""
import json
import logging
import time

logger = logging.getLogger("InstitutionalTradingBot")

# Statuts (mandat §4/§6)
STATUSES = ("RESEARCH", "VALIDATION", "PAPER", "CANDIDATE",
            "PRODUCTION", "DEGRADED", "DISABLED", "ROLLBACK", "REJECTED")

# Actions possibles après validation
ACTIONS = ("REJECT", "KEEP", "PROMOTE")


class ResearchMemory:
    """Registre des hypothèses et expériences + kill list."""

    def __init__(self, db):
        self.db = db

    # ------------------------------------------------------------------ #
    def record_hypothesis(self, hypothesis: str, observation: str,
                          modification: str = "", dataset: str = "",
                          period: str = "", regimes: str = "") -> int:
        """
        Enregistre une hypothèse FALSIFIABLE (issue d'une observation).
        Refuse silencieusement si l'hypothèse a déjà été tuée (kill list).
        Retourne l'id ou 0 si tuée / erreur.
        """
        try:
            if self.db.is_hypothesis_killed(hypothesis):
                logger.info(f"🔒 RESEARCH: hypothèse déjà tuée — refusée : {hypothesis[:60]}")
                return 0
            eid = self.db.add_experiment(hypothesis, status="RESEARCH",
                                         result=json.dumps({"observation": observation}))
            self.db.update_experiment(eid, {
                "modification": modification, "dataset": dataset,
                "period": period, "regimes": regimes,
            })
            logger.info(f"🔬 RESEARCH: hypothèse enregistrée #{eid} : {hypothesis[:80]}")
            return int(eid)
        except Exception as e:
            logger.warning(f"record_hypothesis failed ({e})")
            return 0

    # ------------------------------------------------------------------ #
    def record_experiment_result(self, experiment_id: int, results: dict,
                                 conclusion: str, action: str) -> bool:
        """
        Complète une expérience avec ses résultats (backtest, walk-forward,
        OOS, Monte Carlo, stress, adversariaux) + conclusion + action.
        """
        try:
            assert action in ACTIONS, f"action inconnue : {action}"
            fields = {
                "status": "CANDIDATE" if action == "PROMOTE"
                          else ("REJECTED" if action == "REJECT" else "PAPER"),
                "result": json.dumps({k: v for k, v in results.items()
                                      if v is not None}, default=str),
                "oos_results": json.dumps(results.get("oos", {}), default=str),
                "stress_results": json.dumps(results.get("stress", {}), default=str),
                "conclusion": conclusion,
            }
            if action == "REJECT":
                fields["reject_reason"] = conclusion
            return bool(self.db.update_experiment(experiment_id, fields))
        except Exception as e:
            logger.warning(f"record_experiment_result failed ({e})")
            return False

    # ------------------------------------------------------------------ #
    def kill(self, experiment_id: int, reason: str) -> bool:
        """Ajoute une expérience à la KILL LIST (hypothèse invalidée)."""
        try:
            return bool(self.db.update_experiment(experiment_id, {
                "killed": True, "reject_reason": reason, "status": "REJECTED"}))
        except Exception as e:
            logger.warning(f"kill failed ({e})")
            return False

    def kill_list(self, limit: int = 100) -> list:
        return self.db.get_kill_list(limit=limit)

    def is_killed(self, hypothesis: str) -> bool:
        if self.db is None:
            return False
        try:
            return self.db.is_hypothesis_killed(hypothesis)
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    def report(self) -> dict:
        """État de la recherche : expériences récentes + kill list."""
        try:
            experiments = self.db.list_experiments(limit=20)
        except Exception:
            experiments = []
        kill_list = self.kill_list(limit=20)
        return {
            "recent_experiments": experiments,
            "kill_list": kill_list,
            "n_killed": len(kill_list),
            "n_recent": len(experiments),
            "note": "Le Research Brain propose ; seul le processus de validation "
                    "(OOS/stress/paper) décide REJECT/KEEP/PROMOTE. Une hypothèse "
                    "tuée n'est jamais re-proposée automatiquement.",
            "ts": time.time(),
        }
