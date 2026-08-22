"""
PHASE 4 — P4-D : tests du Promotion Committee (core/promotion_committee.py).

Couvre : règles pures (sample, edge, stress, drawdown, friction), vote agrégé
(veto, PROMOTE tout-ou-rien, KEEP), évaluation sur expérience DB (params
enregistrés, statut jamais modifié), overview, expositions.
"""
import json


class MiniDB:
    is_postgres = False

    def __init__(self, experiments=None):
        import sqlite3
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute(
            "CREATE TABLE experiments (id INTEGER PRIMARY KEY, hypothesis "
            "TEXT, status TEXT, result TEXT, oos_results TEXT, "
            "stress_results TEXT, params TEXT, killed INTEGER)")
        for e in experiments or []:
            self._conn.execute(
                "INSERT INTO experiments (id, hypothesis, status, result, "
                "oos_results, stress_results, params, killed) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (e.get("id", 1), e.get("hypothesis", "H"),
                 e.get("status", "PAPER"), e.get("result", ""),
                 e.get("oos_results", ""), e.get("stress_results", ""),
                 e.get("params", ""), int(e.get("killed", 0))))

    def get_connection(self):
        return self._conn

    def update_experiment(self, eid, fields):
        for k, v in fields.items():
            self._conn.execute(
                f"UPDATE experiments SET {k} = ? WHERE id = ?", (v, eid))
        return True

    def list_experiments(self, limit=100):
        cur = self._conn.execute(
            "SELECT id, hypothesis, status, killed FROM experiments "
            "ORDER BY id DESC LIMIT ?", (limit,))
        return [{"id": r[0], "hypothesis": r[1], "status": r[2],
                 "killed": r[3]} for r in cur.fetchall()]


def _exp_json(exp_t=None, exp_b=None, n_rt=40, stress_t=None, stress_b=None,
              dd_t=None, dd_b=None, cost=0.213):
    oos = {"baseline": {"expectancy_pct": exp_b, "n_round_trips": n_rt,
                        "max_drawdown_pct": dd_b},
           "treatment": {"expectancy_pct": exp_t, "n_round_trips": n_rt,
                         "max_drawdown_pct": dd_t}}
    stress = {"baseline": {"cumulative_pnl_pct": stress_b},
              "treatment": {"cumulative_pnl_pct": stress_t}}
    return json.dumps(oos), json.dumps(stress), cost


class TestRules:
    def test_sample(self):
        from core.promotion_committee import rule_sample
        assert rule_sample(30)[0] == "PROMOTE"
        assert rule_sample(29)[0] == "KEEP"
        assert rule_sample(0)[0] == "KEEP"
        assert rule_sample(None)[0] == "KEEP"

    def test_edge_positive_promotes(self):
        from core.promotion_committee import rule_edge
        assert rule_edge(0.05, -0.1)[0] == "PROMOTE"

    def test_edge_negative_not_degraded_keeps(self):
        from core.promotion_committee import rule_edge
        # traitement moins négatif que baseline (amélioration) -> KEEP
        assert rule_edge(-0.36, -0.40)[0] == "KEEP"
        # dégradation légère (< 20 %) -> KEEP
        assert rule_edge(-0.083, -0.0813)[0] == "KEEP"

    def test_edge_degraded_rejects(self):
        from core.promotion_committee import rule_edge
        # dégradation > 20 % relative (0.10 vs 0.05) -> REJECT
        assert rule_edge(-0.10, -0.05)[0] == "REJECT"

    def test_edge_none_keeps(self):
        from core.promotion_committee import rule_edge
        assert rule_edge(None, -0.1)[0] == "KEEP"

    def test_stress(self):
        from core.promotion_committee import rule_stress
        assert rule_stress(-100.0, -120.0)[0] == "PROMOTE"  # amélioré
        assert rule_stress(-110.0, -100.0)[0] == "KEEP"     # léger
        assert rule_stress(-130.0, -100.0)[0] == "REJECT"   # > 10 %

    def test_drawdown(self):
        from core.promotion_committee import rule_drawdown
        assert rule_drawdown(-10.0, -15.0)[0] == "PROMOTE"  # moins profond
        assert rule_drawdown(-16.0, -15.0)[0] == "KEEP"     # léger
        assert rule_drawdown(-20.0, -15.0)[0] == "REJECT"   # > 20 %

    def test_friction(self):
        from core.promotion_committee import rule_friction
        assert rule_friction(0.34, 1.0)[0] == "PROMOTE"
        assert rule_friction(1.7, 1.0)[0] == "KEEP"
        assert rule_friction(None, 1.0)[0] == "KEEP"


class TestVote:
    def _sample_vote(self):
        from core.promotion_committee import committee_vote
        oos, stress, cost = _exp_json(exp_t=0.05, exp_b=0.02, n_rt=40,
                                      stress_t=-50.0, stress_b=-60.0,
                                      dd_t=-8.0, dd_b=-10.0, cost=0.34)
        return committee_vote(json.loads(oos), json.loads(stress),
                              max_cost_ar_pct=cost)

    def test_promote_when_all_rules_pass(self):
        v = self._sample_vote()
        assert v["vote"] == "PROMOTE"
        assert all(r["vote"] == "PROMOTE" for r in v["votes"].values())

    def test_veto_reject(self):
        from core.promotion_committee import committee_vote
        oos, stress, cost = _exp_json(exp_t=-0.10, exp_b=-0.05, n_rt=40,
                                      stress_t=-50.0, stress_b=-60.0,
                                      cost=0.34)
        v = committee_vote(json.loads(oos), json.loads(stress),
                           max_cost_ar_pct=cost)
        assert v["vote"] == "REJECT"
        assert v["votes"]["edge"]["vote"] == "REJECT"

    def test_keep_when_negative_edge_not_degraded(self):
        from core.promotion_committee import committee_vote
        oos, stress, cost = _exp_json(exp_t=-0.083, exp_b=-0.0813, n_rt=100,
                                      stress_t=-50.0, stress_b=-60.0,
                                      cost=0.213)
        v = committee_vote(json.loads(oos), json.loads(stress),
                           max_cost_ar_pct=cost)
        assert v["vote"] == "KEEP"


class TestEvaluation:
    def test_evaluate_records_params_but_not_status(self):
        from core.promotion_committee import evaluate_experiment
        oos, stress, cost = _exp_json(exp_t=0.05, exp_b=0.02, n_rt=40,
                                      stress_t=-50.0, stress_b=-60.0,
                                      dd_t=-8.0, dd_b=-10.0, cost=0.213)
        db = MiniDB([{"id": 1, "hypothesis": "H1", "status": "PAPER",
                      "oos_results": oos, "stress_results": stress,
                      "result": json.dumps({"cost_ar_pct": cost})}])
        v = evaluate_experiment(db, 1)
        assert v["vote"] == "PROMOTE"
        # params enregistré, statut INCHANGÉ (jamais automatique)
        cur = db._conn.execute(
            "SELECT params, status FROM experiments WHERE id = 1")
        params, status = cur.fetchone()
        assert status == "PAPER"
        assert "promotion_committee" in params

    def test_unknown_experiment(self):
        from core.promotion_committee import evaluate_experiment
        db = MiniDB()
        v = evaluate_experiment(db, 99)
        assert v["note"] is not None and v["vote"] is None

    def test_overview_counts(self):
        from core.promotion_committee import committee_overview
        oos1, stress1, c1 = _exp_json(exp_t=0.05, exp_b=0.02, n_rt=40,
                                      stress_t=-50.0, stress_b=-60.0,
                                      dd_t=-8.0, dd_b=-10.0, cost=0.213)
        oos2, stress2, c2 = _exp_json(exp_t=-0.10, exp_b=-0.05, n_rt=40,
                                      stress_t=-50.0, stress_b=-60.0,
                                      cost=0.213)
        db = MiniDB([
            {"id": 1, "hypothesis": "A", "status": "PAPER",
             "oos_results": oos1, "stress_results": stress1,
             "result": json.dumps({"cost_ar_pct": c1})},
            {"id": 2, "hypothesis": "B", "status": "PAPER",
             "oos_results": oos2, "stress_results": stress2,
             "result": json.dumps({"cost_ar_pct": c2})},
            {"id": 3, "hypothesis": "C", "status": "REJECTED", "killed": 1},
        ])
        ov = committee_overview(db, limit=10)
        assert ov["n_promote"] == 1 and ov["n_reject"] == 1
        assert ov["n_keep"] == 0
        # l'expérience tuée n'est pas re-votée
        assert all(v["id"] != 3 for v in ov["votes"])
