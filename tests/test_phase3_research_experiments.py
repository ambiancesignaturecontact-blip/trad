"""
PHASE 3 Cycle 2 — tests du moteur d'expériences de recherche
(core/research_experiments.py).

Couvre : anti-look-ahead du backtest, parité du signal vectorisé avec la
production (MomentumStrategy), métriques round-trip, orchestration avec
enregistrement, et la règle de décision REJECT/KEEP.
"""
import numpy as np
import pandas as pd
import pytest


class FakeDB:
    """Mini DB : experiments + candles (mécanique d'orchestration)."""

    def __init__(self):
        self.experiments = []
        self._next = 1
        self._killed = set()

    def ensure_experiments_table(self):
        pass

    def add_experiment(self, hypothesis, status="PENDING", result=""):
        eid = self._next
        self._next += 1
        self.experiments.append({"id": eid, "hypothesis": hypothesis,
                                 "status": status, "result": result,
                                 "killed": 0})
        return eid

    def update_experiment(self, eid, fields):
        for e in self.experiments:
            if e["id"] == eid:
                e.update(fields)
                return True
        return False

    def get_kill_list(self, limit=100):
        return [e for e in self.experiments if e.get("killed")][:limit]

    def is_hypothesis_killed(self, hypothesis):
        return any(e["hypothesis"] == hypothesis and e.get("killed")
                   for e in self.experiments)

    def list_experiments(self, limit=100):
        return self.experiments[-limit:][::-1]

    def load_candles(self, symbol, limit=200):
        n = min(limit, 100)
        close = np.linspace(100.0, 110.0, n)
        idx = pd.date_range("2026-01-01", periods=n, freq="h")
        return pd.DataFrame({"open": close, "high": close + 1,
                             "low": close - 1, "close": close,
                             "volume": np.full(n, 1000.0)}, index=idx)


class TestNoLookAhead:
    def test_signal_perfect_on_same_bar_not_captured(self):
        """Un signal « parfait » (signe du rendement de la MÊME barre) ne doit
        PAS capturer ce rendement : la position est décalée d'une barre.
        Avec look-ahead, le cumul serait ≈ +56 % (somme |ret| − coûts) ;
        sans, il est dominé par les coûts : ≈ −35 %."""
        from core.research_experiments import backtest_signals
        rng = np.random.default_rng(42)
        ret = rng.normal(0, 0.002, 500)
        close = pd.Series((1 + ret).cumprod() * 100.0)
        sig = pd.Series(np.sign(ret), index=close.index)  # prédiction « parfaite »
        res = backtest_signals(close, sig)
        # seuil large : sépare nettement le cas sans look-ahead (négatif,
        # coûts dominants) du cas avec look-ahead (+56 %)
        assert res["cumulative_pnl_pct"] < 30.0


class TestSignalParity:
    def test_vectorized_matches_production_on_seeded_data(self):
        """La réplique vectorisée doit reproduire EXACTEMENT le signal de
        MomentumStrategy (production) sur la dernière barre, sur 10 seeds
        (attrape les divergences : RSI dépendant, loss=0, off-by-one ROC)."""
        from core.research_experiments import momentum_signal_series
        from strategies.momentum import MomentumStrategy
        rng = np.random.default_rng(7)
        for seed in range(10):
            ret = rng.normal(0, 0.015, 600)
            close = pd.Series((1 + ret).cumprod() * 100.0)
            vol = pd.Series(rng.integers(50, 2000, 600).astype(float))
            df = pd.DataFrame({"close": close.values, "volume": vol.values})
            prod_signal, _ = MomentumStrategy().generate_signal({"df": df})
            vec = momentum_signal_series(close, volume=vol)
            assert abs(float(prod_signal) - float(vec.iloc[-1])) < 1e-9, \
                f"seed {seed}: prod {prod_signal} != vec {vec.iloc[-1]}"


class TestRoundTrips:
    def test_round_trip_metrics_and_shift(self):
        """Cas déterministe : le trade voit la hausse APRÈS le signal (shift),
        le round-trip clôturé est compté avec son pnl net de coûts."""
        from core.research_experiments import backtest_signals
        close = pd.Series([100.0] * 3 + [110.0, 120.0] + [100.0] * 5)
        sig = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        res = backtest_signals(close, sig, cost_ar_pct=0.213)
        assert res["n_round_trips"] == 1        # entrée -> retour à zéro
        assert res["n_open_trades"] == 0
        # rendements capturés APRÈS le signal : entrée au close de la barre 2,
        # sortie au close de la barre 5 (chute incluse) — net des coûts AR
        expected = (0.10 + (120.0 / 110.0 - 1.0) + (100.0 / 120.0 - 1.0)) \
            * 100.0 - 0.213
        assert res["expectancy_rt_pct"] == pytest.approx(expected, abs=0.05)
        assert res["win_rate_rt"] == 1.0

    def test_open_trade_at_end_not_counted_as_closed(self):
        from core.research_experiments import backtest_signals
        close = pd.Series([100.0] * 10)
        sig = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        res = backtest_signals(close, sig)
        assert res["n_round_trips"] == 0        # jamais revenu à zéro
        assert res["n_open_trades"] == 1


class TestDecide:
    def test_reject_when_nothing_improves(self):
        from core.research_experiments import decide
        d, c = decide(-0.02, -0.03, -100.0, -120.0, 50)
        assert d == "REJECT"

    def test_keep_mixed(self):
        from core.research_experiments import decide
        d, c = decide(-0.03, -0.031, -200.0, -150.0, 50)  # OOS - / stress +
        assert d == "KEEP"
        assert "MIXTES" in c

    def test_keep_insufficient_round_trips(self):
        from core.research_experiments import decide
        d, c = decide(-0.01, -0.05, -100.0, -50.0, 3)
        assert d == "KEEP"
        assert "insuffisante" in c

    def test_keep_improves_both_but_never_promote(self):
        from core.research_experiments import decide
        d, c = decide(-0.02, -0.01, -100.0, -80.0, 50)
        assert d == "KEEP"
        assert "promotion automatique" in c


class TestOrchestration:
    def test_experiment_recorded_through_memory(self):
        """Pipeline complet sur le FakeDB : résultats OOS/stress enregistrés,
        décision KEEP (preuve insuffisante : 0 round-trip sur données
        linspace) — aucune promotion, aucun crash."""
        from core.research_experiments import run_experiment
        db = FakeDB()
        db.add_experiment("Hypothèse filtre vol Momentum", status="RESEARCH")
        r = run_experiment(db, 1)
        assert r["decision"] == "KEEP"
        assert r["recorded"] is True
        e = db.experiments[0]
        assert e["status"] == "PAPER"           # KEEP -> PAPER (jamais PROMOTE)
        assert e["conclusion"]
        assert "insuffisante" in e["conclusion"]
        assert "oos_results" in e and e["oos_results"]
        assert "stress_results" in e and e["stress_results"]
