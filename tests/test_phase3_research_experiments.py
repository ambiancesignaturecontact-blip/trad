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


# --------------------------------------------------------------------------- #
# PHASE 3 Cycle 4 — horizon long (résampling 4h/1d, Expérience #3)
# --------------------------------------------------------------------------- #
class TestResampleAndTimeframe:
    def test_resample_4h_close_volume(self):
        """Résampling 4h : close = dernière close, high/low = extrema,
        volume = somme (règle propre, sans look-ahead)."""
        from core.research_experiments import resample_candles
        idx = pd.date_range("2026-01-01", periods=8, freq="h")
        df = pd.DataFrame({
            "open": [1, 2, 3, 4, 5, 6, 7, 8],
            "high": [2, 3, 4, 5, 6, 7, 8, 9],
            "low": [0, 1, 2, 3, 4, 5, 6, 7],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
            "volume": [10, 10, 10, 10, 10, 10, 10, 10],
        }, index=idx)
        r = resample_candles(df, "4h")
        assert len(r) == 2
        assert r["close"].iloc[0] == 4.5          # dernière close du 1er bloc
        assert r["high"].iloc[0] == 5.0
        assert r["low"].iloc[0] == 0.0
        assert r["volume"].iloc[0] == 40.0        # somme
        # 1h = inchangé
        assert len(resample_candles(df, "1h")) == 8

    def test_experiment_4h_parity_not_checked(self):
        """En 4h, la parité de signal n'est PAS vérifiée (variante de
        recherche, documentée parity_checked=False) — le pipeline tourne."""
        from core.research_experiments import run_experiment
        db = FakeDB()
        db.add_experiment("Hypothèse horizon 4h", status="RESEARCH")
        r = run_experiment(db, 1, timeframe="4h")
        assert r["timeframe"] == "4h"
        assert r["recorded"] is True
        for d in r.get("per_symbol", {}).values():
            assert d["parity_checked"] is False

    def test_experiment_1h_parity_checked(self):
        from core.research_experiments import run_experiment
        db = FakeDB()
        db.add_experiment("Hypothèse 1h", status="RESEARCH")
        r = run_experiment(db, 1, timeframe="1h")
        assert r["timeframe"] == "1h"
        for d in r.get("per_symbol", {}).values():
            assert d["parity_checked"] is True


# --------------------------------------------------------------------------- #
# PHASE 3 Cycle 6 — contrarian post-extrême (Exp#5, contre-hypothèse momentum)
# --------------------------------------------------------------------------- #
class TestContrarianSignal:
    def test_no_lookahead_shift(self):
        """Le contrarian capte le rendement APRÈS le mouvement extrême, pas
        celui de la barre extrême elle-même (shift d'1 barre)."""
        from core.research_experiments import backtest_signals, contrarian_signal_series
        # mouvement extrême à la barre 5 (+10 %) ; le signal doit être -1 APRÈS
        close = pd.Series([100.0] * 5 + [110.0] + [110.0] * 5)
        sig = contrarian_signal_series(close, ret_quantile=0.90)
        # le signal est généré SUR la barre extrême (position opposée)…
        assert sig.iloc[5] == -1.0
        # …mais le backtest décale d'1 barre : la position s'applique à la
        # barre 6 — le +10 % de la barre 5 n'est JAMAIS capté
        res = backtest_signals(close, sig)
        assert res["cumulative_pnl_pct"] > -0.5   # aucun gain du +10 % capté

    def test_signal_family_wired(self):
        from core.research_experiments import run_experiment
        db = FakeDB()
        db.add_experiment("Contrarian", status="RESEARCH")
        r = run_experiment(db, 1, timeframe="1h", signal_family="contrarian")
        assert r["signal_family"] == "contrarian"
        assert r["recorded"] is True
        for d in r.get("per_symbol", {}).values():
            assert d["parity_checked"] is False  # pas de production équivalente


# --------------------------------------------------------------------------- #
# PHASE 3 Cycle 7 — calibrage TRAIN-only + signal flow (micro-structure)
# --------------------------------------------------------------------------- #
class TestCalibrageTrainOnly:
    def test_contrarian_threshold_passed_not_recalibrated(self):
        """Le seuil contrarian est FOURNI (calibré train) — pas de recalcul
        sur la série entière dans la fonction de signal."""
        from core.research_experiments import contrarian_signal_series
        close = pd.Series([100.0] * 50 + [110.0] + [100.0] * 20)
        # seuil très bas : les 2 rendements non nuls (+10 %, -9,09 %) sont
        # extrêmes -> signal opposé sur ces 2 barres exactement
        sig = contrarian_signal_series(close, threshold=0.0)
        assert (sig != 0).sum() == 2
        assert sig.iloc[50] == -1.0 and sig.iloc[51] == 1.0
        # seuil très haut : aucun signal
        sig2 = contrarian_signal_series(close, threshold=1e9)
        assert (sig2 != 0).sum() == 0

    def test_flow_signal_causal_zscore(self):
        """Le z-score du volume directionnel est causal (rolling) ; un pic de
        volume directionnel génère un signal de même signe."""
        from core.research_experiments import flow_signal_series
        idx = pd.date_range("2026-01-01", periods=120, freq="h")
        close = pd.Series(100.0, index=idx)
        # petite tendance haussière -> vol_delta positif en moyenne
        close = close * (1.0 + np.linspace(0, 0.05, 120))
        vol = pd.Series(1000.0, index=idx)
        # énorme pic de volume haussier à la barre 80
        vol.iloc[80] = 100_000.0
        sig = flow_signal_series(close, vol, z_window=24, threshold=2.0)
        assert sig.iloc[80] == 1.0          # flux haussier extrême -> +1
        assert sig.iloc[70] == 0.0          # pas de signal avant le pic

    def test_calibrate_threshold_train_only(self):
        """calibrate_threshold_train : seuil = percentile 90 des |z| du TRAIN
        (split), la valeur est finie et raisonnable."""
        from core.research_experiments import calibrate_threshold_train
        rng = np.random.default_rng(11)
        close = pd.Series((1 + rng.normal(0, 0.01, 500)).cumprod() * 100)
        vol = pd.Series(rng.integers(100, 3000, 500).astype(float))
        thr = calibrate_threshold_train(close, vol, split=350, quantile=0.90)
        assert 0.5 < thr < 5.0

    def test_flow_family_wired_and_no_parity(self):
        from core.research_experiments import run_experiment
        db = FakeDB()
        db.add_experiment("Flow micro-structure", status="RESEARCH")
        r = run_experiment(db, 1, timeframe="1h", signal_family="flow")
        assert r["signal_family"] == "flow"
        assert r["recorded"] is True
        for d in r.get("per_symbol", {}).values():
            assert d["parity_checked"] is False


# --------------------------------------------------------------------------- #
# PHASE 3 Cycle 8 — momentum paramétrable, symbols restreints, filtre vol seul
# --------------------------------------------------------------------------- #
class TestCycle8Extensions:
    def test_momentum_kwargs_passed(self):
        """Les paramètres momentum (roc long) sont transmis au générateur."""
        from core.research_experiments import run_experiment
        db = FakeDB()
        db.add_experiment("Momentum lent", status="RESEARCH")
        r = run_experiment(db, 1, timeframe="4h", signal_family="momentum",
                           momentum_kwargs={"roc_period": 48,
                                            "min_momentum": 0.02})
        assert r["recorded"] is True

    def test_symbols_restricted(self):
        """symbols=("AAPL",) ne traite QUE AAPL (per_symbol)."""
        from core.research_experiments import run_experiment
        db = FakeDB500()
        db.add_experiment("Actions only", status="RESEARCH")
        r = run_experiment(db, 1, symbols=("AAPL",), timeframe="1h")
        assert set(r["per_symbol"].keys()) == {"AAPL"}

    def test_symbols_none_default(self):
        from core.research_experiments import DEFAULT_SYMBOLS, run_experiment
        db = FakeDB500()
        db.add_experiment("Défaut", status="RESEARCH")
        r = run_experiment(db, 1, symbols=None)
        assert set(r["per_symbol"].keys()) <= set(DEFAULT_SYMBOLS)
        assert len(r["per_symbol"]) > 0

    def test_vol_filter_reject_when_no_protection(self):
        """Filtre vol seul sur données calmes (drawdown hors HIGH_VOL,
        stress = rallye) : REJECT — aucune protection mesurée."""
        from core.research_experiments import run_vol_filter_experiment
        db = FakeDB500()
        db.add_experiment("Filtre vol seul", status="RESEARCH")
        r = run_vol_filter_experiment(db, 1, cost_ar_pct=0.213)
        assert r["decision"] in ("KEEP", "REJECT")
        assert r["recorded"] is True
        assert "signal_family" in r

    def test_vol_filter_stress_reduces_pnl_when_rally(self):
        """Vérification mécanique DÉTERMINISTE : en période de vol haute
        HAUSSIÈRE pure, le filtre ×0,5 réduit le PnL (il coupe le rallye)."""
        from core.research_experiments import backtest_signals, high_vol_mask, volatility_ewma
        ret = np.zeros(200)
        ret[100:130] = 0.02                      # rallye pur de 30 barres
        close = pd.Series((1 + ret).cumprod() * 100.0)
        vol = volatility_ewma(close)
        thr = vol.quantile(0.8)
        mask = high_vol_mask(vol, thr)
        scale = pd.Series(1.0, index=close.index)
        scale[mask] = 0.5
        ones = pd.Series(1.0, index=close.index)
        base = backtest_signals(close, ones)
        treat = backtest_signals(close, ones, position_scale=scale)
        assert int(mask.sum()) > 10              # le rallye est bien masqué
        assert treat["cumulative_pnl_pct"] < base["cumulative_pnl_pct"]


class FakeDB500(FakeDB):
    """FakeDB avec >= 400 barres par symbole (le pipeline exige un minimum
    d'échantillon ; FakeDB n'en fournit que 100 -> per_symbol vide)."""

    def load_candles(self, symbol, limit=200):
        n = min(limit, 600)
        close = np.linspace(100.0, 110.0, n)
        idx = pd.date_range("2026-01-01", periods=n, freq="h")
        return pd.DataFrame({"open": close, "high": close + 1,
                             "low": close - 1, "close": close,
                             "volume": np.full(n, 1000.0)}, index=idx)
