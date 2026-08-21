"""
LOT D (F4 — online learning) : détection de drift par PSI (Population
Stability Index) sur les features clés + accélération de l'oubli du bandit.

Principes vérifiés :
  1. PSI = 0 sur distributions identiques ; PSI croît avec le décalage.
  2. Seuils standards : < 0.10 STABLE, 0.10-0.25 MODERATE, > 0.25 SEVERE.
  3. Drift sévère -> decay du bandit ACCÉLÉRÉ (jamais < 0.85 : un oubli
     total = bandit vierge qui repart à l'exploration pure).
  4. Pas de signal fabriqué : échantillon insuffisant -> STABLE (aucun
     drift annoncé sur du vide).
  5. DÉMO == RÉAL : aucun paramètre de mode dans le module.
"""
import numpy as np
import pandas as pd
import pytest

from core.drift_psi import (
    BANDIT_DECAY_DRIFT,
    BANDIT_DECAY_MAX,
    BANDIT_DECAY_MIN,
    BANDIT_DECAY_STABLE,
    PSI_FEATURES,
    PSI_N_BINS,
    PSI_SEVERE_THRESHOLD,
    PSI_STABLE_THRESHOLD,
    DriftMonitor,
    MultiAssetDriftMonitor,
    bandit_decay_for_psi,
    compute_psi,
    extract_psi_features,
    psi_status,
    run_drift_check,
    unified_drift_state,
)
from strategies.engine import BANDIT_DECAY, MetaAllocationEngine, TrendFollowingStrategy


# --------------------------------------------------------------------------- #
# 1. PSI : propriétés mathématiques
# --------------------------------------------------------------------------- #
class TestPsiMath:
    def test_identical_distributions_zero(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0.0, 1.0, 2000)
        psi = compute_psi(x, x)
        assert psi == pytest.approx(0.0, abs=0.05)

    def test_shifted_distribution_positive(self):
        rng = np.random.default_rng(7)
        ref = rng.normal(0.0, 1.0, 2000)
        rec = rng.normal(0.8, 1.0, 2000)   # moyenne décalée
        psi = compute_psi(ref, rec)
        assert psi > 0.3

    def test_volatility_shift_detected(self):
        rng = np.random.default_rng(3)
        ref = rng.normal(0.0, 1.0, 2000)
        rec = rng.normal(0.0, 3.0, 2000)   # vol x3
        psi = compute_psi(ref, rec)
        assert psi > 0.1

    def test_small_samples_neutral(self):
        """Échantillons minuscules -> 0.0 (pas de drift mesurable, honnêteté)."""
        assert compute_psi([1.0], [1.0]) == 0.0
        assert compute_psi(np.array([]), np.array([1.0])) == 0.0

    def test_constant_reference_neutral(self):
        """Référence constante -> 0.0 (le PSI sur variance nulle n'est pas fiable)."""
        assert compute_psi(np.ones(100), np.ones(100) + 0.01) == 0.0

    def test_nan_handled(self):
        rng = np.random.default_rng(11)
        x = rng.normal(0, 1, 500)
        y = x.copy()
        y[::10] = np.nan
        psi = compute_psi(x, y)
        assert 0.0 <= psi < 10.0   # jamais de NaN/inf


# --------------------------------------------------------------------------- #
# 2. Statut + decay du bandit
# --------------------------------------------------------------------------- #
class TestStatusAndDecay:
    def test_status_thresholds(self):
        # seuils MARCHÉ (LOT D calibré : le crédit 0.10/0.25 -> SEVERE permanent)
        assert psi_status(0.05) == "STABLE"
        assert psi_status(0.20) == "STABLE"
        assert psi_status(0.45) == "MODERATE"
        assert psi_status(0.80) == "SEVERE"
        assert psi_status(PSI_STABLE_THRESHOLD) == "MODERATE"
        assert psi_status(PSI_SEVERE_THRESHOLD) == "SEVERE"

    def test_decay_nominal_when_stable(self):
        """Pas de drift -> decay nominal (0.98, comportement pré-LOT D)."""
        assert bandit_decay_for_psi(0.02) == pytest.approx(BANDIT_DECAY_STABLE)
        assert bandit_decay_for_psi(0.0) == pytest.approx(BANDIT_DECAY_STABLE)
        assert bandit_decay_for_psi(0.20) == pytest.approx(BANDIT_DECAY_STABLE)

    def test_decay_accelerated_when_severe(self):
        """Drift sévère -> oubli accéléré (decay plus faible)."""
        assert bandit_decay_for_psi(0.80) == pytest.approx(BANDIT_DECAY_DRIFT)
        assert bandit_decay_for_psi(0.80) < bandit_decay_for_psi(0.02)
        # modéré : interpolation (accélération douce)
        assert bandit_decay_for_psi(0.45) < bandit_decay_for_psi(0.20)

    def test_decay_bounded_hard(self):
        """Bornes dures : jamais d'oubli total ni de mémoire infinie."""
        assert bandit_decay_for_psi(0.0) >= BANDIT_DECAY_MIN
        assert bandit_decay_for_psi(5.0) >= BANDIT_DECAY_MIN
        assert bandit_decay_for_psi(0.0) <= BANDIT_DECAY_MAX
        assert 0.80 <= bandit_decay_for_psi(0.0) <= 1.0


# --------------------------------------------------------------------------- #
# 3. Features réelles depuis les candles
# --------------------------------------------------------------------------- #
class TestFeatures:
    def _candles(self, n=700, drift_vol=1.0, drift_mean=0.0, seed=0,
                 vol_shock_bars=0, vol_shock_mult=1.0):
        rng = np.random.default_rng(seed)
        rets = rng.normal(drift_mean, 0.01 * drift_vol, n)
        if vol_shock_bars > 0:
            rets[-vol_shock_bars:] *= vol_shock_mult
        close = 100.0 * np.exp(np.cumsum(rets))
        volume = np.abs(rng.normal(1000, 200, n)) + 50.0
        idx = pd.date_range("2026-01-01", periods=n, freq="h")
        return pd.DataFrame({"open": close, "high": close * 1.001,
                             "low": close * 0.999, "close": close,
                             "volume": volume}, index=idx)

    def test_extract_features_from_real_candles(self):
        df = self._candles()
        feats = extract_psi_features(df)
        assert set(feats) == set(PSI_FEATURES)
        for name, series in feats.items():
            assert len(series) > 100
            assert np.all(np.isfinite(series))

    def test_empty_df_no_features(self):
        assert extract_psi_features(pd.DataFrame()) == {}

    def test_monitor_stable_on_flat_market(self):
        """Marché homogène -> pas de SEVERE (le bruit d'échantillonnage donne
        au plus MODERATE) + PAS d'accélération forte de l'oubli."""
        df = self._candles(n=700, seed=1)
        m = DriftMonitor(reference_window=400, recent_window=150)
        d = m.update(df)
        assert d["status"] in ("STABLE", "MODERATE")
        assert d["max_psi"] < PSI_SEVERE_THRESHOLD
        # pas d'accélération maximale : decay > midpoint stable/drift
        assert d["bandit_decay_recommended"] >= (BANDIT_DECAY_STABLE + BANDIT_DECAY_DRIFT) / 2

    def test_monitor_detects_vol_regime_change(self):
        """Choc de volatilité en fin de série -> PSI élevé (returns_abs) -> SEVERE."""
        df = self._candles(n=700, vol_shock_bars=150, vol_shock_mult=5.0, seed=2)
        m = DriftMonitor(reference_window=400, recent_window=150)
        d = m.update(df)
        # |rendements| (détecteur de vol direct) doit réagir fortement
        assert d["psi_per_feature"].get("returns_abs", 0.0) > 0.25
        assert d["status"] == "SEVERE"
        assert d["bandit_decay_recommended"] < BANDIT_DECAY_STABLE

    def test_monitor_insufficient_data_neutral(self):
        """Pas assez de barres -> aucun signal (STABLE, decay nominal)."""
        df = self._candles(n=100, seed=3)
        m = DriftMonitor(reference_window=300, recent_window=100)
        d = m.update(df)
        assert d["max_psi"] == 0.0
        assert d["bandit_decay_recommended"] == pytest.approx(BANDIT_DECAY_STABLE)
        assert d["n_updates"] == 0   # aucun calcul annoncé

    def test_monitor_exposes_honest_state(self):
        df = self._candles(n=600, seed=4)
        m = DriftMonitor()
        d = m.update(df)
        for key in ("psi_per_feature", "max_psi", "status", "bandit_decay_recommended",
                    "thresholds", "windows", "n_updates", "n_bars"):
            assert key in d


# --------------------------------------------------------------------------- #
# 4. Bandit : oubli accéléré au drift
# --------------------------------------------------------------------------- #
class TestBanditForgetting:
    def _engine(self):
        return MetaAllocationEngine(strategies=[TrendFollowingStrategy(),
                                                TrendFollowingStrategy()])

    def test_default_decay_is_config(self):
        e = self._engine()
        assert e.bandit_decay == pytest.approx(BANDIT_DECAY)

    def test_set_bandit_decay_bounded(self):
        e = self._engine()
        assert e.set_bandit_decay(0.92) == pytest.approx(0.92)
        assert e.set_bandit_decay(0.01) >= 0.80   # borne dure
        assert e.set_bandit_decay(2.0) <= 1.0

    def test_forgetting_accelerated_with_drift_decay(self):
        """Même feedback, decay accéléré -> alpha/beta oublient plus vite."""
        e1 = self._engine()
        e2 = self._engine()
        signals = {"TrendFollowingStrategy": {"signal": 1.0}}
        # 20 feedbacks gagnants identiques
        for _ in range(20):
            e1.update_bandit_feedback("BTCUSDT", signals, 0.01)
            e2.update_bandit_feedback("BTCUSDT", signals, 0.01, decay=0.92)
        a1, b1 = e1.alpha_bandit[0], e1.beta_bandit[0]
        a2, b2 = e2.alpha_bandit[0], e2.beta_bandit[0]
        # le bandit à oubli accéléré garde MOINS de mémoire
        assert (a2 + b2) < (a1 + b1)

    def test_bandit_decay_dynamic_used_by_update(self):
        """set_bandit_decay influence les updates suivants (sans paramètre)."""
        e = self._engine()
        e.set_bandit_decay(0.90)
        signals = {"TrendFollowingStrategy": {"signal": 1.0}}
        for _ in range(10):
            e.update_bandit_feedback("BTCUSDT", signals, 0.01)
        assert e.alpha_bandit[0] + e.beta_bandit[0] < 30.0  # oubli marqué


# --------------------------------------------------------------------------- #
# 5. DÉMO == RÉAL + config
# --------------------------------------------------------------------------- #
class TestDemoEqualsReal:
    def test_no_mode_in_drift_api(self):
        import inspect
        sig = inspect.signature(DriftMonitor.update)
        assert "mode" not in sig.parameters
        src = inspect.getsource(DriftMonitor)
        assert "active_mode" not in src
        assert "STATE" not in src   # aucun état global du bot

    def test_config_driven(self):
        from core.config import settings
        assert PSI_STABLE_THRESHOLD == settings.get_float("drift", "psi_stable_threshold", 0.10)
        assert PSI_SEVERE_THRESHOLD == settings.get_float("drift", "psi_severe_threshold", 0.25)
        assert PSI_N_BINS == settings.get_int("drift", "psi_n_bins", 10)
        assert BANDIT_DECAY_STABLE == settings.get_float("drift", "bandit_decay_stable", 0.98)
        assert BANDIT_DECAY_DRIFT == settings.get_float("drift", "bandit_decay_drift", 0.92)
        assert BANDIT_DECAY_MIN == settings.get_float("drift", "bandit_decay_min", 0.85)

    def test_deterministic(self):
        df = pd.DataFrame({
            "open": np.linspace(100, 110, 600),
            "high": np.linspace(101, 111, 600),
            "low": np.linspace(99, 109, 600),
            "close": np.linspace(100, 110, 600),
            "volume": np.linspace(1000, 1100, 600),
        })
        m1, m2 = (DriftMonitor(reference_window=400, recent_window=150) for _ in range(2))
        d1 = m1.update(df)
        d2 = m2.update(df)
        assert d1["max_psi"] == d2["max_psi"]
        assert d1["bandit_decay_recommended"] == d2["bandit_decay_recommended"]


# --------------------------------------------------------------------------- #
# 6. Tick intégré (run_drift_check) : intervalle + application au bandit
# --------------------------------------------------------------------------- #
class TestRunDriftCheck:
    def _state(self):
        return {"drift_psi": {}, "drift_psi_last_ts": 0.0}

    def test_interval_gate(self, monkeypatch):
        """Avant l'intervalle -> pas de recalcul (état inchangé)."""
        from strategies.engine import MetaAllocationEngine, TrendFollowingStrategy
        engine = MetaAllocationEngine(strategies=[TrendFollowingStrategy()])
        calls = {"n": 0}

        class FakeDB:
            def load_candles(self, *a, **k):
                calls["n"] += 1
                return pd.DataFrame()
            def add_audit_log(self, *a, **k):
                pass

        state = self._state()
        state["drift_psi_last_ts"] = 1_000_000.0
        import core.drift_psi as dp
        monkeypatch.setattr(dp, "PSI_INTERVAL_SECONDS", 900.0)
        monkeypatch.setattr(dp, "time", type("T", (), {"time": lambda: 1_000_100.0})())
        run_drift_check(state, FakeDB(), DriftMonitor(), engine, lambda: "ip")
        assert calls["n"] == 0   # intervalle non écoulé -> pas de load
        assert state["drift_psi"] == {}

    def test_applies_decay_to_bandit_on_drift(self, monkeypatch):
        """En drift, le decay recommandé est appliqué au bandit (bornee)."""
        import core.drift_psi as dp
        from strategies.engine import MetaAllocationEngine, TrendFollowingStrategy
        engine = MetaAllocationEngine(strategies=[TrendFollowingStrategy()])
        engine.bandit_decay = BANDIT_DECAY_STABLE

        # candles avec choc de vol en fin -> SEVERE
        rng = np.random.default_rng(5)
        n = 700
        rets = rng.normal(0.0, 0.01, n)
        rets[-150:] *= 5.0
        close = 100.0 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({"open": close, "high": close * 1.001,
                           "low": close * 0.999, "close": close,
                           "volume": np.abs(rng.normal(1000, 200, n)) + 50.0})

        class FakeDB:
            def load_candles(self, *a, **k):
                return df
            def add_audit_log(self, *a, **k):
                pass

        state = self._state()
        monkeypatch.setattr(dp, "PSI_INTERVAL_SECONDS", 0.0)   # toujours à jour
        mon = MultiAssetDriftMonitor(symbols=["BTCUSDT"],
                                     reference_window=400, recent_window=150)
        out = run_drift_check(state, FakeDB(), mon, engine, lambda: "ip",
                              load_limit=700)
        assert out["status"] == "SEVERE"
        assert state["drift_psi"]["status"] == "SEVERE"
        assert state["drift_psi"]["bandit_decay_applied"] == pytest.approx(BANDIT_DECAY_DRIFT)
        assert engine.bandit_decay == pytest.approx(BANDIT_DECAY_DRIFT)

    def test_never_blocking_on_error(self):
        """Une erreur (db en panne) ne doit jamais bloquer ni lever."""
        from strategies.engine import MetaAllocationEngine, TrendFollowingStrategy
        engine = MetaAllocationEngine(strategies=[TrendFollowingStrategy()])

        class BrokenDB:
            def load_candles(self, *a, **k):
                raise RuntimeError("db down")

        state = self._state()
        mon = MultiAssetDriftMonitor(symbols=["BTCUSDT"])
        out = run_drift_check(state, BrokenDB(), mon, engine, lambda: "ip")
        assert out == {}            # état inchangé
        assert engine.bandit_decay == pytest.approx(BANDIT_DECAY_STABLE)  # nominal


# --------------------------------------------------------------------------- #
# 7. Corrections LOT D : multi-actifs, fusion CUSUM+PSI, minimum de données
# --------------------------------------------------------------------------- #
class TestMultiAsset:
    def _candles(self, n, seed, vol_mul=1.0, shock=0):
        rng = np.random.default_rng(seed)
        rets = rng.normal(0.0, 0.01, n)
        if shock:
            rets[-shock:] *= vol_mul
        close = 100.0 * np.exp(np.cumsum(rets))
        return pd.DataFrame({"open": close, "high": close * 1.001,
                             "low": close * 0.999, "close": close,
                             "volume": np.abs(rng.normal(1000, 200, n)) + 50.0})

    def test_max_psi_is_worst_case_across_assets(self):
        """2 actifs : un plat + un choc de vol -> max_psi = celui du choc."""
        mon = MultiAssetDriftMonitor(symbols=["BTCUSDT", "ETHUSDT"],
                                     reference_window=400, recent_window=150)
        df_flat = self._candles(700, seed=1)
        df_shock = self._candles(700, seed=2, vol_mul=5.0, shock=150)
        out = mon.update_all({"BTCUSDT": df_flat, "ETHUSDT": df_shock})
        assert out["per_asset"]["BTCUSDT"]["status"] in ("STABLE", "MODERATE")
        assert out["per_asset"]["ETHUSDT"]["status"] == "SEVERE"
        assert out["status"] == "SEVERE"
        assert out["max_psi"] == pytest.approx(
            out["per_asset"]["ETHUSDT"]["max_psi"], abs=1e-6)

    def test_per_asset_exposed(self):
        mon = MultiAssetDriftMonitor(symbols=["BTCUSDT", "XAUUSD"],
                                     reference_window=400, recent_window=150)
        df = self._candles(700, seed=3)
        out = mon.update_all({"BTCUSDT": df, "XAUUSD": df})
        assert set(out["per_asset"]) == {"BTCUSDT", "XAUUSD"}

    def test_no_data_no_state_change(self):
        mon = MultiAssetDriftMonitor(symbols=["BTCUSDT"])
        out = mon.update_all({})
        assert out == {} or out.get("per_asset") == {}


class TestUnifiedFusion:
    def test_psi_severe_alone_is_severe(self):
        u = unified_drift_state(
            {"status": "SEVERE", "max_psi": 0.6, "bandit_decay_recommended": 0.92},
            {"detected": False})
        assert u["status"] == "SEVERE"
        assert u["sources"]["cusum"] == "OK"

    def test_cusum_alone_is_severe(self):
        u = unified_drift_state(
            {"status": "STABLE", "max_psi": 0.02, "bandit_decay_recommended": 0.98},
            {"detected": True, "ts": 1.0})
        assert u["status"] == "SEVERE"
        assert u["bandit_decay_recommended"] == pytest.approx(BANDIT_DECAY_DRIFT)

    def test_both_stable_is_stable(self):
        u = unified_drift_state(
            {"status": "STABLE", "max_psi": 0.02, "bandit_decay_recommended": 0.98},
            {"detected": False})
        assert u["status"] == "STABLE"
        assert u["bandit_decay_recommended"] == pytest.approx(BANDIT_DECAY_STABLE)

    def test_moderate_stays_moderate(self):
        u = unified_drift_state(
            {"status": "MODERATE", "max_psi": 0.15, "bandit_decay_recommended": 0.95},
            {"detected": False})
        assert u["status"] == "MODERATE"

    def test_decay_is_most_aggressive(self):
        u = unified_drift_state(
            {"status": "STABLE", "max_psi": 0.02, "bandit_decay_recommended": 0.98},
            {"detected": True})
        assert u["bandit_decay_recommended"] <= 0.98


class TestMinimumData:
    def test_insufficient_bars_no_calculation(self):
        """< 550 barres : pas de calcul (le bruit H0 du PSI est prouvé
        erratique en dessous — faux SEVERE possible)."""
        rng = np.random.default_rng(4)
        n = 400
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        df = pd.DataFrame({"open": close, "high": close * 1.001,
                           "low": close * 0.999, "close": close,
                           "volume": np.abs(rng.normal(1000, 200, n)) + 50.0})
        m = DriftMonitor()
        d = m.update(df)
        assert d["n_updates"] == 0
        assert d["max_psi"] == 0.0
        assert d["bandit_decay_recommended"] == pytest.approx(BANDIT_DECAY_STABLE)

    def test_sufficient_bars_calculates(self):
        """Fenêtres nominales (2000/400) : le calcul exige 2400+ barres."""
        rng = np.random.default_rng(6)
        n = 2500
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        df = pd.DataFrame({"open": close, "high": close * 1.001,
                           "low": close * 0.999, "close": close,
                           "volume": np.abs(rng.normal(1000, 200, n)) + 50.0})
        m = DriftMonitor()
        d = m.update(df)
        assert d["n_updates"] == 1
        assert d["n_bars"] == 2500

    def test_flat_market_nominal_windows_stable(self):
        """Avec les fenêtres nominales (2000/400), un marché plat reste
        STABLE (bruit H0 < 0.20 mesuré) — pas de faux SEVERE."""
        rng = np.random.default_rng(9)
        n = 2600
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        df = pd.DataFrame({"open": close, "high": close * 1.001,
                           "low": close * 0.999, "close": close,
                           "volume": np.abs(rng.normal(1000, 200, n)) + 50.0})
        m = DriftMonitor()
        d = m.update(df)
        assert d["n_updates"] == 1
        assert d["status"] != "SEVERE"
        assert d["bandit_decay_recommended"] >= (BANDIT_DECAY_STABLE + BANDIT_DECAY_DRIFT) / 2


class TestDeepFetch:
    def test_binance_url_requests_700_bars(self, monkeypatch):
        """Le fetch historique demande 700 barres (PSI exige ~550)."""
        import main  # noqa: F401  (charge d'abord main COMPLET — évite le

        # circular import quand on importe le module extrait seul)
        import market_data.historical_fetch as hf
        captured = {}

        class FakeResp:
            status_code = 200
            def json(self):
                return [[1700000000000, "100", "110", "90", "105", "1000"]]

        class FakeClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def get(self, url, *a, **k):
                captured["url"] = url
                return FakeResp()

        monkeypatch.setattr(hf, "httpx", type("H", (), {"AsyncClient": lambda *a, **k: FakeClient()})())
        import asyncio
        df = asyncio.run(hf.fetch_historical_market_data("BTCUSDT"))
        assert captured["url"] is not None
        assert "limit=700" in captured["url"]
        assert not df.empty

    def test_yahoo_range_is_6mo(self):
        """Le fetch Yahoo utilise range_str=6mo (profondeur PSI)."""
        import inspect

        import main  # noqa: F401
        import market_data.historical_fetch as hf
        src = inspect.getsource(hf)
        assert 'range_str="6mo"' in src
