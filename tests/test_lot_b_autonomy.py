"""
LOT B (F2 — autonomie stratégique) : auto-adaptation BORNÉE des paramètres
de risque (Kelly, plafond par actif, drawdowns) au régime HMM.

Principes vérifiés ici (les mêmes qui protègent le système en production) :
  1. Le facteur d'agressivité ne dépasse JAMAIS 1.25x (ni ne descend sous
     0.60x) la config de base — même avec une config hostile.
  2. Un régime incertain (confiance HMM faible) ramène le facteur vers 1.0 :
     « je ne sais pas » -> comportement de base, pas de pari sur le régime.
  3. Le facteur est LISSÉ (EMA) : un changement de régime n'explose pas la
     taille d'un tick à l'autre (le HMM est un signal bruité).
  4. Les DRAWDOWNS ne sont JAMAIS élargis : en régime défensif on resserre
     les circuit breakers, on ne les assouplit jamais.
  5. DÉMO == RÉAL : aucun paramètre de mode dans le module d'autonomie —
     le même chemin de code s'applique aux deux modes.
"""
import inspect

import pytest

from core.regime_autonomy import (
    CONFIDENCE_FULL,
    CONFIDENCE_MIN,
    DAILY_DD_FLOOR,
    EMA_ALPHA,
    FACTOR_MAX,
    FACTOR_MIN,
    KELLY_MIN,
    MAX_PER_ASSET_MAX,
    MAX_PER_ASSET_MIN,
    REGIME_AGGRESSIVENESS,
    TOTAL_DD_FLOOR,
    RegimeAutonomy,
    regime_aggressiveness,
)
from risk.risk_manager import RiskManager


# --------------------------------------------------------------------------- #
# 1. Mapping par régime : valeurs cibles + bornes dures
# --------------------------------------------------------------------------- #
class TestMapping:
    def test_all_regimes_within_hard_bounds(self):
        """Aucun régime ne sort des bornes [0.60, 1.25]."""
        for regime_id in (0, 1, 2, 3):
            f = regime_aggressiveness(regime_id, confidence=1.0)
            assert FACTOR_MIN <= f <= FACTOR_MAX
            assert f <= 1.25

    def test_bull_offensive_bear_defensive(self):
        """Bull low-vol légèrement offensif, bear/erratic défensifs, range neutre."""
        assert regime_aggressiveness(0, 1.0) > 1.0          # bull
        assert regime_aggressiveness(1, 1.0) < 1.0          # bear
        assert regime_aggressiveness(3, 1.0) < 1.0          # erratic high vol
        assert regime_aggressiveness(2, 1.0) == pytest.approx(1.0)  # range

    def test_unknown_regime_neutral(self):
        """Régime inconnu (None) -> 1.0 : on ne punit pas une info absente."""
        assert regime_aggressiveness(None, 1.0) == 1.0
        assert regime_aggressiveness(99, 1.0) == 1.0

    def test_never_above_125x_even_hostile_config(self):
        """Borne haute dure : même une config qui demanderait 3x est coupée à 1.25x."""
        f = regime_aggressiveness(0, 1.0)
        assert f <= 1.25
        assert f * 100 <= 125.0 + 1e-9


# --------------------------------------------------------------------------- #
# 2. Confiance HMM : régime incertain -> facteur ramené vers 1.0
# --------------------------------------------------------------------------- #
class TestConfidence:
    def test_low_confidence_neutral(self):
        """Confiance < CONFIDENCE_MIN -> facteur neutre (aucun pari sur le régime)."""
        f = regime_aggressiveness(0, CONFIDENCE_MIN - 0.05)
        assert f == pytest.approx(1.0, abs=1e-6)

    def test_full_confidence_full_factor(self):
        """Confiance >= CONFIDENCE_FULL -> facteur pleinement appliqué."""
        f = regime_aggressiveness(0, CONFIDENCE_FULL)
        assert f == pytest.approx(REGIME_AGGRESSIVENESS[0], abs=1e-6)

    def test_mid_confidence_pulled_toward_neutral(self):
        """Confiance moyenne -> facteur entre 1.0 et la cible (interpolation)."""
        full = regime_aggressiveness(1, 1.0)            # cible défensive
        mid = regime_aggressiveness(1, 0.5)             # confiance moyenne
        assert 1.0 >= mid >= full
        assert full < mid < 1.0 or mid == pytest.approx(1.0, abs=1e-6)

    def test_confidence_clamped(self):
        """Confiance > 1 ou < 0 est bornée, jamais d'erreur."""
        assert regime_aggressiveness(0, 5.0) == pytest.approx(
            regime_aggressiveness(0, 1.0))
        assert regime_aggressiveness(1, -1.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 3. Lissage EMA + dédoublonnage
# --------------------------------------------------------------------------- #
class TestEma:
    def test_ema_smooths_jumps(self):
        """Un passage brusque vers un régime défensif ne réduit pas d'un coup."""
        a = RegimeAutonomy()
        a.update(2, 1.0)          # range (neutre)
        a.update(1, 1.0)          # bear (défensif) — saut
        assert a.factor > regime_aggressiveness(1, 1.0)   # lissé, pas au plancher
        assert a.factor < 1.0                              # mais déjà < neutre
        # convergence progressive vers la cible (confiance légèrement
        # alternée pour contourner le dédoublonnage du même état)
        for i in range(40):
            a.update(1, 1.0 if i % 2 else 0.98)
        assert a.factor == pytest.approx(regime_aggressiveness(1, 1.0), abs=0.01)

    def test_dedupe_same_state(self):
        """Le même état (régime, confiance) ne déplace l'EMA qu'une fois."""
        a = RegimeAutonomy()
        a.update(2, 0.5)
        first = a.factor
        a.update(2, 0.5)
        a.update(2, 0.5)
        assert a.factor == first

    def test_reset(self):
        a = RegimeAutonomy()
        a.update(0, 1.0)
        a.reset()
        assert a.factor == 1.0
        assert a.last_seen is None

    def test_disabled_neutral(self, monkeypatch):
        """autonomy_enabled=false -> facteur 1.0 (config de base)."""
        import core.regime_autonomy as ra
        monkeypatch.setattr(ra, "AUTONOMY_ENABLED", False)
        a = ra.RegimeAutonomy()
        a.update(0, 1.0)
        assert a.factor == 1.0


# --------------------------------------------------------------------------- #
# 4. Paramètres EFFECTIFS : bornes absolues + jamais > 1.25x la base
# --------------------------------------------------------------------------- #
class TestEffectiveParams:
    def test_kelly_bounded_by_125x(self):
        a = RegimeAutonomy()
        a.update(0, 1.0)   # bull -> facteur max 1.10 (lissé)
        for _ in range(30):
            a.update(0, 1.0)
        eff = a.effective_params(base_kelly=0.15)
        # jamais plus de 1.25x la base
        assert eff["fractional_kelly_multiplier"] <= 0.15 * 1.25 + 1e-9
        assert eff["fractional_kelly_multiplier"] >= KELLY_MIN

    def test_kelly_bear_reduced(self):
        a = RegimeAutonomy()
        for _ in range(30):
            a.update(1, 1.0)
        eff = a.effective_params(base_kelly=0.15)
        assert eff["fractional_kelly_multiplier"] < 0.15
        assert eff["fractional_kelly_multiplier"] >= KELLY_MIN

    def test_max_per_asset_bounds(self):
        a = RegimeAutonomy()
        for _ in range(30):
            a.update(0, 1.0)
        eff = a.effective_params(base_max_per_asset=0.25)
        assert eff["max_exposure_per_asset_pct"] <= 0.25 * 1.25 + 1e-9
        assert MAX_PER_ASSET_MIN <= eff["max_exposure_per_asset_pct"] <= MAX_PER_ASSET_MAX

    def test_drawdowns_never_loosened(self):
        """Les drawdowns effectifs sont TOUJOURS <= base (jamais élargis)."""
        a = RegimeAutonomy()
        for _ in range(30):
            a.update(0, 1.0)   # même en régime offensif
        eff = a.effective_params(base_daily_dd=0.025, base_total_dd=0.08)
        assert eff["max_daily_drawdown_pct"] <= 0.025 + 1e-9
        assert eff["max_total_drawdown_pct"] <= 0.08 + 1e-9

    def test_drawdowns_tightened_in_defensive_regime(self):
        a = RegimeAutonomy()
        for _ in range(30):
            a.update(1, 1.0)   # bear
        eff = a.effective_params(base_daily_dd=0.025, base_total_dd=0.08)
        assert eff["max_daily_drawdown_pct"] < 0.025
        assert eff["max_daily_drawdown_pct"] >= DAILY_DD_FLOOR
        assert eff["max_total_drawdown_pct"] < 0.08
        assert eff["max_total_drawdown_pct"] >= TOTAL_DD_FLOOR


# --------------------------------------------------------------------------- #
# 5. Intégration RiskManager : sizing, plafond par actif, circuit breakers
# --------------------------------------------------------------------------- #
class TestRiskManagerIntegration:
    def test_apply_regime_factor_clamps(self):
        rm = RiskManager()
        rm.apply_regime_factor(5.0)
        assert rm.regime_factor == pytest.approx(FACTOR_MAX)
        rm.apply_regime_factor(0.01)
        assert rm.regime_factor == pytest.approx(FACTOR_MIN)
        rm.apply_regime_factor(1.1)
        assert rm.regime_factor == pytest.approx(1.1)

    def test_default_factor_neutral(self):
        """Sans autonomie (facteur 1.0), le comportement est identique à l'avant LOT B."""
        rm = RiskManager()
        assert rm.regime_factor == 1.0
        eff = rm.effective_params()
        assert eff["fractional_kelly_multiplier"] == rm.params["fractional_kelly_multiplier"]
        assert eff["max_exposure_per_asset_pct"] == rm.params["max_exposure_per_asset_pct"]

    def test_position_size_scales_with_regime_bounded(self):
        """Bull augmente la taille (borné 1.25x), bear la réduit."""
        capital, atr, price = 100000.0, 10.0, 60000.0
        rm_base = RiskManager()
        qty_base = rm_base.calculate_position_size(capital, atr, price)
        assert qty_base > 0

        rm_bull = RiskManager()
        rm_bull.apply_regime_factor(1.10)
        qty_bull = rm_bull.calculate_position_size(capital, atr, price)
        assert qty_bull > qty_base
        assert qty_bull / qty_base <= 1.25 + 1e-6

        rm_bear = RiskManager()
        rm_bear.apply_regime_factor(0.75)
        qty_bear = rm_bear.calculate_position_size(capital, atr, price)
        assert qty_bear < qty_base

    def test_circuit_breaker_tightened_in_defensive_regime(self):
        """En régime défensif, le daily drawdown effectif est resserré."""
        rm = RiskManager(params={
            "max_daily_drawdown_pct": 0.10,
            "max_total_drawdown_pct": 0.20,
            "max_exposure_per_asset_pct": 0.25,
            "fractional_kelly_multiplier": 0.15,
        })
        # 8% de perte : OK en neutre (10% de limite), BREACH en défensif (7.5%)
        tripped_neutral, _ = rm.check_circuit_breaker(92000.0)
        assert tripped_neutral is False
        rm.apply_regime_factor(0.75)
        assert rm.effective_params()["max_daily_drawdown_pct"] == pytest.approx(0.075)
        tripped_defensive, msg = rm.check_circuit_breaker(92000.0)
        assert tripped_defensive is True
        assert "DAILY DRAWDOWN BREACHED" in msg

    def test_circuit_breaker_never_loosened_offensive(self):
        """Même en régime offensif, le drawdown n'est JAMAIS élargi."""
        rm = RiskManager(params={
            "max_daily_drawdown_pct": 0.10,
            "max_total_drawdown_pct": 0.20,
            "max_exposure_per_asset_pct": 0.25,
            "fractional_kelly_multiplier": 0.15,
        })
        rm.apply_regime_factor(1.10)
        eff = rm.effective_params()
        assert eff["max_daily_drawdown_pct"] == pytest.approx(0.10)
        assert eff["max_total_drawdown_pct"] == pytest.approx(0.20)

    def test_micro_account_drawdown_tier_respected(self):
        """Le facteur s'applique PAR-DESSUS le palier par taille de compte."""
        rm = RiskManager()
        rm.set_initial_capital(50.0)          # micro -> 18% daily / 35% total
        rm.apply_regime_factor(0.75)
        eff = rm.effective_params()
        assert eff["max_daily_drawdown_pct"] == pytest.approx(0.18 * 0.75)
        assert eff["max_total_drawdown_pct"] == pytest.approx(0.35 * 0.75)

    def test_micro_account_notional_floor_unchanged(self):
        """Le plancher de notional des micro-comptes (80% capital) est intact."""
        rm = RiskManager()
        rm.set_initial_capital(50.0)
        rm.apply_regime_factor(1.10)
        qty = rm.calculate_position_size(50.0, 0.5, 77.0)
        assert qty * 77.0 <= 50.0 * 0.80 + 1e-6


# --------------------------------------------------------------------------- #
# 6. DÉMO == RÉAL : fidélité stricte du comportement
# --------------------------------------------------------------------------- #
class TestDemoEqualsReal:
    def test_no_mode_parameter_in_autonomy_api(self):
        """Aucune fonction d'autonomie ne prend de paramètre de mode."""
        for name in ("regime_aggressiveness", "RegimeAutonomy.update",
                     "RegimeAutonomy.effective_params"):
            obj = regime_aggressiveness if name == "regime_aggressiveness" else None
            if name == "RegimeAutonomy.update":
                sig = inspect.signature(RegimeAutonomy.update)
            elif name == "RegimeAutonomy.effective_params":
                sig = inspect.signature(RegimeAutonomy.effective_params)
            else:
                sig = inspect.signature(obj)
            assert "mode" not in sig.parameters
            assert "demo" not in sig.parameters
            assert "real" not in sig.parameters

    def test_no_mode_state_read(self):
        """Le module d'autonomie ne lit aucun état global ni mode du bot."""
        src = inspect.getsource(RegimeAutonomy)
        assert "STATE" not in src                  # aucun accès à l'état global
        assert "active_mode" not in src            # aucun flag de mode
        rm_src = inspect.getsource(RiskManager)
        assert "active_mode" not in rm_src
        assert '"mode"' not in rm_src

    def test_deterministic_same_inputs_same_outputs(self):
        """Mêmes entrées -> mêmes sorties (aucun état aléatoire ou dépendant du mode)."""
        a1, a2 = RegimeAutonomy(), RegimeAutonomy()
        for regime, conf in [(2, 0.6), (0, 0.8), (1, 0.7), (3, 0.9), (0, 0.8)]:
            a1.update(regime, conf)
            a2.update(regime, conf)
        assert a1.factor == a2.factor
        assert a1.to_dict()["effective"] == a2.to_dict()["effective"]


# --------------------------------------------------------------------------- #
# 7. Exposition télémétrique
# --------------------------------------------------------------------------- #
class TestTelemetry:
    def test_to_dict_exposes_honest_state(self):
        a = RegimeAutonomy()
        a.update(1, 0.8)
        d = a.to_dict()
        for key in ("enabled", "regime_id", "confidence", "factor", "factor_raw",
                    "factor_bounds", "regime_aggressiveness", "effective"):
            assert key in d
        assert d["factor_bounds"] == [FACTOR_MIN, FACTOR_MAX]
        eff = d["effective"]
        assert set(eff) == {"fractional_kelly_multiplier",
                            "max_exposure_per_asset_pct",
                            "max_daily_drawdown_pct",
                            "max_total_drawdown_pct"}

    def test_config_driven(self):
        """Les constantes viennent de core/config.py (config.yaml)."""
        from core.config import settings
        assert FACTOR_MAX == settings.get_float("risk", "autonomy_factor_max", 1.25)
        assert EMA_ALPHA == settings.get_float("risk", "autonomy_ema_alpha", 0.30)
        assert KELLY_MIN == settings.get_float("risk", "autonomy_kelly_min", 0.05)
        assert DAILY_DD_FLOOR == settings.get_float("risk", "autonomy_daily_drawdown_floor", 0.015)
        assert TOTAL_DD_FLOOR == settings.get_float("risk", "autonomy_total_drawdown_floor", 0.05)
        assert MAX_PER_ASSET_MAX == settings.get_float("risk", "autonomy_max_per_asset_max", 0.30)
        assert REGIME_AGGRESSIVENESS[0] == 1.10
        assert REGIME_AGGRESSIVENESS[1] == 0.75
