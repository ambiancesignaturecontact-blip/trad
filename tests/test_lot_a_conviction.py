"""
LOT A (F1 — conviction / fréquence de trading).

1. Plancher anti-empilement relevé 0.15 -> 0.25 : les signaux sont calibrés
   (p50≈0.27), la conviction n'est plus le frein ; le risque est porté par
   Kelly ¼ / CVaR / drawdowns. La taille post-pipeline n'est plus écrasée à
   15 % systématiquement.
2. Seuil de conviction ADAPTATIF à la distribution réelle des signaux (p25
   des |signaux|) au lieu de la constante 0.08 qui bornait le seuil à
   [0.08, 0.14] — le seuil suit la conviction réellement produite.
3. active_factors : nombre de facteurs réellement actifs (< 1.0) exposé —
   transparence de l'empilement (le diagnostic « c'est quoi qui bloque »).
"""
import pytest

from core.meta_cognition import adaptive_conviction_threshold
from core.risk_pipeline import FINAL_SCALE_FLOOR, apply_risk_pipeline

# --------------------------------------------------------------------------- #
# 1. Plancher 0.25
# --------------------------------------------------------------------------- #

def test_floor_raised_to_25():
    """Le plancher par défaut est 0.25 (plus 0.15)."""
    assert FINAL_SCALE_FLOOR == pytest.approx(0.25)


def test_floor_applies_at_25():
    """Un produit sous 0.25 est clampé à 0.25 (ex. 16 facteurs à 0.8)."""
    res = apply_risk_pipeline(
        base_qty=1000.0, cvar_qty=1e9, max_asset_qty=1e9, conviction=0.8,
        risk_state_scale=0.8, news_scale=0.8, macro_scale=0.8, onchain_scale=0.8,
        corr_scale=0.8, confidence_scale=0.8, org_scale=0.8, rlhf_scale=0.8,
        vol_scale=0.8, tradability_scale=0.8, capacity_scale=0.8,
        cash_reserve_scale=0.8, order_flow_scale=0.8, regime_confidence_scale=0.8)
    assert res["final_scale"] == pytest.approx(0.25)
    assert res["qty"] == pytest.approx(250.0)


def test_floor_not_applied_above():
    """Un produit à 0.5 reste 0.5 (le plancher ne gonfle jamais la taille)."""
    res = apply_risk_pipeline(
        base_qty=100.0, cvar_qty=1000.0, max_asset_qty=1000.0,
        conviction=1.0, risk_state_scale=1.0, capacity_scale=0.5)
    assert res["qty"] == pytest.approx(50.0)
    assert res["steps"][-1]["step"] != "cumulative_floor"


def test_floor_hard_block_still_zero():
    """Un blocage dur (HALT=0) reste intégral : pas de plancher sur 0."""
    res = apply_risk_pipeline(
        base_qty=100.0, cvar_qty=1000.0, max_asset_qty=1000.0,
        conviction=0.5, risk_state_scale=0.0)
    assert res["qty"] == 0.0 and res["final_scale"] == 0.0


# --------------------------------------------------------------------------- #
# 2. Seuil adaptatif (base = p25 des signaux réels)
# --------------------------------------------------------------------------- #

def test_threshold_uses_real_signal_distribution():
    """Base = p25 des |signaux| récents (plus une constante arbitraire)."""
    sigs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 3
    rets = [0.01] * 30
    thr = adaptive_conviction_threshold(sigs, rets, base_threshold=None)
    # p25 de [0.1..1.0] = 0.325 ; accuracy 1.0 -> factor 0.25 -> 0.081 -> clip 0.08
    # mais le seuil n'est PLUS collé à la constante : avec accuracy 1.0 il
    # descend au min 0.08 (on fait confiance), avec accuracy 0.5 il reste ~p25
    thr_mid = adaptive_conviction_threshold(
        sigs, [0.01]*15 + [-0.01]*15, base_threshold=None)
    assert thr == pytest.approx(0.08, abs=0.001)          # accuracy 1.0 -> min
    assert thr_mid == pytest.approx(0.30, abs=0.02)       # accuracy 0.5 -> p25
    assert thr_mid > 0.08                                 # plus collé au plancher


def test_threshold_base_bounded_by_distribution():
    """Avec des signaux forts (p25 élevé), le seuil peut monter bien au-dessus
    de 0.08 (max_threshold 0.30) — impossible avant (constante 0.08)."""
    sigs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0] * 3
    rets = [0.01]*15 + [-0.01]*15  # accuracy 0.5 -> neutre
    thr = adaptive_conviction_threshold(sigs, rets, base_threshold=None)
    assert 0.3 <= thr <= 0.45  # p25 ≈ 0.55 -> clamp max 0.30, factor ~1
    # en fait p25 de la liste = 0.55 -> clip(0.55, 0.08, 0.30) = 0.30
    assert thr == pytest.approx(0.30, abs=0.02)


def test_threshold_no_history_falls_back_min():
    assert adaptive_conviction_threshold([], [], base_threshold=None) == pytest.approx(0.08)


def test_main_passes_none_base():
    """La boucle live passe base_threshold=None (verrou anti-régression)."""
    src = open("main.py").read()
    assert "adaptive_conviction_threshold(" in src
    assert "base_threshold=None" in src


# --------------------------------------------------------------------------- #
# 3. active_factors (transparence de l'empilement)
# --------------------------------------------------------------------------- #

def test_active_factors_counted():
    """active_factors = nb de facteurs < 1.0 (les neutres ne comptent pas)."""
    res = apply_risk_pipeline(
        base_qty=100.0, cvar_qty=1000.0, max_asset_qty=1000.0,
        conviction=1.0, risk_state_scale=1.0, capacity_scale=0.5,
        cash_reserve_scale=0.8)
    assert res["active_factors"] == 2  # capacity + cash_reserve
    res2 = apply_risk_pipeline(
        base_qty=100.0, cvar_qty=1000.0, max_asset_qty=1000.0,
        conviction=1.0, risk_state_scale=1.0)
    assert res2["active_factors"] == 0


def test_active_factors_in_telemetry():
    """La télémétrie expose active_factors_last."""
    import main  # noqa: F401  (nécessaire : telemetry s'importe via main)
    import telemetry
    tel = telemetry.compile_telemetry_data()
    assert "active_factors_last" in tel
    assert isinstance(tel["active_factors_last"], int)
