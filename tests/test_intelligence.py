"""
Mission « intelligence » — Axes 1/2/3 implémentés et verrouillés.

Axe 1 : conviction CALIBRÉE par le win rate réel (meta-labeling — la taille
        reflète la probabilité calibrée de succès, pas seulement |signal|).
Axe 2 : RegimeSwitchingAllocator ADAPTATIF — poids de régime appris en ligne
        (performance réelle par régime×stratégie), bornés par l'a priori.
Axe 3 : budget de risque CONDITIONNEL au régime (forte vol -> budget réduit).

Principes : bornes serrées (jamais de sur-réaction au bruit), neutre quand
l'information manque (pas d'échantillon -> pas de changement), et le pipeline
de risque garde la main sur la taille finale.
"""
import pytest

from core.portfolio_allocator import PortfolioAllocator, regime_risk_scale
from core.risk_pipeline import (
    CONVICTION_CALIB_MAX,
    CONVICTION_CALIB_MIN,
    WIN_RATE_CEIL,
    WIN_RATE_FLOOR,
    calibrated_conviction,
)
from strategies.regime_switching import MAX_ADAPTIVE_SHIFT, RegimeSwitchingAllocator

# --------------------------------------------------------------------------- #
# AXE 1 : conviction calibrée
# --------------------------------------------------------------------------- #

def test_conviction_neutral_without_winrate():
    """Pas d'historique (win_rate=None) -> conviction = |signal| (neutre)."""
    assert calibrated_conviction(0.2, None) == pytest.approx(0.2)
    assert calibrated_conviction(-0.3, None) == pytest.approx(0.3)


def test_conviction_scaled_by_winrate():
    """win rate élevé -> conviction amplifiée ; faible -> réduite ; bornée."""
    base = 0.2
    lo = calibrated_conviction(base, WIN_RATE_FLOOR)      # 0.45 -> x0.60
    hi = calibrated_conviction(base, WIN_RATE_CEIL)       # 0.65 -> x1.25
    mid = calibrated_conviction(base, 0.55)               # 0.55 -> x~0.93
    assert lo < base < hi
    assert mid == pytest.approx(0.2 * (0.60 + 0.10 * 3.25), abs=0.01)  # 0.2 * 0.925
    assert lo == pytest.approx(base * CONVICTION_CALIB_MIN, abs=1e-6)
    assert hi == pytest.approx(base * CONVICTION_CALIB_MAX, abs=1e-6)
    # bornes dures
    assert 0.0 <= lo <= CONVICTION_CALIB_MAX
    assert 0.0 <= hi <= CONVICTION_CALIB_MAX


def test_conviction_bounds_respected_extremes():
    """Même avec un win rate extrême, la conviction reste bornée."""
    v = calibrated_conviction(1.0, 0.99)
    assert v <= CONVICTION_CALIB_MAX
    v2 = calibrated_conviction(0.5, 0.0)
    assert v2 == pytest.approx(0.5 * CONVICTION_CALIB_MIN, abs=1e-6)


def test_main_uses_calibrated_conviction():
    """La boucle live appelle calibrated_conviction (verrou anti-régression)."""
    src = open("main.py").read()
    assert "calibrated_conviction(" in src
    assert "win_tracker.get(_dom_kelly)" in src


# --------------------------------------------------------------------------- #
# AXE 3 : risk budget conditionnel au régime
# --------------------------------------------------------------------------- #

def test_regime_risk_scale_high_vol_reduces_budget():
    """Régimes à forte volatilité (1, 3) -> budget réduit ; calmes -> 1.0."""
    assert regime_risk_scale(0) == 1.0
    assert regime_risk_scale(2) == 1.0
    assert regime_risk_scale(1) == pytest.approx(0.70)
    assert regime_risk_scale(3) == pytest.approx(0.80)
    # régime inconnu / None -> neutre (pas de punition sans information)
    assert regime_risk_scale(None) == 1.0
    assert regime_risk_scale(99) == 1.0


def test_total_risk_budget_applies_regime_scale():
    """Le budget total intègre le scale de régime (et l'expose)."""
    pa = PortfolioAllocator()
    b = pa.total_risk_budget(100000.0, regime_id=1)
    b0 = pa.total_risk_budget(100000.0, regime_id=0)
    assert b["regime_scale"] == pytest.approx(0.70)
    assert b["budget"] == pytest.approx(b0["budget"] * 0.70, rel=0.01)
    assert b["regime_id"] == 1


def test_rebalance_exposes_regime_scale():
    pa = PortfolioAllocator()
    state = {}
    pa.rebalance(state, 100000.0, regime_id=1)
    alloc = state["portfolio_allocation"]["total_risk_budget"]
    assert alloc["regime_scale"] == pytest.approx(0.70)


# --------------------------------------------------------------------------- #
# AXE 2 : regime allocator adaptatif
# --------------------------------------------------------------------------- #

def test_regime_allocator_neutral_without_history():
    """Sans historique, les poids = a priori statique normalisé."""
    ra = RegimeSwitchingAllocator()
    w = ra.get_regime_weights(0)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    # a priori : Trend dominant en régime 0
    assert w["Trend Following"] > w["Mean Reversion"]


def test_regime_allocator_learns_from_real_pnl():
    """Après des pertes réelles de Trend en régime 0, son poids baisse mais
    reste borné (jamais de sur-réaction)."""
    ra = RegimeSwitchingAllocator()
    # 6 trades perdants pour Trend en régime 0 (>= min_observations)
    for _ in range(6):
        ra.update_regime_performance(0, "Trend Following", -0.01)
    w = ra.get_regime_weights(0)
    w_neutral = RegimeSwitchingAllocator().get_regime_weights(0)
    assert w["Trend Following"] < w_neutral["Trend Following"]
    # borne : le poids ne peut pas s'effondrer (a priori conservé à >= 70 %)
    assert w["Trend Following"] >= w_neutral["Trend Following"] * (1 - MAX_ADAPTIVE_SHIFT)
    # toujours normalisé
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_regime_allocator_boosts_winner():
    """Des gains réels de Grid en régime 2 boostent son poids (borné)."""
    ra = RegimeSwitchingAllocator()
    for _ in range(6):
        ra.update_regime_performance(2, "Grid Trading", 0.01)
    w = ra.get_regime_weights(2)
    w_neutral = RegimeSwitchingAllocator().get_regime_weights(2)
    assert w["Grid Trading"] > w_neutral["Grid Trading"]
    assert w["Grid Trading"] <= w_neutral["Grid Trading"] * (1 + MAX_ADAPTIVE_SHIFT)


def test_regime_allocator_requires_min_observations():
    """Moins de 5 observations -> aucun changement (pas de réaction au bruit)."""
    ra = RegimeSwitchingAllocator()
    for _ in range(2):  # seulement 2 trades
        ra.update_regime_performance(0, "Trend Following", -0.05)
    w = ra.get_regime_weights(0)
    w_neutral = RegimeSwitchingAllocator().get_regime_weights(0)
    assert w["Trend Following"] == pytest.approx(w_neutral["Trend Following"], abs=1e-6)


def test_main_calls_regime_performance_update():
    """La boucle live met à jour la performance par régime (verrou)."""
    src = open("main.py").read()
    assert "update_regime_performance(" in src
    assert 'STATE.get("regime_id", 2)' in src
