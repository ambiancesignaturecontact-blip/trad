"""
LOT 5 — P1-10 / P1-11 / P1-12 (audit indépendant §4.4 / §2.5 / §2.6 / §2.2).

P1-12 : le bandit Thompson a un facteur d'oubli (non-stationnarité, §2.6)
        et son tirage est FIGÉ par cycle de décision, plus par tick (§2.2).
P1-11 : update_pnl_attribution passe par le Sharpe DÉFLATÉ (López de Prado,
        §2.5) avec plancher d'échantillon : sous le plancher, l'ajustement de
        poids est borné à ±20 % par mise à jour.
P1-10 : corrélation des SIGNAUX inter-stratégies (§4.4) — le méta-allocateur
        pénalise les stratégies redondantes au lieu de croire diversifier.
"""
import time
from collections import deque

import numpy as np
import pandas as pd
import pytest

import strategies.engine as eng
from strategies.engine import (MetaAllocationEngine, TrendFollowingStrategy,
                               MeanReversionStrategy, BaseStrategy)


# ---------------------------------------------------------------- P1-12 ----
# §2.6 : facteur d'oubli
# ---------------------------------------------------------------------------

def test_bandit_decay_applied_before_update():
    """P1-12 (§2.6) : alpha/beta décroissent (x0.98) avant d'ajouter
    l'observation — un vieux succès s'estompe, il ne se fige pas à vie."""
    meta = MetaAllocationEngine(strategies=[BaseStrategy("S")])
    assert meta.alpha_bandit[0] == 1.0 and meta.beta_bandit[0] == 1.0
    meta.update_bandit_feedback("BTCUSDT", {"S": {"signal": 1.0}}, 0.01)
    # sans oubli : alpha serait exactement 2.0 ; avec oubli : 1.0*0.98+1.0 = 1.98
    assert meta.alpha_bandit[0] == pytest.approx(1.98, abs=1e-9)
    assert meta.beta_bandit[0] == pytest.approx(0.98, abs=1e-9)
    # l'oubli est bien dans le code de update_bandit_feedback
    src = open("strategies/engine.py", encoding="utf-8").read()
    assert "BANDIT_DECAY" in src


def test_bandit_decay_configurable():
    """La config strategies.bandit_decay est branchée (défaut 0.98)."""
    assert eng.BANDIT_DECAY == pytest.approx(0.98)


# ---------------------------------------------------------------- P1-12 ----
# §2.2 : tirage figé par cycle de décision
# ---------------------------------------------------------------------------

def test_bandit_sample_frozen_between_refresh(monkeypatch):
    """Deux appels consécutifs dans la fenêtre -> MÊME tirage (plus de
    ré-échantillonnage à chaque tick)."""
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(eng.time, "time", lambda: clock["t"])
    meta = MetaAllocationEngine(strategies=[BaseStrategy("S")])
    v1 = meta._sample_bandit(0)
    v2 = meta._sample_bandit(0)
    assert v1 == v2
    # après la fenêtre (60 s), un nouveau tirage est effectué
    clock["t"] += eng.BANDIT_SAMPLE_REFRESH_SECONDS + 1.0
    v3 = meta._sample_bandit(0)
    assert meta._bandit_sample_cache[0][0] == clock["t"]  # cache rafraîchi
    assert 0.0 <= v3 <= 1.0


def test_allocate_uses_frozen_sample(monkeypatch):
    """allocate() passe par _sample_bandit (plus de np.random.beta direct)."""
    src = open("strategies/engine.py", encoding="utf-8").read()
    alloc = src.split("def allocate")[1].split("def ")[0]
    assert "self._sample_bandit(i)" in alloc
    assert "np.random.beta(" not in alloc


# ----------------------------------------------------------------- P1-11 ----
# §2.5 : Sharpe déflaté + plancher d'échantillon sur l'attribution PnL
# ---------------------------------------------------------------------------

def test_dsr_score_separates_signal_from_noise():
    """Série gagnante -> DSR ~ 1 ; série perdante -> DSR ~ 0 ; série à
    moyenne nulle (pas d'edge) -> DSR faible mais non nul ; ordre strict."""
    meta = MetaAllocationEngine(strategies=[BaseStrategy("A"), BaseStrategy("B")])
    win = meta._dsr_score([0.01] * 10)
    loss = meta._dsr_score([-0.01] * 10)
    zero = meta._dsr_score([0.01] * 5 + [-0.01] * 5)  # moyenne nulle
    assert win > 0.99
    assert loss < 0.01
    assert 0.0 < zero < 0.5
    assert win > zero > loss


def test_pnl_attribution_bounded_below_sample_floor():
    """P1-11 : avec un échantillon < plancher (20), le poids ne peut PAS
    atteindre la cible complète : l'ajustement est borné à ±20 % par update."""
    meta = MetaAllocationEngine(strategies=[BaseStrategy("A"), BaseStrategy("B")])
    # 5 trades gagnants pour A, 5 perdants pour B (n=5 < 20 -> plancher actif)
    for _ in range(5):
        meta.update_pnl_attribution("A", 0.01)
        meta.update_pnl_attribution("B", -0.01)
    w = meta.pnl_weights
    # la cible complète (softmax DSR ~1 vs ~0) serait ~0.73 ; le plancher
    # borne le mouvement à +20 % de 0.5 -> 0.6 max
    assert w["A"] <= 0.5 * (1.0 + eng.PNL_MAX_ADJUSTMENT) + 1e-9
    assert w["B"] >= 0.5 * (1.0 - eng.PNL_MAX_ADJUSTMENT) - 1e-9
    # mais l'écart existe déjà (A > B)
    assert w["A"] > w["B"]


def test_pnl_attribution_full_adjustment_above_floor():
    """Avec >= 20 échantillons, l'ajustement est COMPLET (cible atteinte)."""
    meta = MetaAllocationEngine(strategies=[BaseStrategy("A"), BaseStrategy("B")])
    for _ in range(20):
        meta.update_pnl_attribution("A", 0.01)
        meta.update_pnl_attribution("B", -0.01)
    w = meta.pnl_weights
    # softmax sur DSR ~ [1.0, 0.0] -> [0.731, 0.269]
    assert w["A"] == pytest.approx(0.731, abs=0.02)
    assert w["B"] == pytest.approx(0.269, abs=0.02)
    # somme normalisée à 1
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


def test_update_pnl_attribution_uses_deflated_sharpe():
    """Le code d'attribution appelle bien calculate_deflated_sharpe_ratio."""
    src = open("strategies/engine.py", encoding="utf-8").read()
    assert "calculate_deflated_sharpe_ratio" in src


def test_pnl_weights_in_allocate_and_weights():
    """Les poids PnL-DSR sont exposés dans allocate() et combinés dans
    get_strategy_weights (le test existant Alpha>Beta reste vrai)."""
    meta = MetaAllocationEngine(strategies=[BaseStrategy("A"), BaseStrategy("B")])
    for _ in range(10):
        meta.update_pnl_attribution("A", 0.01)
        meta.update_pnl_attribution("B", -0.01)
    gw = meta.get_strategy_weights()
    assert gw["A"] > gw["B"]
    assert gw["A"] > 0.0 and gw["B"] > 0.0
    # les poids pnl sont exposés
    assert set(meta.pnl_weights.keys()) == {"A", "B"}


# ----------------------------------------------------------------- P1-10 ----
# §4.4 : corrélation des signaux inter-stratégies
# ---------------------------------------------------------------------------

def test_signal_corr_needs_sample():
    meta = MetaAllocationEngine(strategies=[BaseStrategy("A"), BaseStrategy("B")])
    assert meta.signal_diversification_weights() == {}


def test_signal_corr_penalizes_redundant_strategies():
    """Deux stratégies parfaitement corrélées -> facteur de diversification
    ~0.5 (pénalité max)."""
    meta = MetaAllocationEngine(strategies=[BaseStrategy("A"), BaseStrategy("B")])
    x = np.sin(np.linspace(0, 10, 40))
    meta.signal_history["A"] = deque(x, maxlen=200)
    meta.signal_history["B"] = deque(x, maxlen=200)
    w = meta.signal_diversification_weights()
    assert w["A"] == pytest.approx(eng.SIGNAL_CORR_MAX_PENALTY, abs=0.05)
    assert w["B"] == pytest.approx(eng.SIGNAL_CORR_MAX_PENALTY, abs=0.05)


def test_signal_corr_spares_uncorrelated_strategies():
    """Deux stratégies décorrélées -> facteur ~1.0 (pas de pénalité)."""
    meta = MetaAllocationEngine(strategies=[BaseStrategy("A"), BaseStrategy("B")])
    meta.signal_history["A"] = deque(np.sin(np.linspace(0, 10, 40)), maxlen=200)
    meta.signal_history["B"] = deque(np.cos(np.linspace(0, 10, 40)), maxlen=200)
    w = meta.signal_diversification_weights()
    assert w["A"] > 0.85 and w["B"] > 0.85


def test_allocate_exposes_and_applies_signal_corr():
    """allocate() expose signal_correlation et applique la pénalité quand
    l'historique de signaux est suffisant."""
    meta = MetaAllocationEngine(strategies=[TrendFollowingStrategy(),
                                            MeanReversionStrategy()])
    df = pd.DataFrame({"close": np.linspace(100, 110, 80),
                       "high": np.linspace(101, 111, 80),
                       "low": np.linspace(99, 109, 80),
                       "volume": [1000] * 80})
    md = {"df": df, "price_primary": 105.0, "price_secondary": 105.0,
          "bids": [[104, 1]], "asks": [[106, 1]], "inventory": 0.0,
          "max_inventory": 100.0, "vpin": 0.5, "kyle_lambda": 0.0001,
          "onchain_risk": 0.2, "sentiment": 0.1}
    res1 = meta.allocate(md, 2, 0.0, 0.0)
    assert res1["signal_correlation"] == {}   # 1 seul échantillon : neutre
    # simule 40 ticks de signaux identiques (stratégies redondantes)
    x = np.sin(np.linspace(0, 10, 40))
    for k in meta.signal_history:
        meta.signal_history[k] = deque(x, maxlen=200)
    res2 = meta.allocate(md, 2, 0.0, 0.0)
    assert res2["signal_correlation"], "la corrélation doit être active"
    assert all(v < 1.0 for v in res2["signal_correlation"].values())
