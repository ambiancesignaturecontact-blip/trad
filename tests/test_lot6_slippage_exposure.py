"""
LOT 6 — P1-13 / P1-14 (audit indépendant §4.6 / §2.7).

P1-13 : le slippage live est estimé par BOOK-WALKING du carnet réel (et non
        plus seulement par le modèle de fills fixes). Le book-walking existait
        déjà dans simulate_paper_fill (_book_walk_price) ; il est maintenant
        branché sur le sizing (tradability) et le SOR.
P1-14 : max_exposure_normal (total) nettement supérieur à max_per_asset_pct
        (par actif) : 75 % total / 25 % par actif — une position ne doit plus
        épuiser tout le budget d'exposition.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- P1-13 ----
# book-walking du slippage
# ---------------------------------------------------------------------------

def test_estimate_slippage_from_thick_book():
    """Carnet épais : le slippage reflète le VWAP des niveaux consommés."""
    from core.paper_execution import estimate_slippage_bps_from_book
    # ask 100$ (1 BTC), puis 101$ (1 BTC) -> acheter 2 BTC => VWAP 100.5
    book = {"bids": [[99.0, 5.0]], "asks": [[100.0, 1.0], [101.0, 1.0]]}
    bps = estimate_slippage_bps_from_book("BUY", 2.0, book, arrival_price=100.0)
    assert bps is not None
    assert bps == pytest.approx(50.0, abs=0.1)  # 0.5$ / 100$ = 50 bps


def test_estimate_slippage_thin_book_returns_bps_anyway():
    """Carnet moins profond que la taille : le VWAP des niveaux disponibles
    est quand même retourné (le fill serait partiel — le bps reste réel)."""
    from core.paper_execution import estimate_slippage_bps_from_book
    book = {"bids": [[99.0, 1.0]], "asks": [[100.0, 0.5], [100.5, 0.5]]}
    bps = estimate_slippage_bps_from_book("BUY", 5.0, book, arrival_price=100.0)
    assert bps is not None and bps > 0


def test_estimate_slippage_no_book_returns_none():
    """Pas de carnet -> None (jamais de slippage inventé)."""
    from core.paper_execution import estimate_slippage_bps_from_book
    assert estimate_slippage_bps_from_book("BUY", 1.0, None, 100.0) is None
    assert estimate_slippage_bps_from_book("BUY", 1.0, {}, 100.0) is None
    assert estimate_slippage_bps_from_book("BUY", 0.0, {"asks": [[100, 1]]}, 100.0) is None


def test_estimate_slippage_implausible_book_returns_none():
    """Carnet d'un autre actif (écart > 10 %) -> None (régression EURUSD)."""
    from core.paper_execution import estimate_slippage_bps_from_book
    # carnet à 60 000 alors que le prix d'arrivée est 100 -> incohérent
    book = {"asks": [[60000.0, 5.0]]}
    assert estimate_slippage_bps_from_book("BUY", 1.0, book, arrival_price=100.0) is None


def test_main_uses_book_walk_for_tradability():
    """main.py : le sizing (tradability) tente le book-walking AVANT le
    modèle de fills, et expose book_slippage_bps."""
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "estimate_slippage_bps_from_book" in src
    assert "book_slippage_bps" in src
    # l'import est présent (l'ordre des noms peut varier après tri ruff I001)
    assert "from core.paper_execution import" in src
    assert "estimate_slippage_bps_from_book" in src.split("from core.paper_execution import")[1][:120]
    assert "simulate_paper_fill" in src.split("from core.paper_execution import")[1][:120]


def test_sor_accepts_qty_and_book_walks_per_venue():
    """pick_best_venue_net accepte une qty et book-walk le carnet par venue."""
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "async def pick_best_venue_net(symbol: str, side: str, qty: float = None)" in src
    # l'appel du site de trade passe la quantité réelle
    assert "pick_best_venue_net(symbol, side, qty=trade_qty_formatted)" in src
    assert "estimate_slippage_bps_from_book(side, qty, _book" in src


# ---------------------------------------------------------------- P1-14 ----
# cohérence max_exposure_normal vs max_per_asset_pct
# ---------------------------------------------------------------------------

def test_config_defaults_total_vs_per_asset():
    """P1-14 : total (0.75) >> par actif (0.25) — 3 positions possibles."""
    from core.config import settings
    total = settings.get_float("risk", "max_exposure_normal", 0.25)
    per_asset = settings.get_float("risk", "max_per_asset_pct", 0.25)
    assert total >= 0.70, f"exposition totale {total} < 70% (recommandation audit)"
    assert total >= per_asset * 2.5, \
        f"total {total} pas nettement supérieur au par-actif {per_asset}"


def test_portfolio_allocator_applies_total_cap():
    """Le plafond total explicite (0.75) est appliqué dans le facteur
    d'exposition (le plus strict de cash_reserve / max_exposure_normal)."""
    from core.portfolio_allocator import PortfolioAllocator
    pa = PortfolioAllocator()
    # 80 % investi : au-dessus du plafond total 75 % -> plus de nouveau trade
    state = {"cached_positions": [{"symbol": "BTCUSDT", "qty": 1.0}],
             "assets": {"BTCUSDT": {"price": 80000.0}},
             "last_known_prices": {}, "current_equity": 100000.0}
    assert pa.portfolio_exposure_factor(state, "balance_demo") == 0.0
    # 60 % investi : sous le plafond -> facteur positif
    state2 = {"cached_positions": [{"symbol": "BTCUSDT", "qty": 1.0}],
              "assets": {"BTCUSDT": {"price": 60000.0}},
              "last_known_prices": {}, "current_equity": 100000.0}
    f = pa.portfolio_exposure_factor(state2, "balance_demo")
    assert 0.0 < f <= 1.0


def test_existing_exposure_factor_test_still_valid():
    """Le test préexistant (84 % investi -> facteur < 1) reste vert : 84 %
    dépasse le nouveau plafond 75 % -> 0.0, toujours dans [0, 1)."""
    from core.portfolio_allocator import PortfolioAllocator
    pa = PortfolioAllocator()
    state2 = {"cached_positions": [{"symbol": "BTCUSDT", "qty": 1.5}],
              "assets": {"BTCUSDT": {"price": 56000.0}},
              "last_known_prices": {}, "current_equity": 100000.0}
    f = pa.portfolio_exposure_factor(state2, "balance_demo")
    assert 0.0 <= f < 1.0


def test_config_yaml_documents_new_default():
    cfg = (ROOT / "config.yaml").read_text(encoding="utf-8")
    assert "max_exposure_normal: 0.75" in cfg
