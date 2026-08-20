"""
P2-multi-devise : devise de compte configurable + conversion réelle.

Principe institutionnel (base currency) : le portefeuille multi-actifs se
mesure dans UNE devise de référence (celle du compte). L'interne reste en USD
(actifs cotés en USD) ; l'affichage (balance, équité, PnL, min notional) est
converti dans la devise du compte via des taux RÉELS (open.er-api.com) —
jamais de taux inventé (mandat : données 100% réelles).
"""

import pytest

# --------------------------------------------------------------------------- #
# 1. Moteur FX (core/fx.py)
# --------------------------------------------------------------------------- #

def test_default_account_currency_is_usd(monkeypatch):
    from core.fx import get_account_currency
    monkeypatch.delenv("ACCOUNT_CURRENCY", raising=False)
    assert get_account_currency() == "USD"


def test_account_currency_from_env(monkeypatch):
    from core.fx import get_account_currency
    monkeypatch.setenv("ACCOUNT_CURRENCY", "EUR")
    assert get_account_currency() == "EUR"


def test_account_currency_invalid_falls_back(monkeypatch):
    from core.fx import get_account_currency
    monkeypatch.setenv("ACCOUNT_CURRENCY", "XYZ")
    assert get_account_currency() == "USD"


def test_convert_usd_to_eur_with_real_rate(monkeypatch):
    """Conversion avec un taux RÉEL (mocké ici pour le test unitaire —
    la source réelle est testée séparément dans test_live)."""
    from core import fx
    monkeypatch.setattr(fx, "fetch_usd_rates", lambda: {"EUR": 0.85, "USD": 1.0})
    assert fx.convert(1000.0, "EUR") == pytest.approx(850.0)
    assert fx.convert(1000.0, "USD") == 1000.0


def test_convert_returns_none_when_fx_unavailable(monkeypatch):
    """Source FX indisponible -> None (jamais de taux inventé)."""
    from core import fx
    monkeypatch.setattr(fx, "fetch_usd_rates", lambda: None)
    assert fx.convert(1000.0, "EUR") is None


def test_display_honest_when_fx_unavailable(monkeypatch):
    from core import fx
    monkeypatch.setattr(fx, "fetch_usd_rates", lambda: None)
    d = fx.display(1000.0, "EUR")
    assert d["currency"] == "USD"
    assert d["fx_rate"] is None
    assert "FX indisponible" in d["note"]


def test_fetch_usd_rates_real_source():
    """La source réelle open.er-api.com fonctionne (taux USD de base)."""
    from core.fx import fetch_usd_rates
    rates = fetch_usd_rates()
    assert rates is not None
    assert rates.get("USD") == 1.0
    for c in ("EUR", "GBP", "JPY"):
        assert c in rates and rates[c] > 0


# --------------------------------------------------------------------------- #
# 2. Télémétrie multi-devise
# --------------------------------------------------------------------------- #

def test_telemetry_exposes_account_currency(monkeypatch):
    """La télémétrie expose la devise du compte + les valeurs converties."""
    monkeypatch.setenv("ACCOUNT_CURRENCY", "EUR")
    # recharger la devise dans le module télémétrie (résolue à l'import)
    import importlib

    import telemetry
    importlib.reload(telemetry)
    tel = telemetry.compile_telemetry_data()
    assert tel["account_currency"] == "EUR"
    assert tel["fx_rate_usd_to_account"] is not None and tel["fx_rate_usd_to_account"] > 0
    bal = tel["balance_account_ccy"]
    assert bal["currency"] == "EUR"
    # la valeur convertie est cohérente avec le taux (à 1% près)
    assert bal["value"] == pytest.approx(tel["balance"] * tel["fx_rate_usd_to_account"], rel=0.01)


def test_telemetry_usd_when_currency_usd():
    """Devise USD : pas de conversion, fx_rate=1.0."""
    import telemetry
    tel = telemetry.compile_telemetry_data()
    if tel["account_currency"] == "USD":
        assert tel["balance_account_ccy"]["currency"] == "USD"
        assert tel["balance_account_ccy"]["value"] == pytest.approx(tel["balance"])


# --------------------------------------------------------------------------- #
# 3. UI : la devise est affichée (mini-app et dashboard)
# --------------------------------------------------------------------------- #

def test_miniapp_shows_account_currency():
    src = open("templates/telegram_mini_app.html").read()
    assert "account-ccy" in src
    assert "account_currency" in src
    assert "balance_account_ccy" in src
    assert "Devise: USD" in src  # défaut HTML honnête


def test_dashboard_shows_account_currency():
    src = open("templates/dashboard.html").read()
    assert "account_currency" in src
    assert "balance_account_ccy" in src
    assert "equity_account_ccy" in src
