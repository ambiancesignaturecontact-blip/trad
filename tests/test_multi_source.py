"""
LOT 1bis — Tests du moteur multi-sources (PDF : « redondance & fiabilité »).

Chaque catégorie critique (prix, funding) doit avoir AU MOINS 2 sources
indépendantes croisées. Divergence anormale -> DIVERGENT -> gel du trading.
Aucun prix fabriqué : source down = None, consensus UNAVAILABLE sinon.
"""
import asyncio

import pytest


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, *a, **k):
        return FakeResponse(self._payload)


# --------------------------------------------------------------------------- #
# CONSENSUS PRIX
# --------------------------------------------------------------------------- #
class TestPriceConsensus:
    @pytest.mark.asyncio
    async def test_consensus_ok_multiple_sources(self):
        from market_data.multi_source import MultiSourcePriceEngine
        eng = MultiSourcePriceEngine()

        async def fake(name, price):
            async def f(symbol):
                return price
            return f

        eng._fetch_binance = await fake("binance", 64500.0)
        eng._fetch_bybit = await fake("bybit", 64510.0)
        eng._fetch_coinbase = await fake("coinbase", 64505.0)
        eng._fetch_kraken = await fake("kraken", 64520.0)
        eng._fetch_okx = await fake("okx", 64512.0)
        eng._fetch_coingecko = await fake("coingecko", 64500.0)
        eng._fetch_cryptocompare = await fake("cryptocompare", 64508.0)

        cons = await eng.get_consensus("BTCUSDT")
        assert cons["status"] == "OK"
        assert cons["n_sources"] >= 5
        assert cons["price"] == pytest.approx(64510.0, rel=1e-3)
        assert cons["divergence_pct"] < cons["threshold_pct"]

    @pytest.mark.asyncio
    async def test_outlier_ignored_by_median(self):
        from market_data.multi_source import MultiSourcePriceEngine
        eng = MultiSourcePriceEngine()

        async def fake(price):
            async def f(symbol):
                return price
            return f

        eng._fetch_binance = await fake(64500.0)
        eng._fetch_bybit = await fake(64510.0)
        eng._fetch_coinbase = await fake(70000.0)  # outlier massif
        eng._fetch_kraken = await fake(64520.0)
        eng._fetch_okx = await fake(64505.0)
        # la médiane (64510) n'est PAS affectée par l'outlier
        cons = await eng.get_consensus("BTCUSDT")
        assert cons["price"] == pytest.approx(64510.0, rel=1e-3)
        # mais la divergence est énorme -> DIVERGENT (gel)
        assert cons["status"] == "DIVERGENT"

    @pytest.mark.asyncio
    async def test_single_source_honest(self):
        from market_data.multi_source import MultiSourcePriceEngine
        eng = MultiSourcePriceEngine()

        async def f(symbol):
            return 64500.0

        eng._fetch_binance = f
        for m in ("_fetch_bybit", "_fetch_coinbase", "_fetch_kraken",
                  "_fetch_okx", "_fetch_coingecko", "_fetch_cryptocompare"):
            async def none_f(symbol):
                return None
            setattr(eng, m, none_f)

        cons = await eng.get_consensus("BTCUSDT")
        assert cons["status"] == "SINGLE_SOURCE"
        assert cons["n_sources"] == 1
        assert cons["price"] == 64500.0

    @pytest.mark.asyncio
    async def test_unavailable_when_all_sources_down(self):
        from market_data.multi_source import MultiSourcePriceEngine
        eng = MultiSourcePriceEngine()

        async def none_f(symbol):
            return None

        for m in ("_fetch_binance", "_fetch_bybit", "_fetch_coinbase",
                  "_fetch_kraken", "_fetch_okx", "_fetch_coingecko",
                  "_fetch_cryptocompare", "_fetch_yahoo", "_fetch_goldapi",
                  "_fetch_erapi"):
            setattr(eng, m, none_f)

        cons = await eng.get_consensus("BTCUSDT")
        assert cons["status"] == "UNAVAILABLE"
        assert cons["price"] is None
        cons2 = await eng.get_consensus("EURUSD")
        assert cons2["status"] == "UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_trad_assets_two_sources(self):
        """XAUUSD : Yahoo + gold-api ; EURUSD : Yahoo + er-api."""
        from market_data.multi_source import MultiSourcePriceEngine
        eng = MultiSourcePriceEngine()

        async def fake_yahoo(symbol):
            return {"XAUUSD": 4300.0, "EURUSD": 1.10}.get(symbol)

        async def fake_gold(symbol):
            return 4310.0 if symbol == "XAUUSD" else None

        async def fake_er(symbol):
            return 1.105 if symbol == "EURUSD" else None

        eng._fetch_yahoo = fake_yahoo
        eng._fetch_goldapi = fake_gold
        eng._fetch_erapi = fake_er

        cons_xau = await eng.get_consensus("XAUUSD")
        assert cons_xau["status"] == "OK"
        assert cons_xau["n_sources"] == 2

        cons_eur = await eng.get_consensus("EURUSD")
        assert cons_eur["status"] == "OK"
        assert cons_eur["n_sources"] == 2

    @pytest.mark.asyncio
    async def test_erapi_inverts_rate(self):
        """
        open.er-api.com renvoie rates.EUR = EUR pour 1 USD (~0.86).
        EURUSD doit être 1/0.86 ≈ 1.16 — sinon divergence artificielle.
        """
        from market_data.multi_source import MultiSourcePriceEngine
        eng = MultiSourcePriceEngine()

        class FakeClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def get(self, *a, **k):
                return FakeResponse({"rates": {"EUR": 0.8635}})

        import market_data.multi_source as ms
        ms.httpx.AsyncClient = lambda *a, **k: FakeClient()
        try:
            price = await eng._fetch_erapi("EURUSD")
            assert price == pytest.approx(1.0 / 0.8635, rel=1e-6)
            assert price > 1.0  # EURUSD doit être ~1.16, pas 0.86
        finally:
            import httpx as real_httpx
            ms.httpx = real_httpx


# --------------------------------------------------------------------------- #
# FUNDING CROISÉ
# --------------------------------------------------------------------------- #
class TestFundingConsensus:
    @pytest.mark.asyncio
    async def test_funding_cross_check_ok(self):
        from market_data.multi_source import MultiSourcePriceEngine
        eng = MultiSourcePriceEngine()

        async def f_bin(symbol):
            return 0.0001
        async def f_byb(symbol):
            return 0.00012

        eng._fetch_binance_funding = f_bin
        eng._fetch_bybit_funding = f_byb
        res = await eng.get_funding_consensus("BTCUSDT", max_age_seconds=0)
        assert res["status"] == "OK"
        assert res["funding_rate_8h"] == pytest.approx(0.00011)

    @pytest.mark.asyncio
    async def test_funding_divergent_ignored(self):
        from market_data.multi_source import MultiSourcePriceEngine
        eng = MultiSourcePriceEngine()

        async def f_bin(symbol):
            return 0.0001
        async def f_byb(symbol):
            return 0.0009  # 9x plus grand -> divergence anormale

        eng._fetch_binance_funding = f_bin
        eng._fetch_bybit_funding = f_byb
        res = await eng.get_funding_consensus("BTCUSDT", max_age_seconds=0)
        assert res["status"] == "DIVERGENT"
        assert res["funding_rate_8h"] is None  # jamais de valeur non corroborée

    @pytest.mark.asyncio
    async def test_funding_single_source(self):
        from market_data.multi_source import MultiSourcePriceEngine
        eng = MultiSourcePriceEngine()

        async def f_bin(symbol):
            return 0.0001
        async def none_f(symbol):
            return None

        eng._fetch_binance_funding = f_bin
        eng._fetch_bybit_funding = none_f
        res = await eng.get_funding_consensus("BTCUSDT", max_age_seconds=0)
        assert res["status"] == "SINGLE_SOURCE"
        assert res["funding_rate_8h"] == 0.0001


# --------------------------------------------------------------------------- #
# INTÉGRATION MAIN : carnet multi-exchange BBO + flags divergence
# --------------------------------------------------------------------------- #
class TestMainIntegration:
    def test_exchange_order_books_bbo(self):
        import main
        # Bybit : spread 1.0 ; Binance : spread 0.5 (meilleur) -> BBO Binance
        main.update_asset_order_book("BTCUSDT",
                                     [[64500.0, 1.0]], [[64501.0, 1.0]],
                                     exchange="bybit")
        main.update_asset_order_book("BTCUSDT",
                                     [[64499.5, 2.0]], [[64500.0, 2.0]],
                                     exchange="binance")
        assert "binance" in main.STATE["exchange_order_books"]
        assert main.STATE["order_books"]["BTCUSDT"]["exchange"] == "binance"
        assert main.STATE["order_books"]["BTCUSDT"]["bids"][0][0] == 64499.5
        # Alias historique
        assert main.STATE["order_book"]["bids"][0][0] == 64499.5

    def test_price_divergent_flag_blocks_real(self):
        """Le flag price_divergent fait échouer la REAL safety gate."""
        import main
        main.STATE.setdefault("price_divergent", {})["BTCUSDT"] = True
        main.STATE["price_consensus"] = {"BTCUSDT": {
            "divergence_pct": 2.5, "threshold_pct": 0.3}}
        # La gate échoue à cause de la divergence (avant même le client CCXT)
        ok = main.evaluate_real_safety_gate("BTCUSDT")
        assert ok is False
