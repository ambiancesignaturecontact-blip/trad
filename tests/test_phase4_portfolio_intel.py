"""
PHASE 4 — P4-A : tests du module Portfolio Intelligence
(core/portfolio_intel.py — gate d'exposition factorielle).

Couvre : bêta causal, exposition du portefeuille, décision PURE (facteur BTC
et concentration corrélée), absence de blocage sans mesure, câblage main.py,
bucket NO_TRADE, config.
"""
from pathlib import Path

import numpy as np
import pandas as pd


class TestBetaCausal:
    def test_beta_positive_on_correlated_series(self):
        from core.portfolio_intel import estimate_beta
        rng = np.random.default_rng(3)
        btc = pd.Series(rng.normal(0, 0.01, 1000))
        # ETH = 1.5 × BTC + bruit (même index)
        eth = pd.Series(1.5 * btc.values + rng.normal(0, 0.002, 1000),
                        index=btc.index)
        beta = estimate_beta(eth, btc)
        assert beta is not None
        assert abs(beta - 1.5) < 0.3

    def test_beta_none_on_short_sample(self):
        from core.portfolio_intel import estimate_beta
        btc = pd.Series(np.random.default_rng(1).normal(0, 0.01, 50))
        assert estimate_beta(btc, btc) is None   # < BETA_MIN_SAMPLES

    def test_beta_none_when_btc_flat(self):
        from core.portfolio_intel import estimate_beta
        idx = pd.RangeIndex(500)
        btc = pd.Series(100.0, index=idx)
        asset = pd.Series(np.random.default_rng(2).normal(0, 0.01, 500),
                          index=idx)
        assert estimate_beta(asset, btc) is None  # variance BTC nulle


class TestExposureGate:
    def _positions(self):
        return [{"symbol": "BTCUSDT", "qty": 0.5, "price": 60000.0}]

    def test_blocks_when_btc_factor_over_limit(self):
        from core.portfolio_intel import exposure_gate_blocks
        betas = {"BTCUSDT": 1.0, "ETHUSDT": 1.5}
        # équité 100k ; position BTC = 0.5×60k×1.0 = 30k (30 %)
        # candidat ETH : 2×4000×1.5 = 12k -> net 42 % < 50 % (OK)
        blk, _, _ = exposure_gate_blocks("ETHUSDT", "BUY", 2.0, 4000.0,
                                         100000.0, betas, self._positions(),
                                         max_btc_beta_pct=50.0)
        assert blk is False
        # candidat ETH 10×4000×1.5 = 60k -> net 90 % > 50 % (BLOQUÉ)
        blk, reason, detail = exposure_gate_blocks(
            "ETHUSDT", "BUY", 10.0, 4000.0, 100000.0, betas,
            self._positions(), max_btc_beta_pct=50.0)
        assert blk is True
        assert "portfolio_exposure" in reason
        assert "déjà trop exposé" in detail

    def test_short_reduces_exposure(self):
        """Un SHORT ETH réduit l'exposition nette au facteur BTC : autorisé."""
        from core.portfolio_intel import exposure_gate_blocks
        betas = {"BTCUSDT": 1.0, "ETHUSDT": 1.5}
        blk, _, _ = exposure_gate_blocks("ETHUSDT", "SELL", 10.0, 4000.0,
                                         100000.0, betas, self._positions())
        assert blk is False

    def test_blocks_on_correlated_concentration(self):
        from core.portfolio_intel import exposure_gate_blocks
        # position BTC longue + candidat ETH corrélé à 0.95 (même sens)
        corr = {"ETHUSDT": {"BTCUSDT": 0.95}}
        blk, reason, detail = exposure_gate_blocks(
            "ETHUSDT", "BUY", 1.0, 4000.0, 100000.0, {},
            self._positions(), correlations=corr)
        assert blk is True
        assert "concentration corrélée" in reason

    def test_opposite_direction_not_blocked(self):
        """Candidat SHORT vs position longue : pas de concentration même sens."""
        from core.portfolio_intel import exposure_gate_blocks
        corr = {"ETHUSDT": {"BTCUSDT": 0.95}}
        blk, _, _ = exposure_gate_blocks("ETHUSDT", "SELL", 1.0, 4000.0,
                                         100000.0, {}, self._positions(),
                                         correlations=corr)
        assert blk is False

    def test_no_measure_no_block(self):
        from core.portfolio_intel import exposure_gate_blocks
        blk, reason, _ = exposure_gate_blocks("XAUUSD", "BUY", 1.0, 2000.0,
                                              100000.0, {}, [])
        assert blk is False and reason is None


class TestWiring:
    def test_gate_after_friction_before_journal(self):
        src = (Path(__file__).parent.parent / "main.py").read_text(encoding="utf-8")
        i_fric = src.find("GATE DE FRICTION")
        i_px = src.find("GATE D'EXPOSITION FACTORIELLE")
        i_journal = src.find("_dj_id = journal_decision(")
        assert -1 not in (i_fric, i_px, i_journal)
        assert i_fric < i_px < i_journal
        assert "exposure_gate_blocks(" in src

    def test_bucket_portfolio_exposure(self):
        from core.meta_cognition import _no_trade_bucket
        assert _no_trade_bucket("portfolio_exposure: facteur BTC") == \
            "portfolio_exposure"

    def test_config_default(self):
        import yaml
        cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml",
                                  encoding="utf-8"))
        assert cfg["portfolio"]["max_btc_beta_exposure_pct"] == 50.0
        from core.config import settings
        assert settings.get_float("portfolio", "max_btc_beta_exposure_pct",
                                  50.0) == 50.0
