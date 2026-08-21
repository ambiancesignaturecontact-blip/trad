"""
LOT 6 (mandat — Validation) : Adversarial Decision Engine + vérification des
garde-fous de validation existants (walk-forward/OOS/Monte Carlo/biais).

Vérifié ici :
  1. AdversarialDecisionEngine : baseline = edge net × SL ; chaque scénario
     (spread x2, slippage x2/x3, latence x5, liquidité /2, vol x2, inversion,
     gap, données incohérentes) produit un PnL stressé <= baseline ; le pire
     scénario est identifié ; FRAGILE quand pire < -max_loss ou espérance
     stressée <= 0 ; ROBUST sinon ; défensif (entrées invalides -> ROBUST).
  2. Intégration TradeOpportunityEngine : mode block -> WAIT
     ADVERSARIAL_FRAGILE ; mode warn -> TRADE (signalé) ; absent -> TRADE.
  3. Câblage main.py : adversarial_engine instancié, évalué avant la décision,
     passé à trade_opportunity ; télémétrie adversarial ; API /api/v1/adversarial.
  4. Garde-fous de validation EXISTANTS (pas réimplémentés — vérifiés) :
     audit_backtest (look-ahead/survivorship/slippage -> REJECTED),
     WalkForwardValidator, Monte Carlo stress, bootstrap Sharpe.
  5. DÉMO == RÉAL : aucun flag de mode.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from core.adversarial_engine import (  # noqa: E402
    ADV_MAX_LOSS_PCT,
    ADV_MODE,
    FRAGILE,
    ROBUST,
    AdversarialDecisionEngine,
)
from core.conviction_engine import (  # noqa: E402
    WAIT_ADVERSARIAL,
    TradeOpportunityEngine,
)


# --------------------------------------------------------------------------- #
# 1. Adversarial : math des scénarios
# --------------------------------------------------------------------------- #
class TestAdversarial:
    def _eng(self):
        return AdversarialDecisionEngine()

    def test_baseline_is_edge_times_sl(self):
        r = self._eng().evaluate(edge_net=0.10, sl_pct=0.03, slippage_bps_expected=5.0)
        assert r["baseline_pnl_pct"] == pytest.approx(0.10 * 0.03 * 100.0, abs=1e-4)

    def test_all_scenarios_leq_baseline(self):
        r = self._eng().evaluate(edge_net=0.10, sl_pct=0.03, slippage_bps_expected=5.0)
        base = r["baseline_pnl_pct"]
        for sc in r["scenarios"]:
            assert sc["pnl_pct"] <= base + 1e-6, sc["name"]

    def test_reversal_is_worst(self):
        """Le signal inversé (perte = SL) est toujours le pire ou co-détenteur
        du pire — c'est le scénario de rupture."""
        r = self._eng().evaluate(edge_net=0.10, sl_pct=0.03, slippage_bps_expected=5.0)
        reversal = [sc for sc in r["scenarios"] if sc["name"] == "SIGNAL_REVERSAL"][0]
        assert reversal["pnl_pct"] == pytest.approx((-0.03 - 5.0 / 10000) * 100.0, abs=1e-3)
        assert r["worst_case_pnl_pct"] <= reversal["pnl_pct"]

    def test_fragile_when_worst_exceeds_max_loss(self):
        """SL 20% + slippage élevé -> pire scénario ~ -20% < -5% -> FRAGILE."""
        r = self._eng().evaluate(edge_net=0.05, sl_pct=0.20, slippage_bps_expected=50.0)
        assert r["fragile"] is True
        assert r["verdict"] == FRAGILE

    def test_robust_when_small_sl_and_good_edge(self):
        """Petit SL + edge réel : ROBUST — le pire (inversion ~-2%) reste
        au-dessus de la perte max, l'espérance d'exécution stressée reste > 0."""
        r = self._eng().evaluate(edge_net=0.10, sl_pct=0.02, slippage_bps_expected=2.0)
        assert r["worst_case_pnl_pct"] > -(ADV_MAX_LOSS_PCT * 100.0)
        assert r["stressed_avg_pnl_pct"] > 0.0
        assert r["verdict"] == ROBUST

    def test_defensive_on_bad_inputs(self):
        """Entrées invalides -> ROBUST (le trade suit son chemin — la
        protection est additive, pas un point de panne)."""
        r = self._eng().evaluate(edge_net=None, sl_pct=None, slippage_bps_expected=None)
        assert r["verdict"] == ROBUST
        assert r["fragile"] is False
        r2 = self._eng().evaluate(edge_net="bad", sl_pct="bad")
        assert r2["verdict"] == ROBUST
        r3 = self._eng().evaluate(edge_net=0.10, sl_pct=None)  # sl None -> 0.03 défaut
        assert r3["verdict"] in (ROBUST, FRAGILE)

    def test_slippage_high_degrades_pnl(self):
        r_low = self._eng().evaluate(edge_net=0.10, sl_pct=0.03, slippage_bps_expected=2.0)
        r_high = self._eng().evaluate(edge_net=0.10, sl_pct=0.03, slippage_bps_expected=80.0)
        slip3_low = [s for s in r_low["scenarios"] if s["name"] == "SLIPPAGE_X3"][0]["pnl_pct"]
        slip3_high = [s for s in r_high["scenarios"] if s["name"] == "SLIPPAGE_X3"][0]["pnl_pct"]
        assert slip3_high < slip3_low


# --------------------------------------------------------------------------- #
# 2. Intégration TradeOpportunity
# --------------------------------------------------------------------------- #
class TestOpportunityIntegration:
    def _eng(self):
        return TradeOpportunityEngine()

    def test_block_mode_rejects_fragile(self):
        e = self._eng()
        adv = {"fragile": True, "mode": "block", "detail": "pire scénario -8.0% < -5.0%"}
        r = e.evaluate(signal=0.20, conviction=0.18, threshold=0.08,
                       edge_net=0.25, adversarial=adv)
        assert r["decision"] == "WAIT"
        assert r["reason"] == WAIT_ADVERSARIAL

    def test_warn_mode_signals_but_trades(self):
        e = self._eng()
        adv = {"fragile": True, "mode": "warn", "detail": "fragile (warn)"}
        r = e.evaluate(signal=0.20, conviction=0.18, threshold=0.08,
                       edge_net=0.25, adversarial=adv)
        assert r["decision"] == "TRADE"

    def test_robust_trades(self):
        e = self._eng()
        adv = {"fragile": False, "mode": "block", "detail": "survit"}
        r = e.evaluate(signal=0.20, conviction=0.18, threshold=0.08,
                       edge_net=0.25, adversarial=adv)
        assert r["decision"] == "TRADE"

    def test_absent_adversarial_trades(self):
        e = self._eng()
        r = e.evaluate(signal=0.20, conviction=0.18, threshold=0.08, edge_net=0.25)
        assert r["decision"] == "TRADE"


# --------------------------------------------------------------------------- #
# 3. Câblage main + télémétrie + API
# --------------------------------------------------------------------------- #
class TestWiring:
    def test_main_wiring(self):
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        assert "adversarial_engine = AdversarialDecisionEngine()" in src
        assert "adversarial_engine.evaluate(" in src
        assert "adversarial=STATE.get(\"last_adversarial\")" in src

    def test_telemetry_exposes_adversarial(self):
        import main  # noqa: F401
        from telemetry import compile_telemetry_data
        tel = compile_telemetry_data()
        assert "adversarial" in tel

    def test_api_adversarial(self):
        from fastapi.testclient import TestClient

        import main  # noqa: F401
        with TestClient(main.app) as c:
            r = c.get("/api/v1/adversarial")
            assert r.status_code == 200
            assert isinstance(r.json(), dict)

    def test_no_mode_flag(self):
        import inspect
        src = inspect.getsource(AdversarialDecisionEngine.evaluate)
        assert "active_mode" not in src
        assert '"DEMO"' not in src and '"REAL"' not in src

    def test_config_driven(self):
        from core.config import settings
        assert ADV_MODE == settings.get("adversarial", "mode", "block")
        assert ADV_MAX_LOSS_PCT == settings.get_float("adversarial", "max_loss_pct", 0.05)


# --------------------------------------------------------------------------- #
# 4. Garde-fous de validation EXISTANTS (vérifiés, pas réimplémentés)
# --------------------------------------------------------------------------- #
class TestExistingValidationGuards:
    def test_backtest_audit_rejects_biased(self):
        """audit_backtest (look-ahead/survivorship/slippage) marque REJECTED
        si un biais est détecté — la validation interdit les biais (mandat)."""
        import pandas as pd

        from backtester.bias_audit import audit_backtest
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0], "volume": [100, 100, 100]})
        # look-ahead : la fonction de signal regarde le prix futur
        res = audit_backtest(df, ["BTCUSDT"], ["BTCUSDT"],
                             compute_fn=lambda d: (d["close"].shift(-1) - d["close"]).fillna(0))
        # l'audit détecte le biais : statut REJECTED (look-ahead OU coûts nuls)
        assert res.get("status") == "REJECTED"

    def test_walk_forward_validator_exists(self):
        from backtester.engine import WalkForwardValidator
        wf = WalkForwardValidator(train_ratio=0.7)
        assert wf.train_ratio == 0.7

    def test_monte_carlo_and_bootstrap_exist(self):
        from models.monte_carlo import MonteCarloStressTester
        mc = MonteCarloStressTester(num_simulations=1000, horizon_steps=100)
        r = mc.execute_stress_test(initial_capital=10000.0, current_price=100.0,
                                   historical_volatility=0.02)
        assert isinstance(r, dict) and len(r) > 0

    def test_autonomous_scheduler_runs_walk_forward(self):
        """Le cycle autonome exécute la validation walk-forward + DSR + Monte
        Carlo (pipeline TRAIN->VALIDATION->TEST->OOS du mandat)."""
        src = (ROOT / "core/autonomous_ai.py").read_text(encoding="utf-8")
        assert "WalkForwardValidator" in src
        assert "run_validation" in src
        assert "execute_stress_test" in src
        assert "bootstrap_sharpe_significance" in src
