"""
LOT 8 — Tests « Rigueur de recherche, coûts réels, attribution »
(PROMPT MAÎTRE : Piliers N, O, Q).

Vérifie :
 1. Audit des biais de backtest (look-ahead, survivorship, slippage)
 2. Stress par scénarios de crises RÉELLES (COVID 2020, krach 2018, FTX 2022)
    sur le portefeuille complet
 3. Monte Carlo bootstrap : le Sharpe observé est-il dû à la chance ?
 4. Coût total par trade (frais + slippage + impact + gas + funding) + portage
 5. PnL NET (coûts retranchés du PnL affiché)
 6. Attribution complète par facteur / régime / actif / stratégie
 7. Métriques : Sharpe, Sortino, Calmar, maxDD, win rate, profit factor, expectancy
 8. Intégration : télémétrie + rapport quotidien + endpoints
"""
import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# 1. AUDIT DES BIAIS
# --------------------------------------------------------------------------- #
class TestBiasAudit:
    def test_slippage_bias_zero_rejected(self):
        from backtester.bias_audit import audit_slippage_bias
        # slippage 0 et frais 0 -> REJETÉ (coûts sous-estimés)
        res = audit_slippage_bias(slippage_bps=0.0, commission_pct=0.0)
        assert res["ok"] is False
        assert len(res["issues"]) >= 2
        # slippage 5 bps + frais 0.1% -> OK
        res2 = audit_slippage_bias(slippage_bps=5.0, commission_pct=0.001)
        assert res2["ok"] is True

    def test_lookahead_order_check(self):
        from backtester.bias_audit import audit_lookahead_bias
        # DataFrame non trié chronologiquement -> détecté
        # (>= 10 lignes pour passer le garde d'échantillon)
        vals = list(range(1, 13))
        idx = pd.to_datetime([f"2026-01-{d:02d}" for d in [3, 1, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12]])
        df = pd.DataFrame({"close": vals}, index=idx)
        res = audit_lookahead_bias(df)
        assert res["ok"] is False
        assert any("non trié" in i for i in res["issues"])

    def test_survivorship(self):
        from backtester.bias_audit import audit_survivorship_bias
        res = audit_survivorship_bias(["BTC", "ETH", "SOL"], ["BTC"])
        assert res["ok"] is False
        assert "SOL" in res["issues"][0] or "ETH" in res["issues"][0]

    def test_full_audit(self):
        from backtester.bias_audit import audit_backtest
        idx = pd.date_range("2026-01-01", periods=50, freq="h")
        df = pd.DataFrame({"close": np.linspace(100, 110, 50)}, index=idx)
        res = audit_backtest(df, ["BTCUSDT"], ["BTCUSDT"],
                             slippage_bps=5.0, commission_pct=0.001)
        assert res["status"] == "PASSED"
        assert res["score"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 2. STRESS PAR CRISES RÉELLES
# --------------------------------------------------------------------------- #
class TestScenarioStress:
    def test_crisis_scenarios_defined(self):
        from models.scenario_stress import CRISIS_SCENARIOS
        assert "COVID_2020" in CRISIS_SCENARIOS
        assert "CRYPTO_CRASH_2018" in CRISIS_SCENARIOS
        assert "FTX_COLLAPSE_2022" in CRISIS_SCENARIOS

    def test_stress_run(self):
        from models.scenario_stress import ScenarioStressTester
        st = ScenarioStressTester(max_loss_pct=0.15)
        positions = [{"symbol": "BTCUSDT", "qty": 1.0},
                     {"symbol": "AAPL", "qty": 100.0}]
        prices = {"BTCUSDT": 60000.0, "AAPL": 200.0}
        cash = 20000.0
        res = st.run_stress(positions, cash, prices)
        assert res["status"] in ("PASSED", "VULNERABLE")
        assert "COVID_2020" in res["scenarios"]
        # Le portefeuille vaut 60000 + 20000 + 20000 = 100000
        covid = res["scenarios"]["COVID_2020"]
        assert covid["portfolio_value"] == pytest.approx(100000.0)
        # BTC -50% et AAPL -30% -> perte attendue
        assert covid["loss_pct"] < 0

    def test_stress_without_prices_is_honest(self):
        from models.scenario_stress import ScenarioStressTester
        st = ScenarioStressTester()
        res = st.run_stress([{"symbol": "BTCUSDT", "qty": 1.0}], 10000.0, {})
        # sans prix réel -> seule le cash est compté, aucune perte fabriquée
        assert res["scenarios"]["COVID_2020"]["loss_pct"] == 0.0


# --------------------------------------------------------------------------- #
# 3. BOOTSTRAP DU SHARPE
# --------------------------------------------------------------------------- #
class TestBootstrapSharpe:
    def test_significant_edge(self):
        from models.monte_carlo import MonteCarloStressTester
        mc = MonteCarloStressTester()
        rng = np.random.RandomState(0)
        # rendements avec un vrai drift positif -> Sharpe significatif
        eq = np.cumsum(rng.normal(0.001, 0.01, 100)) + 100.0
        res = mc.bootstrap_sharpe_significance(list(eq), n_permutations=200, seed=42)
        assert res["significant"] is True
        assert res["p_value"] < 0.05

    def test_insufficient_data(self):
        from models.monte_carlo import MonteCarloStressTester
        mc = MonteCarloStressTester()
        res = mc.bootstrap_sharpe_significance([100.0, 100.1], n_permutations=10)
        assert res["significant"] is None


# --------------------------------------------------------------------------- #
# 4. COÛTS RÉELS
# --------------------------------------------------------------------------- #
class TestCostAccounting:
    def test_trade_cost_total(self):
        from core.cost_accounting import CostAccounting
        ca = CostAccounting()
        rec = ca.record_trade_cost("BTCUSDT", "BUY", 1.0, 60000.0,
                                   fee_rate=0.001, slippage_bps=5.0,
                                   impact_bps=1.0, gas_usd=2.0)
        # frais = 60$ ; slippage = 30$ ; impact = 6$ ; gas = 2$ -> 98$
        assert rec["fee_usd"] == pytest.approx(60.0)
        assert rec["slippage_usd"] == pytest.approx(30.0)
        assert rec["impact_usd"] == pytest.approx(6.0)
        assert rec["total_cost_usd"] == pytest.approx(98.0)
        assert ca.total_costs_usd == pytest.approx(98.0)

    def test_net_pnl(self):
        from core.cost_accounting import CostAccounting
        ca = CostAccounting()
        ca.record_trade_cost("BTCUSDT", "BUY", 1.0, 60000.0)
        # PnL brut 500$ - coûts = net
        assert ca.net_pnl(500.0) < 500.0

    def test_carry_cost(self):
        from core.cost_accounting import CostAccounting
        ca = CostAccounting()
        # position longue 100000$ avec funding 0.01% / 8h -> coût ~10$ / période
        c = ca.record_carry_cost("BTCUSDT", 100000.0, 0.0001, hold_hours=8.0)
        assert c == pytest.approx(10.0)
        assert ca.carry_cost["BTCUSDT"] == pytest.approx(10.0)

    def test_no_funding_neutral(self):
        from core.cost_accounting import CostAccounting
        ca = CostAccounting()
        assert ca.record_carry_cost("BTCUSDT", 100000.0, None) == 0.0


# --------------------------------------------------------------------------- #
# 5. PNL NET DANS MAIN
# --------------------------------------------------------------------------- #
class TestPnLNet:
    def test_pnl_net_code_present(self):
        import inspect
        src = inspect.getsource(__import__("main", fromlist=["x"]))
        assert "cost_accounting.total_costs_usd" in src
        assert "live_pnl_usd -= _costs" in src


# --------------------------------------------------------------------------- #
# 6. ATTRIBUTION
# --------------------------------------------------------------------------- #
class TestAttribution:
    def _attr(self):
        from core.attribution import PerformanceAttribution
        a = PerformanceAttribution()
        a.record("BTCUSDT", "Trend Following", 0.02, "Bull Trend (Low Vol)")
        a.record("BTCUSDT", "Trend Following", -0.01, "Bull Trend (Low Vol)")
        a.record("ETHUSDT", "Carry", 0.005, "Mean-Reverting Range")
        return a

    def test_by_factor(self):
        a = self._attr()
        factors = a.by_factor()
        assert "momentum" in factors   # Trend Following -> momentum
        assert "carry" in factors      # Carry -> carry
        assert factors["momentum"]["n"] == 2

    def test_by_regime(self):
        a = self._attr()
        regimes = a.by_regime()
        assert "Bull Trend (Low Vol)" in regimes
        assert regimes["Bull Trend (Low Vol)"]["n"] == 2

    def test_by_asset_strategy(self):
        a = self._attr()
        assert a.by_asset()["BTCUSDT"]["n"] == 2
        assert a.by_strategy()["Trend Following"]["n"] == 2

    def test_factor_mapping(self):
        from core.attribution import STRATEGY_FACTOR
        assert STRATEGY_FACTOR["Trend Following"] == "momentum"
        assert STRATEGY_FACTOR["Carry"] == "carry"
        assert STRATEGY_FACTOR["Volatility Breakout"] == "vol"


# --------------------------------------------------------------------------- #
# 7. MÉTRIQUES DE QUALITÉ
# --------------------------------------------------------------------------- #
class TestQualityMetrics:
    def test_metrics_from_equity(self):
        from core.attribution import quality_metrics
        rng = np.random.RandomState(0)
        eq = np.cumsum(rng.normal(0.001, 0.01, 100)) + 100.0
        m = quality_metrics(list(eq), periods_per_year=365)
        assert m["available"] is True
        assert m["sharpe"] > 0
        assert m["max_drawdown_pct"] >= 0
        assert m["sortino"] > 0

    def test_trade_metrics(self):
        from core.attribution import quality_metrics
        trades = [{"pnl": 0.02}, {"pnl": 0.01}, {"pnl": -0.01},
                  {"pnl": 0.03}, {"pnl": -0.005}]
        m = quality_metrics([100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                            trades=trades)
        assert m["win_rate_pct"] == pytest.approx(60.0)
        assert m["profit_factor"] > 1.0
        assert m["expectancy_pct"] > 0
        assert m["n_trades"] == 5

    def test_insufficient(self):
        from core.attribution import quality_metrics
        assert quality_metrics([100.0])["available"] is False


# --------------------------------------------------------------------------- #
# 8. INTÉGRATION
# --------------------------------------------------------------------------- #
class TestIntegration:
    def test_state_and_telemetry(self):
        import main
        assert "cost_metrics" in main.STATE
        assert "attribution_report" in main.STATE
        assert "quality_metrics" in main.STATE
        assert "stress_test_report" in main.STATE
        assert "bootstrap_sharpe" in main.STATE
        tel = main.compile_telemetry_data()
        for k in ("cost_metrics", "attribution_report", "quality_metrics",
                  "stress_test_report", "bootstrap_sharpe"):
            assert k in tel

    def test_endpoints_exist(self):
        import inspect
        src = inspect.getsource(__import__("main", fromlist=["x"]))
        assert 'api_v1_attribution' in src
        assert 'api_v1_stress' in src
        assert "audit_backtest" in src

    def test_report_has_metrics(self):
        import main
        from core.reporting import build_daily_report
        report = build_daily_report(main.STATE, main.db)
        assert "quality_metrics" in report
        assert "attribution" in report
        assert "costs" in report
        assert "stress_test" in report
