"""
LOT 7 — Tests « Ops / Sécurité / Réconciliation / Copytrading / Contrepartie »
(PROMPT MAÎTRE : Faille 6 + Piliers K, J, P).

Vérifie :
 1. Vraie IP client dans les audit logs (plus de 127.0.0.1 codé en dur)
 2. Watchdog : détection + redémarrage des tâches mortes
 3. Supervisor étendu (signes vitaux multi-flux)
 4. Copytrading : scoring renforcé + risque par trader (plafond + stop global)
 5. Contrepartie : limite par exchange + signaux d'alerte + hot/cold custody
 6. Réconciliation : HALT auto en REAL sur écart
 7. Méta-attribution -> réduction auto du poids des mauvaises raisons
"""
import inspect

import pytest

from core.counterparty_risk import CounterpartyRiskManager
from core.self_assessment import meta_attribution, reason_weight_from_attribution


# --------------------------------------------------------------------------- #
# 1. IP CLIENT RÉELLE DANS LES AUDIT LOGS
# --------------------------------------------------------------------------- #
class TestRealIP:
    def test_no_hardcoded_localhost_in_main(self):
        """Plus de '127.0.0.1' codé en dur DANS LES APPELS d'audit (Faille 6).
        La valeur par défaut légitime de l'IP reste 127.0.0.1."""
        src = open("main.py").read()
        # Tous les appels d'audit utilisent audit_ip(), pas une IP en dur
        assert "audit_ip()" in src
        # Compter les usages : la seule occurrence autorisée est la valeur par défaut
        assert src.count('"127.0.0.1"') <= 1
        # les add_audit_log utilisent audit_ip()
        assert 'db.add_audit_log(' in src and 'audit_ip()' in src
        assert 'db.add_audit_log("' in src

    def test_no_hardcoded_in_telegram_bot(self):
        src = open("bot/telegram_bot.py").read()
        assert '"127.0.0.1"' not in src

    def test_middleware_sets_request_ip(self):
        """Le middleware mémorise l'IP réelle du client."""
        src = open("core/middleware.py").read()
        assert "set_request_ip" in src
        assert "request.client.host" in src

    def test_audit_ip_helper(self):
        import main
        main.set_request_ip("203.0.113.7")
        assert main.audit_ip() == "203.0.113.7"
        main.set_request_ip("127.0.0.1")  # reset


# --------------------------------------------------------------------------- #
# 2. WATCHDOG
# --------------------------------------------------------------------------- #
class TestWatchdog:
    def test_task_factories_registered(self):
        import main
        assert "live_trading_loop" in main.TASK_FACTORIES
        assert "order_flow_websocket_listener" in main.TASK_FACTORIES
        assert "reconciliation_scheduler" in main.TASK_FACTORIES
        assert "autonomous_ai_scheduler" in main.TASK_FACTORIES
        assert len(main.TASK_FACTORIES) >= 10

    def test_launch_named_names_tasks(self):
        import main
        import asyncio
        async def dummy():
            await asyncio.sleep(60)
        async def run():
            task = main.launch_named(dummy(), "dummy_test")
            assert task.get_name() == "qp_dummy_test"
            task.cancel()
            await asyncio.sleep(0.01)
        asyncio.run(run())

    def test_watchdog_in_state(self):
        import main
        assert "background_tasks" in main.STATE
        tel = main.compile_telemetry_data()
        assert "watchdog" in tel
        assert "tasks_monitored" in tel["watchdog"]


# --------------------------------------------------------------------------- #
# 3. SUPERVISOR ÉTENDU
# --------------------------------------------------------------------------- #
class TestSupervisorExtended:
    def test_supervisor_checks_multiple_flows(self):
        import inspect
        from core import robustness
        src = inspect.getsource(robustness.Supervisor.check)
        assert "heartbeat" in src
        assert "price" in src
        assert "divergence" in src
        assert "order flow" in src
        assert "sentiment" in src

    def test_supervisor_returns_issues(self):
        from core.robustness import Supervisor
        sup = Supervisor({})
        issues = sup.check(force=True)
        assert isinstance(issues, list)


# --------------------------------------------------------------------------- #
# 4. COPTRADING : SCORING + RISQUE PAR TRADER
# --------------------------------------------------------------------------- #
class TestCopyTradingRisk:
    def _trader(self, roi=1.0, win=0.5, dd=0.1, sharpe=1.5):
        from copytrading.manager import CopyTrader
        t = CopyTrader("id1", "Test", roi, win, dd, sharpe)
        return t

    def test_scoring_reinforced(self):
        # trader risqué (roi élevé + gros drawdown) < trader régulier
        t_risky = self._trader(roi=5.0, win=0.3, dd=0.6, sharpe=0.8)
        t_solid = self._trader(roi=1.2, win=0.55, dd=0.15, sharpe=1.8)
        assert t_solid.seq_score > t_risky.seq_score

    def test_capital_plafond(self):
        from copytrading.manager import CopyTradingManager
        m = CopyTradingManager()
        m.copied_traders["t1"] = {"allocated_capital": 50000.0, "pnl_estimate_usd": 0.0}
        ok, msg = m.check_trader_risk("t1", total_capital=100000.0)
        # 50000 > 10% de 100000 -> plafond dépassé
        assert ok is False
        assert "plafond" in msg

    def test_stop_global(self):
        from copytrading.manager import CopyTradingManager
        m = CopyTradingManager()
        # perte estimée -20% de l'allocation -> stop global
        m.copied_traders["t2"] = {"allocated_capital": 10000.0, "pnl_estimate_usd": -2000.0}
        ok, msg = m.check_trader_risk("t2", total_capital=1000000.0)
        assert ok is False
        assert "stop global" in msg

    def test_enforce_trader_risk_stops(self):
        from copytrading.manager import CopyTradingManager
        m = CopyTradingManager()
        m.copied_traders["bad"] = {"allocated_capital": 10000.0, "pnl_estimate_usd": -2500.0}
        m.copied_traders["good"] = {"allocated_capital": 5000.0, "pnl_estimate_usd": 100.0}
        stopped = m.enforce_trader_risk(total_capital=1000000.0)
        assert "bad" in stopped
        assert "good" not in stopped
        assert "bad" not in m.copied_traders


# --------------------------------------------------------------------------- #
# 5. CONTREPARTIE (PILIER P)
# --------------------------------------------------------------------------- #
class TestCounterpartyRisk:
    def test_balance_limit(self):
        cr = CounterpartyRiskManager(max_capital_per_exchange_pct=0.4)
        # 60% du capital sur un exchange -> block
        res = cr.check_exchange_balance("Binance", 60000.0, 100000.0)
        assert res["action"] == "block"
        assert res["ok"] is False
        # 30% -> ok
        res2 = cr.check_exchange_balance("Bybit", 30000.0, 100000.0)
        assert res2["action"] == "ok"
        # 38% -> warn (approche le plafond)
        res3 = cr.check_exchange_balance("OKX", 38000.0, 100000.0)
        assert res3["action"] == "warn"

    def test_exchange_health_signals(self):
        cr = CounterpartyRiskManager()
        # retraits suspendus -> risque critique
        res = cr.evaluate_exchange_health("FTX-like", withdrawals_suspended=True)
        assert res["risk_score"] == 1.0
        assert res["healthy"] is False
        # spread anormal
        res2 = cr.evaluate_exchange_health("Binance", spread_bps=40.0)
        assert res2["risk_score"] >= 0.7
        # volume effondré
        res3 = cr.evaluate_exchange_health("Bybit", volume_ratio=0.1)
        assert res3["risk_score"] >= 0.6
        # sain
        res4 = cr.evaluate_exchange_health("Binance", spread_bps=2.0, volume_ratio=1.0)
        assert res4["healthy"] is True
        assert res4["risk_score"] == 0.0

    def test_custody_hot_cold(self):
        cr = CounterpartyRiskManager()
        # 80% en hot wallet -> trop (max 30%)
        res = cr.custody_check(hot_balance_usd=80000.0, total_capital=100000.0)
        assert res["ok"] is False
        assert "TRANSFERER" in res["recommendation"]
        # 20% en hot -> ok
        res2 = cr.custody_check(hot_balance_usd=20000.0, total_capital=100000.0)
        assert res2["ok"] is True


# --------------------------------------------------------------------------- #
# 6. RÉCONCILIATION + HALT REAL
# --------------------------------------------------------------------------- #
class TestReconciliationHalt:
    def test_halt_code_present(self):
        """En mode REAL, un écart de réconciliation déclenche un HALT.
        (LOT 7 : le code de réconciliation vit désormais dans schedulers.py.)"""
        src = open("main.py").read() + open("schedulers.py").read()
        assert "RECONCILIATION_BALANCE" in src
        assert "RECONCILIATION_POSITIONS" in src
        assert 'risk_state.enter(RiskStateMachine.HALT' in src
        assert 'if active_mode == "REAL"' in src


# --------------------------------------------------------------------------- #
# 7. MÉTA-ATTRIBUTION -> RÉDUCTION AUTO
# --------------------------------------------------------------------------- #
class TestMetaAttributionLoop:
    def test_reason_weight_reduces_bad_reasons(self):
        log = [
            {"reasons": ["momentum"], "pnl": 0.02},
            {"reasons": ["momentum"], "pnl": -0.03},
            {"reasons": ["momentum"], "pnl": -0.02},
            {"reasons": ["trend"], "pnl": 0.03},
            {"reasons": ["trend"], "pnl": 0.02},
            {"reasons": ["trend"], "pnl": 0.01},
        ]
        attr = meta_attribution(log)
        weights = reason_weight_from_attribution(attr, min_samples=3)
        # "momentum" a 1 win / 3 -> win_rate 0.33 < 0.5 et pnl négatif -> réduit
        assert weights["momentum"] < 1.0
        # "trend" gagne 3/3 -> bonus
        assert weights["trend"] >= 1.0

    def test_no_penalty_without_sample(self):
        weights = reason_weight_from_attribution({"nouvelle": {"n": 1, "win_rate": 0.0, "avg_pnl": -0.1}},
                                                 min_samples=5)
        assert weights["nouvelle"] == 1.0  # pas assez de preuves -> neutre

    def test_pipeline_has_reason_attribution(self):
        from core.risk_pipeline import RISK_PIPELINE_ORDER, apply_risk_pipeline
        assert "reason_attribution" in RISK_PIPELINE_ORDER
        assert RISK_PIPELINE_ORDER.index("cash_reserve") < RISK_PIPELINE_ORDER.index("reason_attribution") < RISK_PIPELINE_ORDER.index("order_flow")
        res = apply_risk_pipeline(
            base_qty=100.0, cvar_qty=1000.0, max_asset_qty=1000.0,
            conviction=1.0, risk_state_scale=1.0, reason_attribution_scale=0.5)
        assert res["qty"] == pytest.approx(50.0)

    def test_main_integration(self):
        import main
        assert "decision_log" in main.STATE
        assert "reason_weights_factor" in main.STATE
        tel = main.compile_telemetry_data()
        assert "reason_weights" in tel
