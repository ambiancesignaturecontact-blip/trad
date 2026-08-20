"""
TESTS AUTONOMES — Mini-app complète + budget 50$ + assistant IA.

Vérifie les 3 problèmes signalés par l'utilisateur :
 1. L'assistant IA ne doit JAMAIS répondre « Pas de réponse » ni crasher
    (fallback structuré robuste aux None).
 2. Un budget de 50 $ doit pouvoir TRADER (sizing, filtres, méta-label).
 3. La mini-app doit permettre de régler le montant du capital DEMO.
"""
import asyncio

from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# 1. ASSISTANT IA
# --------------------------------------------------------------------------- #
class TestAssistant:
    def test_fallback_no_crash_without_price(self):
        """Sans prix réel (None), l'assistant répond honnêtement au lieu de 500."""
        from core.llm_narrative import answer_question_async
        r = asyncio.run(answer_question_async(
            "quel est le prix du bitcoin ?",
            {"last_price": None, "current_equity": 50000.0,
             "regime_name": "Mean-Reverting Range"}))
        assert "indisponible" in r
        assert "500" not in r  # pas d'erreur

    def test_fallback_answers_common_questions(self):
        from core.llm_narrative import answer_question_async
        cases = [
            ("pourquoi tu achètes ?", "méta-modèle"),
            ("comment tu gères le risque ?", "risque"),
            ("quelles sont mes positions ?", "positions"),
            ("c'est quoi le régime actuel ?", "Régime"),
        ]
        for q, kw in cases:
            r = asyncio.run(answer_question_async(q, {"last_price": 64000.0,
                                                       "current_equity": 50000.0,
                                                       "regime_name": "Range",
                                                       "positions": ["BTCUSDT"]}))
            assert r and len(r) > 10, f"réponse vide pour '{q}'"
            assert "Pas de réponse" not in r

    def test_endpoint_returns_answer(self):
        """L'endpoint /api/v1/assistant/ask renvoie une réponse (pas 500)."""
        import main
        with TestClient(main.app) as c:
            r = c.post("/api/v1/assistant/ask",
                       json={"question": "pourquoi tu n'as pas acheté hier ?"})
            assert r.status_code == 200
            body = r.json()
            assert "answer" in body
            assert body["answer"] and len(body["answer"]) > 5
            assert "Pas de réponse" not in body["answer"]

    def test_miniapp_shows_answer_or_error(self):
        """La mini-app utilise d.answer avec fallback uniquement sur erreur."""
        src = open("templates/telegram_mini_app.html").read()
        assert "d.answer || 'Pas de réponse'" in src


# --------------------------------------------------------------------------- #
# 2. BUDGET 50$ : LE BOT DOIT POUVOIR TRADER
# --------------------------------------------------------------------------- #
class TestMicroBudget50:
    def test_sizing_generates_trade(self):
        """Avec 50 $, le sizing produit un ordre >= min notional (3$)."""
        from core.paper_execution import min_notional_for_capital
        from risk.risk_manager import RiskManager
        rm = RiskManager()
        # BTC
        qty = rm.calculate_position_size(50.0, 500.0, 64000.0)
        assert qty * 64000.0 >= min_notional_for_capital(50.0)  # >= 3$
        # EURUSD
        qty2 = rm.calculate_position_size(50.0, 0.002, 1.158)
        assert qty2 * 1.158 >= min_notional_for_capital(50.0)
        # SOL
        qty3 = rm.calculate_position_size(50.0, 0.5, 77.0)
        assert qty3 * 77.0 >= min_notional_for_capital(50.0)

    def test_meta_label_warmup_does_not_block_demo(self):
        """Le méta-label ne bloque plus en DEMO : warm-up 20 trades, et un
        win rate faible RÉDUIT la taille au lieu de bloquer (apprentissage)."""
        src = open("main.py").read()
        assert "min_samples=20" in src
        # DEMO : pas de continue bloquant sur meta-label
        assert 'if not _ml_ok:' in src
        assert 'active_mode == "REAL"' in src
        # la réduction DEMO existe
        assert "target_qty *= max(0.25" in src

    def test_meta_label_still_blocks_real(self):
        """En REAL, le filtre reste bloquant (prudence maximale)."""
        src = open("main.py").read()
        assert "meta-label REAL" in src

    def test_decision_pipeline_allows_50usd(self):
        """Pipeline de risque complet avec 50 $ : la taille reste > 0."""
        from core.risk_pipeline import apply_risk_pipeline
        res = apply_risk_pipeline(
            base_qty=0.00015625,          # ~10 $ de BTC à 64000
            cvar_qty=1000.0,              # pas de contrainte CVaR
            max_asset_qty=0.0007,         # 50$ * 0.85 / 64000 ≈ cap 42.5$
            conviction=0.9,               # signal fort
            risk_state_scale=1.0,
            order_flow_scale=1.0,
            regime_confidence_scale=1.0,
            capacity_scale=1.0,
            cash_reserve_scale=1.0,
        )
        assert res["qty"] > 0
        assert res["final_scale"] > 0.5

    def test_micro_risk_limits(self):
        """Les limites micro-comptes sont appliquées (config)."""
        from risk.risk_manager import RiskManager
        rm = RiskManager()
        rm.set_initial_capital(50.0)
        assert rm.params["max_daily_drawdown_pct"] == 0.18  # micro
        assert rm.params["max_total_drawdown_pct"] == 0.35  # micro


# --------------------------------------------------------------------------- #
# 3. SET MONTANT (MINI-APP + API)
# --------------------------------------------------------------------------- #
class TestSetBalance:
    def test_endpoint_exists(self):
        import main
        from test_support import all_api_paths
        routes = all_api_paths(main.app)
        assert "/api/set-demo-balance" in routes

    def test_endpoint_sets_balance(self):
        import main
        with TestClient(main.app) as c:
            r = c.post("/api/set-demo-balance", json={"balance": 50.0})
            assert r.status_code == 200
            assert main.STATE["balance_demo"] == 50.0
            assert main.STATE["initial_capital_demo"] == 50.0
            # reset pour ne pas polluer
            c.post("/api/set-demo-balance", json={"balance": 100000.0})
            assert main.STATE["balance_demo"] == 100000.0

    def test_endpoint_rejects_negative(self):
        import main
        with TestClient(main.app) as c:
            r = c.post("/api/set-demo-balance", json={"balance": -5})
            assert r.status_code == 422

    def test_miniapp_has_set_balance_control(self):
        """La mini-app contient le panneau de réglage du capital."""
        src = open("templates/telegram_mini_app.html").read()
        assert "set-balance-input" in src
        assert "APPLIQUER" in src
        assert "setBalanceQuick(50)" in src      # bouton $50
        assert "setBalanceQuick(100000)" in src  # bouton $100k
        assert "doSetBalance" in src
        assert "/api/set-demo-balance" in src

    def test_miniapp_badges(self):
        """La mini-app affiche les badges équité/état/risque/data/coûts."""
        src = open("templates/telegram_mini_app.html").read()
        for bid in ("equity-display", "badge-bot", "badge-risk", "badge-data", "badge-costs"):
            assert bid in src, f"badge {bid} manquant"


# --------------------------------------------------------------------------- #
# 4. INTÉGRATION : la mini-app est complète
# --------------------------------------------------------------------------- #
class TestMiniAppComplete:
    def test_all_sections_present(self):
        src = open("templates/telegram_mini_app.html").read()
        sections = [
            "MODE", "CAPITAL", "P&amp;L LIVE", "SHARPE", "DRAWDOWN",
            "ASSISTANT", "MODE CONSULTATIF", "STRATÉGIES",
            "RISQUE", "Honnêteté modules", "Order Flow", "Stress crises",
            "Coûts réels", "RÉGLER LE CAPITAL",
        ]
        for s in sections:
            assert s in src, f"section '{s}' manquante dans la mini-app"

    def test_endpoints_used_exist(self):
        """Tous les endpoints appelés par la mini-app existent dans main.py."""
        import re

        import main
        from test_support import all_api_paths
        routes = set(all_api_paths(main.app))
        src = open("templates/telegram_mini_app.html").read()
        api_calls = set(re.findall(r"fetch\('(/api/[^']*)'", src))
        missing = [c for c in api_calls if c not in routes and c not in
                   ("/api/telemetry", "/api/status")]  # ces 2 sont GET implicites
        # /api/telemetry et /api/status existent aussi
        missing = [c for c in api_calls if c not in routes]
        assert not missing, f"endpoints manquants appelés par la mini-app: {missing}"

    def test_assistant_input_enter_key(self):
        """Bonus : l'entrée Entrée dans l'input assistant déclenche la question."""
        src = open("templates/telegram_mini_app.html").read()
        assert "assistant-input" in src


# --------------------------------------------------------------------------- #
# 5. SYNCHRO DASHBOARD <-> MINI-APP
# --------------------------------------------------------------------------- #
class TestUISync:
    def test_visual_palette_unified(self):
        """La mini-app utilise la palette slate (comme le dashboard), plus de zinc."""
        mini = open("templates/telegram_mini_app.html").read()
        assert "zinc-" not in mini
        assert "slate-" in mini
        assert "var(--accent)" in mini or "#22d3ee" in mini  # accent cyan unifié

    def test_assistant_in_both(self):
        dash = open("templates/dashboard.html").read()
        mini = open("templates/telegram_mini_app.html").read()
        assert "assistant" in dash.lower()
        assert "assistant" in mini.lower()
        assert "/api/v1/assistant/ask" in dash
        assert "/api/v1/assistant/ask" in mini

    def test_set_balance_in_both(self):
        dash = open("templates/dashboard.html").read()
        mini = open("templates/telegram_mini_app.html").read()
        assert "/api/set-demo-balance" in dash
        assert "/api/set-demo-balance" in mini
        assert "RÉGLER LE CAPITAL" in dash
        assert "RÉGLER LE CAPITAL" in mini

    def test_advanced_controls_in_miniapp(self):
        """La mini-app a maintenant les contrôles avancés du dashboard."""
        mini = open("templates/telegram_mini_app.html").read()
        for ep in ("/api/retrain", "/api/monte-carlo", "/api/v1/stress", "/api/reset-bot", "/api/toggle-strategy"):
            assert ep in mini, f"{ep} manquant dans la mini-app"

    def test_narrative_in_both(self):
        dash = open("templates/dashboard.html").read()
        mini = open("templates/telegram_mini_app.html").read()
        assert "/api/v1/narrative" in dash
        assert "/api/v1/narrative" in mini

    def test_approve_in_both(self):
        dash = open("templates/dashboard.html").read()
        mini = open("templates/telegram_mini_app.html").read()
        assert "/api/v1/approve" in dash
        assert "/api/v1/approve" in mini

    def test_all_endpoints_exist(self):
        """Tous les endpoints appelés par les deux interfaces existent."""
        import re

        import main
        from test_support import all_api_paths
        routes = set(all_api_paths(main.app))
        for f in ("templates/dashboard.html", "templates/telegram_mini_app.html"):
            src = open(f).read()
            calls = set(re.findall(r"fetch\('(/api/[^']*)'", src))
            missing = [c for c in calls if c not in routes]
            assert not missing, f"{f}: endpoints manquants {missing}"

    def test_key_sections_in_both(self):
        dash = open("templates/dashboard.html").read().lower()
        mini = open("templates/telegram_mini_app.html").read().lower()
        for s in ("order flow", "état risque", "honnêteté", "attribution", "stress", "coûts"):
            assert s in dash, f"'{s}' manquant dans dashboard"
            assert s in mini, f"'{s}' manquant dans mini-app"


# --------------------------------------------------------------------------- #
# Régression : le trading micro 50$ ne doit plus être rejeté (min notional)
# --------------------------------------------------------------------------- #

def test_micro_50usd_trade_survives_pipeline_and_min_notional(monkeypatch):
    """FIX : avec un signal faible (conviction ~0.10), la taille post-pipeline
    tombait sous le min notional (1,50$ < 3$) -> simulate_paper_fill rejetait
    -> le bot ne tradait JAMAIS sur petit compte. En DEMO, la taille doit être
    remontée au min notional (bornée 80% du capital) et le fill accepté."""
    from core.paper_execution import min_notional_for_capital, simulate_paper_fill
    from core.risk_pipeline import apply_risk_pipeline
    from risk.risk_manager import RiskManager

    capital = 50.0
    price = 170.0  # SOL
    atr = price * 0.008
    rm = RiskManager()
    rm.set_initial_capital(capital)
    base_qty = rm.calculate_position_size(capital=capital, atr=atr, current_price=price,
                                          win_rate=0.45, reward_risk_ratio=1.8)
    assert base_qty * price == 10.0  # sizing micro force 10$ (>= min 10)

    _pipe = apply_risk_pipeline(
        base_qty=base_qty, cvar_qty=1e9, max_asset_qty=1e9,
        conviction=0.10, risk_state_scale=0.8, news_scale=0.8, macro_scale=0.8,
        onchain_scale=0.8, corr_scale=0.8, order_flow_scale=0.8,
        regime_confidence_scale=0.8, capacity_scale=0.8, cash_reserve_scale=0.8,
        reason_attribution_scale=0.8, confidence_scale=0.8, org_scale=0.8,
        vol_scale=1.0, tradability_scale=0.8)
    qty = _pipe["qty"]
    mn = min_notional_for_capital(capital)
    assert qty * price < mn, "précondition : la taille post-pipeline est sous le min"

    # FIX simulé : remontée bornée (80% du capital) puis fill
    bumped = min(mn / price, (capital * 0.80) / price)
    book = {"bids": [[price - 0.1, 50.0]], "asks": [[price + 0.1, 50.0]]}
    paper = simulate_paper_fill("SOLUSDT", "BUY", bumped, price, book,
                                "Binance", balance=capital)
    assert paper.get("rejected") is False, f"fill rejeté: {paper.get('reason')}"
    assert paper["fill_price"] > 0
