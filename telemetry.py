"""
Télémétrie extraite de main.py (étape 2 du découpage, LOT 7 / P1-7).
Corps STRICTEMENT inchangés ; les symboles partagés viennent de main via
`from main import *` (main est complet quand ce module est importé, en fin
de main.py).
"""
import json
import time

import main  # noqa: F401
from core.fx import convert as _fx_convert  # noqa: F401
from core.fx import get_account_currency as _fx_ccy
from core.fx import usd_to as _fx_usd_to
from core.module_honesty import get_module_status, status_summary  # noqa: F401
# LOT C (F3) : fin du 'from main import *' — imports EXPLICITES des symboles
# utilisés par ce module (vérifiés par AST + test_routes_health).
from main import (  # noqa: F401
    STATE,
    capital_allocator,
    copy_manager,
    cost_accounting,
    counterparty_risk,
    db,
    hypothesis_generator,
    logger,
    macro_calendar,
    meta_engine,
    mixture_of_experts,
    model_selector,
    order_flow,
    platform_metrics,
    supervisor,
)
from main import _BG_TASKS, _paper_validation_stats, _signal_stats, conviction_engine, edge_decay  # noqa: F401

# Devise du compte (résolue une fois par appel — le cache FX interne gère la
# fréquence des appels réseau)
_acct_ccy = _fx_ccy()
_fx_rate = _fx_usd_to(_acct_ccy)


def _fx_display(value_usd: float) -> dict:
    """Affichage multi-devise honnête : {value, currency, fx_rate}."""
    ccy = _acct_ccy
    rate = _fx_rate
    if ccy == "USD" or rate is None:
        return {"value": round(float(value_usd), 2), "currency": ccy if ccy else "USD",
                "fx_rate": rate}
    return {"value": round(float(value_usd) * rate, 2), "currency": ccy, "fx_rate": rate}


def serialize_helper(obj):
    """
    Safely converts any datetime or non-serializable database object into standard string/types
    before sending over WebSockets or JSON responses.
    Also strips NaN/Infinity floats which are not valid JSON (would crash the endpoint with
    "ValueError: Out of range float values are not JSON compliant").
    """
    if isinstance(obj, dict):
        return {k: serialize_helper(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_helper(i) for i in obj]
    elif isinstance(obj, tuple):
        return [serialize_helper(i) for i in obj]
    elif isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    elif isinstance(obj, int):
        return obj
    elif hasattr(obj, "isoformat"):  # Matches datetime.datetime, date, etc.
        return obj.isoformat()
    return obj



def compile_telemetry_data(consensus_signals=None) -> dict:
    """
    Compiles and returns the unified telemetry payload.
    Caches database queries for 3.0 seconds to avoid blocking the event loop on high-frequency ticks.
    """
    now = time.time()
    if now - STATE.get("last_db_query_time", 0.0) >= 3.0 or consensus_signals is not None:
        STATE["last_db_query_time"] = now
        try:
            STATE["cached_positions"] = db.get_positions()
            STATE["cached_orders"] = db.get_all_orders()
            STATE["cached_audit_logs"] = db.get_audit_logs()
        except Exception as e:
            logger.error(f"Failed to fetch telemetry data from database: {str(e)}")

    positions = STATE.get("cached_positions", [])
    orders = STATE.get("cached_orders", [])
    audit_logs = STATE.get("cached_audit_logs", [])

    # Calculate live P&L
    active_mode = STATE["mode"]
    initial_cap = STATE["initial_capital_demo"] if active_mode == "DEMO" else STATE["initial_capital_real"]
    current_eq = STATE["current_equity"]

    live_pnl_usd = current_eq - initial_cap if initial_cap > 0 else 0.0
    # LOT 8 (PDF Pilier O) : PnL NET — les coûts RÉELS (frais + slippage +
    # impact + gas + funding) sont retranchés du PnL affiché. Mentalité n°2 :
    # l'edge est net des coûts.
    try:
        _costs = float(cost_accounting.total_costs_usd)
        live_pnl_usd -= _costs
    except Exception:
        pass
    live_pnl_pct = (live_pnl_usd / initial_cap) * 100.0 if initial_cap > 0 else 0.0

    # Packaged JSON (Passed through serialize_helper to resolve any PostgreSQL datetime serialization mismatches!)
    telemetry = {
        "mode": STATE["mode"],
        "is_running": STATE["is_running"],
        "kill_switch_active": STATE["kill_switch_active"],
        # LOT F (F6) : derniers envois d'alertes opérationnelles (transparence)
        "ops_alerts_last_ts": STATE.get("ops_alerts_last_ts", {}),
        "last_price": STATE["last_price"],
        "price_history": STATE["price_history"],
        "order_book": STATE["order_book"],
        "balance": STATE["balance_demo"] if STATE["mode"] == "DEMO" else STATE["balance_real"],
        "current_equity": STATE["current_equity"],
        # P2-multi-devise : devise du compte + valeurs converties (l'interne
        # reste en USD ; l'affichage est converti dans la devise du compte
        # via des taux RÉELS — principe base currency, jamais de taux inventé).
        "account_currency": _acct_ccy,
        "fx_rate_usd_to_account": _fx_rate,
        "balance_account_ccy": _fx_display(STATE["balance_demo"] if STATE["mode"] == "DEMO" else STATE["balance_real"]),
        "equity_account_ccy": _fx_display(STATE["current_equity"]),
        "pnl_account_ccy": _fx_display(live_pnl_usd),
        "equity_history": STATE["equity_history_demo"] if STATE["mode"] == "DEMO" else STATE["equity_history_real"],
        "live_pnl_usd": live_pnl_usd,
        "live_pnl_pct": live_pnl_pct,
        "regime_id": STATE["regime_id"],
        "regime_name": STATE["regime_name"],
        "ml_prediction_pct": STATE["ml_prediction_pct"],
        "ppo_action": STATE["ppo_action"],
        "consensus": consensus_signals,
        "positions": serialize_helper(positions),
        "orders": serialize_helper(orders[:15]),
        "audit_logs": serialize_helper(audit_logs[:15]),

        # ADVANCED TELEMETRY EXPOSURE
        "sentiment_index": STATE["sentiment_index"],
        "sentiment_available": STATE.get("sentiment_available", False),
        "sentiment_confidence": STATE.get("sentiment_confidence", 0.0),
        "recent_headlines": STATE.get("recent_headlines", [])[:10],
        "news_shock": STATE.get("news_shock", {"shock_detected": False}),
        "macro_phase": STATE.get("macro_phase", "NONE"),
        "macro_event": STATE.get("macro_event", ""),
        "onchain_risk_score": STATE["onchain_risk_score"],
        "onchain_available": STATE.get("onchain_available", False),
        "eth_defi_balance": STATE["eth_defi_balance"],
        "defi_wallet_address": STATE["defi_wallet_address"],
        "assets_telemetry": STATE["assets"],
        "asset_data_status": STATE.get("asset_data_status", {}),
        "order_books": {k: {kk: vv for kk, vv in v.items() if kk != "_ts"} for k, v in STATE.get("order_books", {}).items()},
        "price_consensus": STATE.get("price_consensus", {}),
        "price_divergent": STATE.get("price_divergent", {}),
        "macro_calendar": macro_calendar.get_calendar(limit=5),
        "options_strategy": STATE["options_strategy"],
        "real_iv": STATE.get("real_iv", {}),

        "using_fallback_data": STATE.get("using_fallback_data", False),
        "data_quality_status": STATE["data_quality_status"],
        "vol_target_scale": STATE.get("vol_target_scale", 1.0),
        "last_reasoning": STATE.get("last_reasoning", []),
        "last_reasoning_symbol": STATE.get("last_reasoning_symbol", ""),
        "regime_probs": STATE.get("regime_probs", {}),
        "conviction_threshold": STATE.get("conviction_threshold", 0.15),
        "no_trade_count": STATE.get("no_trade_stats", {}).get("count", 0),
        # LOT 1 (observabilité décisionnelle) : POURQUOI le bot n'a pas tradé
        # — breakdown par catégorie de raison + distribution de conviction.
        "no_trade_reasons": STATE.get("no_trade_stats", {}).get("reasons", {}),
        "signal_stats": _signal_stats(),
        # LOT 2 (mandat) : Conviction Engine — niveau, edge net, décision
        # TRADE/WAIT et calibration mesurée (buckets -> win rate, expectancy).
        "conviction_engine": STATE.get("last_conviction", {}),
        "last_opportunity": STATE.get("last_opportunity", {}),
        "conviction_calibration": conviction_engine.calibration_report(),
        # LOT 3 (mandat) : résumé du journal de décision (TRADE/WAIT par raison,
        # win rate des clôturés) — Trade Intelligence.
        "decision_journal_summary": db.decision_journal_summary(),
        # LOT 4 (mandat) : edge decay — états des stratégies + scales appliqués
        "edge_decay": edge_decay.report(),
        "moe_gate": STATE.get("moe_gate", {}),
        "risk_budget": STATE.get("risk_budget", {}),
        "risk_state": STATE.get("risk_state", {}),
        "last_kelly": STATE.get("last_kelly", {}),
        "last_rr_check": STATE.get("last_rr_check", {}),
        "strategy_win_rates": STATE.get("strategy_win_rates", {}),
        "strategy_trade_counts": STATE.get("strategy_trade_counts", {}),
        "risk_pipeline_steps": STATE.get("risk_pipeline_steps", [])[-12:],
        # LOT A (F1) : nb de facteurs réellement actifs (< 1.0) au dernier tick
        "active_factors_last": main.risk_pipeline_last.get("active_factors", 0),
        # P0-4 (audit §2.1) : distribution observée de final_scale (p10/p50/p90)
        "final_scale_stats": STATE.get("final_scale_stats"),
        "final_scale_samples_count": len(STATE.get("final_scale_samples", [])),
        # P0-6 (audit §5) : paper-trading daté et continu avant REAL
        "paper_validation": _paper_validation_stats(),
        "regime_confidence": STATE.get("regime_confidence", {}),
        # LOT B (F2) : autonomie stratégique — facteur d'agressivité lissé
        # (borné [0.60, 1.25]) + paramètres de risque EFFECTIFS appliqués
        # (Kelly, plafond par actif, drawdowns jamais élargis).
        "regime_autonomy": STATE.get("regime_autonomy", {}),
        # LOT D (F4) : drift de distribution (PSI) — max_psi, statut
        # STABLE/MODERATE/SEVERE, decay du bandit appliqué.
        "drift_psi": STATE.get("drift_psi", {}),
        "hmm_validation": STATE.get("hmm_validation", {}),
        "expert_contribution": mixture_of_experts.expert_contribution_report(),
        "sleeping_experts": list(mixture_of_experts.sleeping),
        "causal_parents": STATE.get("causal_parents", []),
        "causal_analyzed": STATE.get("causal_analyzed", False),
        "research_gate": hypothesis_generator.can_run_research(),
        "order_flow": {s: order_flow.status(s) for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]},
        "last_sor_choice": STATE.get("last_sor_choice", {}),
        "sim_divergence": STATE.get("sim_divergence", 0.0),
        "confidence_index": STATE.get("confidence_index", 100),
        "confidence_factor": STATE.get("confidence_factor", 1.0),
        "live_p_value": STATE.get("live_p_value", 0.5),
        "structural_regimes": STATE.get("structural_regimes", {}),
        "cross_asset_bias": STATE.get("cross_asset_bias", 0.0),
        "desk_allocations": STATE.get("desk_allocations", {}),
        "pending_approvals": len(STATE.get("pending_approvals", [])),
        "consultative_mode": STATE.get("consultative_mode", False),
        "last_narrative": STATE.get("last_narrative", ""),
        "ppo_buffer_size": len(STATE.get("ppo_buffer", [])),
        "strategy_weights": meta_engine.get_strategy_weights(),
        "active_models": model_selector.get_status().get("active_models", []),
        "admitted_signals": list(hypothesis_generator.admitted.keys()),
        "capital_exposure": capital_allocator.get_current_exposure(),
        "portfolio_allocation": STATE.get("portfolio_allocation", {}),
        "strategy_diversification": STATE.get("strategy_diversification", {}),
        "position_pyramids": STATE.get("position_pyramids", {}),
        "counterparty": counterparty_risk.to_dict(),
        "reason_weights": STATE.get("reason_weights", {}),
        "reason_weights_factor": STATE.get("reason_weights_factor", 1.0),
        "cost_metrics": STATE.get("cost_metrics", {}),
        "attribution_report": STATE.get("attribution_report", {}),
        "quality_metrics": STATE.get("quality_metrics", {}),
        "stress_test_report": STATE.get("stress_test_report", {}),
        "bootstrap_sharpe": STATE.get("bootstrap_sharpe", {}),
        "module_honesty": {
            "registry": get_module_status(),
            "summary": status_summary(),
            "note": "Un module ÉDUCATIF n'influence JAMAIS le sizing réel (Faille 7 PDF).",
        },
        "watchdog": {
            "tasks_monitored": list(_BG_TASKS.keys()),
            "tasks_alive": sum(1 for t in _BG_TASKS.values() if t and not t.done()),
            "supervisor_issues": supervisor.last_issues,
            "running": True,
        },

        "copy_traders": [
            {
                "trader_id": t.trader_id,
                "name": t.name,
                "roi_annual": t.roi_annual * 100.0,
                "win_rate": t.win_rate * 100.0,
                "max_drawdown": t.max_drawdown * 100.0,
                "sharpe": t.sharpe,
                "seq_score": t.seq_score,
                "pnl_month": getattr(t, "pnl_month", 0.0),
                "account_value": getattr(t, "account_value", 0.0),
                "active_copied": t.trader_id in copy_manager.copied_traders,
                "allocated_capital": copy_manager.copied_traders[t.trader_id]["allocated_capital"] if t.trader_id in copy_manager.copied_traders else 0.0,
                "follow_mode": copy_manager.copied_traders[t.trader_id].get("mode", "-") if t.trader_id in copy_manager.copied_traders else "-",
                "pnl_estimate_usd": copy_manager.copied_traders[t.trader_id].get("pnl_estimate_usd", 0.0) if t.trader_id in copy_manager.copied_traders else 0.0
            }
            for t in copy_manager.get_ranked_traders()
        ]
    }
    # Sanitize the full payload: strips NaN/Inf (invalid JSON) and datetimes
    return serialize_helper(telemetry)



async def broadcast_telemetry(consensus_signals):
    """
    Broadcasts real-time trading metrics to all active dashboard connections.
    Audit B5-1: payload is serialized ONCE, each client send is isolated with
    try/except and slow/failed clients are dropped immediately.
    """
    if not STATE["connected_websockets"]:
        return

    payload = compile_telemetry_data(consensus_signals)
    try:
        text = json.dumps(payload, default=str)
    except Exception as e:
        logger.warning(f"Telemetry serialization failed: {e}")
        return

    dead_sockets = []
    for ws in list(STATE["connected_websockets"]):
        try:
            await ws.send_text(text)
        except Exception:
            dead_sockets.append(ws)

    for ws in dead_sockets:
        try:
            STATE["connected_websockets"].remove(ws)
            platform_metrics.WS_CLIENTS.set(len(STATE["connected_websockets"]))
        except Exception:
            pass

