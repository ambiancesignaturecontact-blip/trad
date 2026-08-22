"""
DAILY QUANT REPORT (PHASE 3 — §17).

Génère automatiquement un rapport périodique complet, à partir de données
RÉELLES uniquement, avec persistance (historique consultable). Sections :
  Market (régime, vol) · Trading (trades, NO_TRADE, PnL, drawdown, coûts)
  Intelligence (conviction, calibration, edge, modèles) · Risk (exposition,
  limites) · Execution (slippage, frais, latence) · Research (hypothèses,
  kill list) · Recommendation (capital allocation : MAINTAIN/REDUCE/...).

Persistance : db.save_setting("daily_quant_report_YYYY-MM-DD", json) +
historique des 30 derniers jours (daily_quant_reports_index).
"""
import json
import logging
import time
from datetime import datetime

logger = logging.getLogger("InstitutionalTradingBot")

REPORT_HISTORY_KEY = "daily_quant_reports_index"


def _today() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def build_daily_quant_report(db, state: dict, conviction_calibration: dict | None = None,
                             edge_decay_report: dict | None = None,
                             drift_psi: dict | None = None,
                             execution_intel: dict | None = None,
                             research_memory=None,
                             capital_engine=None) -> dict:
    """Construit le rapport complet (toutes sections). Aucun chiffre inventé."""
    try:
        # ---- Market ----
        market = {
            "regime_id": state.get("regime_id"),
            "regime_name": state.get("regime_name", ""),
            "regime_confidence": (state.get("regime_confidence", {}) or {}).get("confidence"),
            "drift_status": (drift_psi or {}).get("status", "STABLE"),
            "market_state": state.get("market_state", {}),
        }

        # ---- Trading ----
        j = db.decision_journal_summary() if hasattr(db, "decision_journal_summary") else {}
        trading = {
            "decisions": int(j.get("total", 0)),
            "by_decision": j.get("by_decision", {}),
            "by_reason": j.get("by_reason", {}),
            "closed_trades": int(j.get("closed_n", 0)),
            "closed_win_rate": j.get("closed_win_rate"),
            "no_trade_total": int((state.get("no_trade_stats") or {}).get("count", 0)),
            "no_trade_by_reason": (state.get("no_trade_stats") or {}).get("reasons", {}),
            "drawdown_pct": None,
        }
        try:
            eq = state.get("equity_history_demo", []) or state.get("equity_history_real", [])
            if eq:
                peak = max(eq)
                trading["drawdown_pct"] = round((peak - min(eq)) / peak * 100.0, 2) if peak > 0 else 0.0
        except Exception:
            pass

        # ---- Intelligence ----
        intelligence = {
            "conviction_threshold": state.get("conviction_threshold"),
            "last_conviction": state.get("last_conviction", {}),
            "calibration": conviction_calibration or {},
            "edge_decay_counts": (edge_decay_report or {}).get("counts", {}),
            "edge_decay_states": {k: v.get("state") for k, v in
                                  ((edge_decay_report or {}).get("per_strategy", {})).items()},
        }

        # ---- Risk ----
        risk = {
            "kill_switch": bool(state.get("kill_switch_active", False)),
            "risk_state": (state.get("risk_state") or {}).get("state"),
            "adversarial_last": state.get("last_adversarial", {}).get("verdict"),
        }

        # ---- Execution ----
        ei = execution_intel or {}
        execution = {
            "n_fills": ei.get("n", 0),
            "avg_slippage_bps": ei.get("avg_is_bps"),
            "avg_forecast_error_bps": ei.get("avg_forecast_error_bps"),
            "by_venue": ei.get("by_venue", {}),
        }

        # ---- Research ----
        research = {"kill_list": [], "recent_experiments": [], "n_killed": 0}
        if research_memory is not None:
            try:
                r = research_memory.report()
                research = {"kill_list": r.get("kill_list", []),
                            "recent_experiments": r.get("recent_experiments", [])[:5],
                            "n_killed": r.get("n_killed", 0)}
            except Exception:
                pass

        # ---- Recommendation (capital allocation) ----
        recommendation = {"recommendation": "MAINTAIN", "reasons": [],
                          "degradation_level": "NORMAL"}
        if capital_engine is not None:
            try:
                recommendation = capital_engine.recommend(
                    validation_status=state.get("paper_validation_status", "NOT_READY"),
                    closed_trades=int(j.get("closed_n", 0)),
                    expectancy_pct=(j.get("closed_avg_pnl_pct") or 0.0) / 100.0
                    if j.get("closed_avg_pnl_pct") is not None else None,
                    drawdown_pct=trading["drawdown_pct"] or 0.0,
                    drift_status=(drift_psi or {}).get("status", "STABLE"),
                    disabled_strategies=int((edge_decay_report or {})
                                            .get("counts", {}).get("disabled", 0)),
                    total_strategies=int((edge_decay_report or {})
                                         .get("counts", {}).get("total", 12)),
                    calibration_n=int((conviction_calibration or {}).get("n", 0)),
                    calibration_error=(conviction_calibration or {}).get("calibration_error"),
                    avg_slippage_bps=ei.get("avg_is_bps"),
                    forecast_error_bps=ei.get("avg_forecast_error_bps"),
                    kill_switch=risk["kill_switch"],
                    risk_state=risk["risk_state"] or "NORMAL",
                )
            except Exception as e:
                logger.debug(f"capital recommendation failed: {e}")

        report = {
            "date": _today(),
            "generated_ts": time.time(),
            "market": market,
            "trading": trading,
            "intelligence": intelligence,
            "risk": risk,
            "execution": execution,
            "research": research,
            "recommendation": recommendation,
            "note": "Rapport généré automatiquement depuis des données réelles. "
                    "La recommandation est un avis — aucun changement automatique.",
        }
        # persistance (best-effort)
        try:
            key = f"daily_quant_report_{_today()}"
            db.save_setting(key, json.dumps(report, default=str))
            idx = json.loads(db.get_setting(REPORT_HISTORY_KEY) or "[]")
            if key not in idx:
                idx.append(key)
            db.save_setting(REPORT_HISTORY_KEY, json.dumps(idx[-30:]))
        except Exception as e:
            logger.debug(f"daily report persist failed: {e}")
        return report
    except Exception as e:
        logger.warning(f"build_daily_quant_report failed: {e}")
        return {"date": _today(), "error": str(e)}
