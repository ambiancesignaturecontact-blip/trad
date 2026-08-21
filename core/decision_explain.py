"""
Explication de décision + push Prometheus extraits de main.py
(LOT C, F3). Corps inchangés.
"""

import logging

from main import (STATE, _neutral, platform_metrics)  # noqa: E402

logger = logging.getLogger("InstitutionalTradingBot")  # LOT C : même canal de logs que main


def explain_last_decision(consensus) -> list:
    """
    VISION §4.5: top-5 contributing features/reasons of the last decision.
    Returns a ranked list of {feature, contribution} derived from real data.
    """
    out = []
    try:
        contribs = consensus.get("contributions", {}) or {}
        ranked = sorted(contribs.items(), key=lambda kv: abs(kv[1].get("signal", 0.0) * kv[1].get("weight", 0.0)), reverse=True)
        for name, c in ranked[:5]:
            out.append({
                "feature": name,
                "signal": round(float(c.get("signal", 0.0)), 4),
                "weight": round(float(c.get("weight", 0.0)), 4),
                "contribution": round(float(c.get("signal", 0.0)) * float(c.get("weight", 0.0)), 4),
            })
        # append market-state features
        extras = [
            ("VPIN", consensus.get("modulate_factor", 1.0) if consensus.get("modulate_factor", 1.0) < 1.0 else 0.0),
        ]
        for fname, val in extras:
            if abs(val) > 1e-6:
                out.append({"feature": fname, "signal": round(val, 4), "weight": 1.0, "contribution": round(val, 4)})
    except Exception:
        pass
    return out


def update_metrics_from_state():
    """
    Pushes the current STATE snapshot into the Prometheus registry (LOT 61).
    Called at the end of every trading-loop tick and on demand.
    """
    active_mode = STATE["mode"]
    active_balance_key = "balance_demo" if active_mode == "DEMO" else "balance_real"
    _lp = STATE["last_price"]
    platform_metrics.MARKET_LAST_PRICE.labels(symbol="BTCUSDT").set(_lp if _lp is not None else 0.0)
    platform_metrics.MARKET_EQUITY.labels(mode=active_mode).set(STATE["current_equity"])
    platform_metrics.MARKET_BALANCE.labels(mode=active_mode).set(STATE[active_balance_key])

    initial_cap = STATE["initial_capital_demo"] if active_mode == "DEMO" else STATE["initial_capital_real"]
    live_pnl_usd = STATE["current_equity"] - initial_cap if initial_cap > 0 else 0.0
    live_pnl_pct = (live_pnl_usd / initial_cap) * 100.0 if initial_cap > 0 else 0.0
    platform_metrics.MARKET_PNL_USD.labels(mode=active_mode).set(live_pnl_usd)
    platform_metrics.MARKET_PNL_PCT.labels(mode=active_mode).set(live_pnl_pct)

    platform_metrics.REGIME_ID.set(STATE["regime_id"])
    platform_metrics.RISK_EXPOSURE.set(
        (STATE["current_equity"] / STATE[active_balance_key] - 1.0) * 100.0 if STATE[active_balance_key] > 0 else 0.0
    )
    platform_metrics.POSITIONS_OPEN.set(len(STATE.get("cached_positions") or []))
    platform_metrics.SENTIMENT_INDEX.set(_neutral(STATE.get("sentiment_index")))
    platform_metrics.ONCHAIN_RISK.set(_neutral(STATE.get("onchain_risk_score"), 0.5))
    platform_metrics.WS_CLIENTS.set(len(STATE["connected_websockets"]))
