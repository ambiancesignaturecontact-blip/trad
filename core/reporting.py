"""
Daily P&L reporting, health score and the Telegram "risk concierge" digest
(audit C2, D3, D4).

- build_daily_report(): per-strategy / per-asset / per-mode P&L from the DB
- compute_health_score(): 0-100 composite (connections, data quality, AI drift,
  drawdown, errors, backups)
- build_concierge_message(): human-readable Telegram digest
"""
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger("Reporting")


def compute_health_score(state: Dict, db) -> int:
    """
    0-100 composite health. Each component contributes a weighted score.
    """
    score = 100.0
    reasons = []

    # 1. Data quality (25 pts)
    dq = state.get("data_quality_status", "UNAVAILABLE")
    dq_map = {"LIVE": 25, "DELAYED": 15, "STALE": 5, "INVALID": 0,
              "DISCONNECTED": 0, "UNAVAILABLE": 0}
    score -= (25 - dq_map.get(dq, 0))
    if dq != "LIVE":
        reasons.append(f"qualité données {dq}")

    # 2. Drawdown vs limits (20 pts)
    mode = state.get("mode", "DEMO")
    initial_cap = state.get("initial_capital_demo" if mode == "DEMO" else "initial_capital_real", 0.0) or 0.0
    equity = state.get("current_equity", 0.0) or 0.0
    if initial_cap > 0:
        dd = max(0.0, (initial_cap - equity) / initial_cap)
        limit = 0.025 if mode == "DEMO" else 0.18  # config-driven elsewhere
        if dd > limit * 0.8:
            score -= 20
            reasons.append(f"drawdown {dd*100:.1f}%")
        else:
            score -= int(dd / max(limit, 1e-6) * 12)

    # 3. Trading loop alive (20 pts) - loop sets a heartbeat timestamp each tick
    last_tick = float(state.get("last_tick_ts", 0.0))
    if time.time() - last_tick > 30:
        score -= 20
        reasons.append("boucle de trading lente/arrêtée")
    else:
        score -= int(min(20, (time.time() - last_tick) / 5.0))

    # 4. AI drift (15 pts)
    try:
        status = db.get_setting("active_model_status") or "DEPLOYED"
        if status in ("RETIRED", "ROLLED_BACK"):
            score -= 15
            reasons.append(f"statut modèle {status}")
    except Exception:
        pass

    # 5. Errors + websockets (10 pts)
    ws = len(state.get("connected_websockets", []))
    score -= 5 if ws == 0 else 0
    if ws == 0:
        reasons.append("aucun client connecté")

    # 6. Backups fresh (10 pts)
    try:
        import glob, os
        snaps = sorted(glob.glob(os.path.join(os.getcwd(), "backups", "trading_platform_*.db")))
        if snaps:
            age_h = (time.time() - os.path.getmtime(snaps[-1])) / 3600.0
            if age_h > 30:
                score -= 10
                reasons.append("backup daté")
        else:
            score -= 10
            reasons.append("aucun backup")
    except Exception:
        pass

    # 7. VISION §7c: honesty - when the simulation diverges from reality, distrust it
    try:
        _div = float(state.get("sim_divergence", 0.0))
        if _div > 0.5:
            score -= 15
            reasons.append(f"écart simulé/réel {_div:.1f}x")
        elif _div > 0.2:
            score -= 5
    except Exception:
        pass

    return max(0, min(100, int(score))), reasons


def build_daily_report(state: Dict, db) -> Dict:
    """P&L report: per strategy, per asset, per mode + positions + risk.

    LOT 8 (PDF Pilier Q) : enrichi avec les métriques de qualité
    (Sharpe/Sortino/Calmar/expectancy), l'attribution par facteur/régime
    et les coûts réels (Pilier O) — le rapport répond « ce dollar vient du
    momentum BTC en régime haussier ».
    """
    mode = state.get("mode", "DEMO")
    initial_cap = state.get("initial_capital_demo" if mode == "DEMO" else "initial_capital_real", 0.0) or 0.0
    equity = state.get("current_equity", 0.0) or 0.0
    live_pnl = equity - initial_cap if initial_cap > 0 else 0.0
    live_pnl_pct = (live_pnl / initial_cap * 100.0) if initial_cap > 0 else 0.0

    orders = []
    try:
        orders = db.get_all_orders() or []
    except Exception as e:
        logger.warning(f"Report: orders fetch failed: {e}")

    by_strategy: Dict[str, Dict] = {}
    by_asset: Dict[str, Dict] = {}
    today_ts = time.time() - 24 * 3600

    def _ts(o) -> float:
        t = o.get("timestamp")
        if isinstance(t, (int, float)):
            return float(t)
        if isinstance(t, str):
            try:
                return float(t)
            except ValueError:
                try:
                    from datetime import datetime
                    return datetime.fromisoformat(t).timestamp()
                except Exception:
                    return 0.0
        return 0.0

    filled = [o for o in orders if o.get("status") == "FILLED"]
    today = [o for o in filled if _ts(o) >= today_ts]

    for o in filled:
        strat = o.get("strategy") or "UNKNOWN"
        sym = o.get("symbol") or "?"
        qty = float(o.get("qty") or 0.0)
        price = float(o.get("price") or 0.0)
        notional = qty * price
        s = by_strategy.setdefault(strat, {"orders": 0, "notional": 0.0})
        s["orders"] += 1
        s["notional"] += notional
        a = by_asset.setdefault(sym, {"orders": 0, "notional": 0.0})
        a["orders"] += 1
        a["notional"] += notional

    positions = []
    try:
        positions = db.get_positions() or []
    except Exception:
        pass
    pos_list = []
    for p in positions:
        sym = p.get("symbol", "?")
        price = state.get("assets", {}).get(sym, {}).get("price") or 0.0
        qty = float(p.get("qty") or 0.0)
        avg = float(p.get("avg_price") or 0.0)
        pnl = (price - avg) * qty if price and avg else 0.0
        pos_list.append({"symbol": sym, "qty": qty, "avg_price": avg, "price": price, "pnl": round(pnl, 2)})

    health, reasons = compute_health_score(state, db)

    return {
        "generated_at": time.time(),
        "mode": mode,
        "equity": round(equity, 2),
        "initial_capital": round(initial_cap, 2),
        "pnl_usd": round(live_pnl, 2),
        "pnl_pct": round(live_pnl_pct, 3),
        "orders_today": len(today),
        "orders_total": len(filled),
        "by_strategy": {k: {**v, "notional": round(v["notional"], 2)} for k, v in by_strategy.items()},
        "by_asset": {k: {**v, "notional": round(v["notional"], 2)} for k, v in by_asset.items()},
        "positions": pos_list,
        "health_score": health,
        "health_reasons": reasons,
        "risk": {
            "regime": state.get("regime_name", "Unknown"),
            "onchain_risk": state.get("onchain_risk_score"),
            "onchain_available": state.get("onchain_available", False),
            "sentiment": state.get("sentiment_index"),
            "sentiment_available": state.get("sentiment_available", False),
            "data_quality": state.get("data_quality_status", "UNAVAILABLE"),
        },
        # LOT 8 (PDF Pilier Q) : métriques de qualité + attribution + coûts
        "quality_metrics": state.get("quality_metrics", {}),
        "attribution": {
            "by_factor": state.get("attribution_report", {}).get("by_factor", {}),
            "by_regime": state.get("attribution_report", {}).get("by_regime", {}),
            "by_asset": state.get("attribution_report", {}).get("by_asset", {}),
        },
        "costs": state.get("cost_metrics", {}),
        "stress_test": state.get("stress_test_report", {}),
        "bootstrap_sharpe": state.get("bootstrap_sharpe", {}),
    }


def build_concierge_message(report: Dict) -> str:
    """Telegram risk-concierge digest (audit D4)."""
    h = report["health_score"]
    emoji = "🟢" if h >= 80 else ("🟠" if h >= 50 else "🔴")
    lines = [
        f"{emoji} *CONCIERGE QUOTIDIEN* — Santé {h}/100",
        "━━━━━━━━━━━━━━━━━━━━━",
        f"💼 Mode : *{report['mode']}*",
        f"💰 Équité : *${report['equity']:,.2f}*",
        f"📈 P&L : *{report['pnl_usd']:+,.2f} $* ({report['pnl_pct']:+.2f}%)",
        f"🧾 Ordres aujourd'hui : {report['orders_today']}",
    ]
    if report["positions"]:
        lines.append("━━━━━━━━━━━━━━━━━━━━━")
        lines.append("📊 *Positions ouvertes :*")
        for p in report["positions"][:5]:
            lines.append(
                f"• {p['symbol']} ×{p['qty']} @ {p['avg_price']:,.2f} "
                f"({'+' if p['pnl'] >= 0 else ''}{p['pnl']:,.2f} $)"
            )
    if report["health_reasons"]:
        lines.append("━━━━━━━━━━━━━━━━━━━━━")
        lines.append("⚠️ Points d'attention : " + ", ".join(report["health_reasons"][:5]))
    lines.append("━━━━━━━━━━━━━━━━━━━━━")
    lines.append("🛡️ Le bot reste protégé (SL/TP + circuit breaker actifs).")
    return "\n".join(lines)
