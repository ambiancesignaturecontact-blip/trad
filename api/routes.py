"""
Routes API extraites de main.py (LOT 7, P1-7 audit §4.1).
Corps des fonctions STRICTEMENT inchangés ; les décorateurs @app.X
deviennent @router.X. Les symboles partagés viennent de main via
`from main import *` (main est complet quand ce module est importé,
en fin de main.py). Étape 1 du découpage : sortir le code, pas encore
refactorer les dépendances.
"""
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict
import json
import os
import time
import asyncio
import numpy as np
import pandas as pd

import main  # noqa: F401  (le module est complet au moment de cet import)
from main import *  # noqa: F401,F403  (symboles partagés, étape 1 du découpage)
from main import _alerts_persist, _final_scale_report, _final_scale_stats, _limiting_factor_stats, _load_final_scale_samples, _mark_paper_validation_day, _neutral, _paper_validation_stats, _signal_stats  # noqa: F401

router = APIRouter()


@router.get("/api/v1/health")
async def api_health():
    """AUDIT D3: composite 0-100 health score."""
    score, reasons = compute_health_score(STATE, db)
    return {"health_score": score, "reasons": reasons, "mode": STATE["mode"], "ts": time.time()}


@router.get("/api/v1/news")
async def api_v1_news(_auth: dict = Depends(require_auth)):
    """
    LOT 5 (PDF Pilier I) : état des ACTUALITÉS RÉELLES.
    Expose le sentiment (index + confiance), les dernières headlines réelles
    avec leur source, le choc systémique éventuel et la pondération des sources.
    Audit : aucune donnée fictive — headlines vides si sources hors ligne.
    """
    return {
        "sentiment_index": STATE.get("sentiment_index"),
        "available": STATE.get("sentiment_available", False),
        "confidence": STATE.get("sentiment_confidence", 0.0),
        "num_headlines": len(STATE.get("recent_headlines", [])),
        "headlines": STATE.get("recent_headlines", [])[:20],
        "shock_status": STATE.get("news_shock", {"shock_detected": False}),
        "source_weights": {k: v for k, v in SOURCE_WEIGHTS_REF.items()},
        "ts": time.time(),
    }


@router.post("/api/v1/macro/override")
async def api_v1_macro_override(payload: MacroOverrideRequest,
                                _auth: dict = Depends(require_auth)):
    """
    LOT 5 (PDF Pilier I) : pilotage HUMAIN du risque macro.
    - reduce : applique un facteur de taille manuel (défaut 0.5)
    - halt   : passe la machine à états en HALT (nouveaux ordres bloqués)
    - reset  : revient à NORMAL, facteur 1.0
    L'opérateur reste le décideur final (mentalité n°10).
    """
    try:
        if payload.action == "reduce":
            STATE["macro_scale_factor_tactile"] = payload.factor
            db.add_audit_log("MACRO_OVERRIDE", audit_ip(),
                             f"Opérateur: réduction macro manuelle x{payload.factor}")
            return {"ok": True, "action": "reduce", "factor": payload.factor,
                    "risk_state": risk_state.to_dict()}
        if payload.action == "halt":
            risk_state.enter(RiskStateMachine.HALT, "MACRO_OVERRIDE")
            STATE["risk_state"] = risk_state.to_dict()
            db.add_audit_log("MACRO_OVERRIDE", audit_ip(), "Opérateur: HALT macro manuel")
            return {"ok": True, "action": "halt", "risk_state": risk_state.to_dict()}
        # reset
        risk_state.reset(reason="macro/override")
        STATE["macro_scale_factor_tactile"] = 1.0
        STATE["risk_state"] = risk_state.to_dict()
        db.add_audit_log("MACRO_OVERRIDE", audit_ip(), "Opérateur: reset macro (NORMAL)")
        return {"ok": True, "action": "reset", "risk_state": risk_state.to_dict()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/v1/report/daily")
async def api_daily_report():
    """AUDIT C2: daily P&L report (per strategy / asset / mode + risk)."""
    return build_daily_report(STATE, db)


@router.get("/api/v1/orders")
async def api_orders_v1(limit: int = 50, offset: int = 0):
    """AUDIT B2-5: paginated order history."""
    limit = max(1, min(500, limit))
    offset = max(0, offset)
    try:
        rows = db.get_all_orders() or []
    except Exception:
        rows = []
    page = [serialize_helper(o) for o in rows[offset:offset + limit]]
    return {"total": len(rows), "limit": limit, "offset": offset, "orders": page}


@router.get("/api/v1/users")
async def api_list_users(_admin: dict = Depends(require_admin)):
    users = db.list_users()
    for u in users:
        u.pop("password_hash", None)
    return {"users": users}


@router.post("/api/v1/users")
async def api_create_user(payload: UserCreateRequest, _admin: dict = Depends(require_admin)):
    import bcrypt as _bc
    if db.get_user(payload.username):
        raise HTTPException(status_code=409, detail="Username already exists.")
    hashed = _bc.hashpw(payload.password.encode("utf-8"), _bc.gensalt()).decode("utf-8")
    ok = db.create_user(payload.username, hashed, payload.role)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to create user.")
    db.add_audit_log("USER_CREATED", audit_ip(), f"User '{payload.username}' created (role {payload.role}).")
    return {"status": "Success", "username": payload.username, "role": payload.role}


@router.delete("/api/v1/users/{username}")
async def api_delete_user(username: str, _admin: dict = Depends(require_admin)):
    if username == os.getenv("ADMIN_USER", "admin"):
        raise HTTPException(status_code=400, detail="Cannot delete the bootstrap admin.")
    if not db.delete_user(username):
        raise HTTPException(status_code=404, detail="User not found.")
    db.add_audit_log("USER_DELETED", audit_ip(), f"User '{username}' deleted.")
    return {"status": "Deleted", "username": username}


@router.get("/api/v1/alerts")
async def api_list_alerts(_auth: dict = Depends(require_auth)):
    return {"alerts": STATE.get("price_alerts", [])}


@router.post("/api/v1/alerts")
async def api_create_alert(payload: PriceAlertCreate, _auth: dict = Depends(require_auth)):
    symbol = payload.symbol.upper()
    if symbol not in STATE["assets"]:
        raise HTTPException(status_code=400, detail=f"Unsupported symbol '{symbol}'.")
    alert = {
        "id": uuid.uuid4().hex[:10],
        "symbol": symbol,
        "direction": payload.direction,
        "target_price": payload.target_price,
        "note": payload.note,
        "triggered": False,
        "created_ts": time.time(),
    }
    STATE["price_alerts"].append(alert)
    _alerts_persist()
    return {"status": "Created", "alert": alert}


@router.delete("/api/v1/alerts/{alert_id}")
async def api_delete_alert(alert_id: str, _auth: dict = Depends(require_auth)):
    before = len(STATE["price_alerts"])
    STATE["price_alerts"] = [a for a in STATE["price_alerts"] if a.get("id") != alert_id]
    if len(STATE["price_alerts"]) == before:
        raise HTTPException(status_code=404, detail="Alert not found.")
    _alerts_persist()
    return {"status": "Deleted", "alert_id": alert_id}


@router.post("/api/v1/replay")
async def api_market_replay(payload: ReplayRequest, _auth: dict = Depends(require_auth)):
    symbol = payload.symbol.upper()
    if symbol not in STATE["assets"]:
        raise HTTPException(status_code=400, detail=f"Unsupported symbol '{symbol}'.")
    return await run_market_replay(symbol, payload.interval, payload.limit)


@router.post("/api/v1/signals/evaluate")
async def api_evaluate_signals(payload: SignalEvalRequest, _auth: dict = Depends(require_auth)):
    """Evaluates the whole signal catalogue over history -> ranking by Deflated Sharpe."""
    symbol = payload.symbol.upper()
    df = db.load_candles(symbol, limit=payload.limit)
    if df is None or df.empty or len(df) < 80:
        if symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            df = await fetch_historical_market_data(symbol)
        else:
            ymap = {"XAUUSD": "GC=F", "EURUSD": "EURUSD=X"}
            df = await fetch_yahoo_finance_candles(ymap.get(symbol, symbol), interval="1h", range_str="3mo")
    if df is None or df.empty or len(df) < 80:
        raise HTTPException(status_code=503, detail="Insufficient data to evaluate signals.")
    md = {"vpin": 0.5, "kyle_lambda": 0.0, "onchain_risk": _neutral(STATE.get("onchain_risk_score"), 0.5),
          "sentiment": _neutral(STATE.get("sentiment_index")), "funding_rate_8h": STATE.get("funding_rates", {}).get(symbol, 0.0),
          "market_avg_return": 0.0}
    return evaluate_all_signals(df, md)


@router.get("/api/v1/experiments")
async def api_list_experiments(_auth: dict = Depends(require_auth), limit: int = 100):
    return {"experiments": db.list_experiments(limit=limit)}


@router.post("/api/v1/experiments")
async def api_create_experiment(payload: ExperimentCreate, _auth: dict = Depends(require_auth)):
    eid = db.add_experiment(payload.hypothesis)
    if not eid:
        raise HTTPException(status_code=500, detail="Failed to register experiment.")
    return {"status": "Registered", "id": eid}


@router.get("/api/v1/events")
async def api_events(event_type: str = "", since: float = 0.0, limit: int = 200):
    """Replayable event journal (ticks + orders + decisions)."""
    return {"events": db.list_events(event_type=event_type, since=since, limit=limit)}


@router.get("/api/v1/ab")
async def api_ab(_auth: dict = Depends(require_auth)):
    """VISION §7.5: A/B paper comparison (baseline vs vol-targeted config)."""
    import math
    base, vol = STATE.get("ab_base", []), STATE.get("ab_vol", [])
    if len(base) < 10 or len(vol) < 10:
        return {"valid": False, "reason": "insufficient samples", "samples": len(base)}
    def _stats(curve):
        eq = np.array(curve)
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-9)
        sharpe = float(rets.mean() / rets.std() * np.sqrt(365 * 24)) if rets.std() > 0 else 0.0
        return {"return_pct": round((eq[-1] - 1.0) * 100.0, 3), "sharpe": round(sharpe, 3), "vol": round(float(rets.std()), 5)}
    s_base, s_vol = _stats(base), _stats(vol)
    leader = "vol_targeted" if s_vol["sharpe"] > s_base["sharpe"] else "baseline"
    return {"valid": True, "samples": len(base), "baseline": s_base, "vol_targeted": s_vol, "leader": leader}


@router.get("/api/v1/factors")
async def api_factors(_auth: dict = Depends(require_auth)):
    """VISION §6: factor exposures of the live equity curve (market/momentum/carry/vol)."""
    from core.factor_model import compute_factor_exposures
    eq = STATE.get("equity_history_demo" if STATE["mode"] == "DEMO" else "equity_history_real", [])
    if len(eq) < 20:
        return {"valid": False, "reason": "insufficient equity history"}
    rets = list(np.diff(eq) / np.maximum(np.array(eq[:-1]), 1e-9))
    mkt = [0.0] * len(rets)  # market proxy: equal-weight asset mean return (approx from price moves)
    mom = [0.0] * len(rets)
    car = [0.0] * len(rets)
    volr = [0.0] * len(rets)
    try:
        prices = [a.get("price") for a in STATE.get("assets", {}).values() if isinstance(a.get("price"), (int, float))]
        if len(prices) > 1:
            # market proxy: mean asset return per tick, broadcast to the equity
            # history length (the assets dict has one price per symbol, not per tick)
            p = np.array(prices, dtype=float)
            avg_ret = float(np.mean(np.diff(p) / np.maximum(p[:-1], 1e-9)))
            mkt = [avg_ret] * len(rets)
            mom = [float(np.sign(avg_ret)) * 0.001] * len(rets)
    except Exception:
        pass
    return compute_factor_exposures(rets, mkt, mom, car, volr)


@router.get("/api/v1/research")
async def api_research(_auth: dict = Depends(require_auth)):
    """VISION §3: hypothesis generator status (admitted signals + meta-prior)."""
    return hypothesis_generator.get_status()


@router.post("/api/v1/research/run")
async def api_research_run(_auth: dict = Depends(require_auth)):
    """Manually triggers one autonomous research cycle (uses cached real candles,
    with Yahoo/Binance fallback so it works anywhere)."""
    df = STATE.get("historical_bars")
    if df is None or df.empty or len(df) < 80:
        df = db.load_candles("BTCUSDT", limit=400)
    if df is None or df.empty or len(df) < 80:
        df = await fetch_yahoo_finance_candles("BTC-USD", interval="1h", range_str="3mo")
    if df is None or df.empty or len(df) < 80:
        raise HTTPException(status_code=503, detail="Insufficient data for research (all feeds unavailable).")
    md = {"vpin": 0.5, "kyle_lambda": 0.0, "sentiment": _neutral(STATE.get("sentiment_index")),
          "onchain_risk": _neutral(STATE.get("onchain_risk_score"), 0.5),
          "funding_rates": STATE.get("funding_rates", {}), "market_avg_return": 0.0}
    return hypothesis_generator.run_research_cycle(df, md, n_candidates=10)


@router.get("/api/v1/committee")
async def api_committee(_auth: dict = Depends(require_auth)):
    """VISION §6: AI risk committee status (vetoes + scores + budget)."""
    return {**risk_committee.status(), "risk_budget": STATE.get("risk_budget", {})}


@router.get("/api/v1/moe")
async def api_moe(_auth: dict = Depends(require_auth)):
    """VISION §2: mixture-of-experts status (votes, gate, offline training)."""
    return {
        "gate": STATE.get("moe_gate", {}),
        "last_votes": STATE.get("moe_votes", {}),
        "buffers": {h: len(e.buffer) for h, e in mixture_of_experts.experts.items()},
        "execution_bandit": execution_bandit.status(),
    }


@router.get("/api/v1/self")
async def api_self(_auth: dict = Depends(require_auth)):
    """VISION §7: self-assessment (divergence, reason effectiveness, honesty)."""
    reason_eff = {}
    try:
        raw = db.get_setting("reason_effectiveness")
        if raw:
            import json as _j
            reason_eff = _j.loads(raw)
    except Exception:
        pass
    return {
        "sim_divergence": STATE.get("sim_divergence", 0.0),
        "causal_parents": STATE.get("causal_parents", []),
        "conviction_threshold": STATE.get("conviction_threshold", 0.15),
        "no_trade_stats": STATE.get("no_trade_stats", {}),
        "reason_effectiveness": reason_eff,
        "strategy_exec_attribution": strategy_exec_attr.report(),
    }


@router.get("/api/v1/organization")
async def api_organization(_auth: dict = Depends(require_auth)):
    """VISION_FUTUR §1: desks + internal capital market allocations."""
    return organization.status()


@router.get("/api/v1/confidence")
async def api_confidence(_auth: dict = Depends(require_auth)):
    """VISION_FUTUR §8: composite confidence index + size factor."""
    return {"index": STATE.get("confidence_index", 100), "factor": STATE.get("confidence_factor", 1.0),
            "live_p_value": STATE.get("live_p_value", 0.5)}


@router.get("/api/v1/supervisor")
async def api_supervisor(_auth: dict = Depends(require_auth)):
    """VISION_FUTUR §5b: vital signs."""
    return {"issues": supervisor.check(force=True), "last_tick_ts": STATE.get("last_tick_ts", 0.0)}


@router.post("/api/v1/assistant/ask")
async def api_assistant(payload: AskRequest, _auth: dict = Depends(require_auth)):
    """VISION_FUTUR §6: the operator talks to the bot (answers grounded in real data)."""
    context = {
        "last_price": STATE.get("last_price", 0.0),
        "current_equity": STATE.get("current_equity", 0.0),
        "regime_name": STATE.get("regime_name", "?"),
        "regime_probs": STATE.get("regime_probs", {}),
        "confidence_index": STATE.get("confidence_index", 100),
        "positions": [p.get("symbol") for p in db.get_positions()],
        "desk_allocations": STATE.get("desk_allocations", {}),
        "admitted_signals": list(hypothesis_generator.admitted.keys()),
    }
    return {"answer": await answer_question_async(payload.question, context)}


@router.post("/api/v1/narrative")
async def api_narrative(_auth: dict = Depends(require_auth)):
    """VISION_FUTUR §3/§6: daily market narrative (LLM via OpenRouter, structured fallback)."""
    from core.reporting import build_daily_report
    report = build_daily_report(STATE, db)
    narrative = await daily_market_narrative_async(report, STATE)
    STATE["last_narrative"] = narrative
    return {"narrative": narrative}


@router.post("/api/v1/chaos")
async def api_chaos(_auth: dict = Depends(require_auth)):
    """VISION_FUTUR §5c: chaos self-test - simulate a feed outage, verify safe HALT."""
    res = chaos_cut_feed(STATE, db, duration_seconds=15.0)
    return res


@router.post("/api/v1/approve")
async def api_approve(_auth: dict = Depends(require_auth)):
    """VISION_FUTUR §6: approve all pending consultative-mode proposals."""
    pending = list(STATE.get("pending_approvals", []))
    STATE["pending_approvals"] = []
    approved = 0
    for p in pending:
        try:
            submit_order_via_oms(p["symbol"], p["side"], p["qty"], p["price"], p["mode"], p["strategy"],
                                 exchange="Binance" if p["symbol"] in ("BTCUSDT","ETHUSDT","SOLUSDT") else "Bybit")
            db.add_audit_log("APPROVED_ORDER", audit_ip(),
                             f"Operator approved {p['side']} {p['qty']} {p['symbol']} ({p['strategy']})")
            approved += 1
        except Exception as e:
            logger.warning(f"Approved order failed: {e}")
    return {"approved": approved, "pending_left": len(STATE["pending_approvals"])}


@router.get("/api/v1/research/export")
async def api_research_export(_auth: dict = Depends(require_auth)):
    """VISION_FUTUR §7: export meta-prior + admitted signals (knowledge sharing)."""
    return hypothesis_generator.get_status()


@router.get("/api/v1/research/packs/{name}")
async def api_research_pack(name: str, _auth: dict = Depends(require_auth)):
    """VISION_FUTUR §7: strategy pack export for the marketplace."""
    cand = hypothesis_generator.admitted.get(name)
    if not cand:
        raise HTTPException(status_code=404, detail="Signal pack not found.")
    return {"pack": cand, "format": "quant-portal-signal-pack-v1"}


@router.post("/api/v1/webhook/trade")
async def webhook_trade(payload: WebhookTradeRequest):
    """
    AUDIT C8: external alert webhook (TradingView, Pine Script, custom bots).
    Protected by WEBHOOK_SECRET env (constant-time comparison).
    Executes a market order via the OMS path, sized to a configurable % of capital.
    """
    expected = os.getenv("WEBHOOK_SECRET", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Webhooks not configured (set WEBHOOK_SECRET).")
    import hmac as _hmac
    if not _hmac.compare_digest(payload.secret, expected):
        raise HTTPException(status_code=401, detail="Invalid webhook secret.")

    symbol = payload.symbol.upper()
    if symbol not in STATE["assets"]:
        raise HTTPException(status_code=400, detail=f"Unsupported symbol '{symbol}'.")

    mode = STATE["mode"]
    bal_key = "balance_demo" if mode == "DEMO" else "balance_real"
    capital = STATE.get(bal_key, 0.0)
    price = payload.price if payload.price > 0 else (STATE["assets"].get(symbol, {}).get("price") or 0.0)
    if price <= 0:
        raise HTTPException(status_code=503, detail="No live price available for this symbol.")

    side = "BUY" if payload.action.upper() == "BUY" else "SELL"
    webhook_size_pct = settings.get_float("trading", "webhook_size_pct", 0.10)
    qty = payload.qty if payload.qty > 0 else (capital * webhook_size_pct) / price
    qty = format_exchange_size(symbol, qty, price)

    try:
        res = submit_order_via_oms(symbol, side, qty, price, mode, "WEBHOOK",
                                   exchange="Binance" if symbol in ("BTCUSDT","ETHUSDT","SOLUSDT") else "Bybit")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Order submission failed: {e}")

    # ledger (mirror of the loop's update)
    if side == "BUY":
        STATE[bal_key] -= qty * price * 1.001
    else:
        STATE[bal_key] += qty * price * 0.999
    db.update_position(symbol, qty if side == "BUY" else -qty, price, mode)
    db.add_order(symbol=symbol, side=side, price=price, qty=qty, status="FILLED",
                 mode=mode, strategy="WEBHOOK", order_type="MARKET")
    db.add_audit_log("WEBHOOK_TRADE", audit_ip(), f"Webhook {side} {qty} {symbol} @ {price:.2f}")
    platform_metrics.ORDERS_TOTAL.labels(mode=mode, side=side).inc()
    STATE["last_order_times"][symbol] = time.time()
    return {"status": "FILLED", "symbol": symbol, "side": side, "qty": qty, "price": price}


@router.get("/api/v1/copy/mirror-status")
async def api_copy_mirror_status(_auth: dict = Depends(require_auth)):
    """Real copy-trading status: followed traders + current mirror signals."""
    signals = {}
    for tid, alloc in list(copy_manager.copied_traders.items()):
        positions = fetch_trader_positions(tid)
        my_pos = {p["symbol"]: float(p["qty"]) for p in db.get_positions()}
        trader = copy_manager.traders.get(tid)
        orders = build_mirror_orders(positions, my_pos,
                                     allocated_capital=float(alloc.get("allocated_capital", 0.0)),
                                     trader_account_value=float(getattr(trader, "account_value", 0.0) or 0.0))
        signals[tid[:12]] = {
            "mode": alloc.get("mode", "FOLLOW_ONLY"),
            "trader_positions": [p["coin"] for p in positions[:8]],
            "mirror_orders": orders[:8],
        }
    return {
        "execution_mode": os.getenv("COPYTRADE_EXECUTION", "signal_only"),
        "following": copy_manager.copied_traders,
        "mirror_signals": signals,
        "summary": mirror_status_text(copy_manager.copied_traders),
    }


@router.get("/api/telemetry")
async def get_telemetry_rest():
    """
    REST Fallback API to query telemetry.
    Ensures 100% platform connectivity even when WebSockets are blocked by client browser or proxy!
    """
    return JSONResponse(compile_telemetry_data())


@router.get("/metrics")
async def get_prometheus_metrics():
    """
    Prometheus text exposition (scraped by the bundled prometheus.yml).
    Institutional observability for Grafana dashboards & auto-scaling alerts.
    """
    platform_metrics.refresh_uptime()
    return Response(content=platform_metrics.get_metrics_text(), media_type="text/plain; version=0.0.4; charset=utf-8")


@router.get("/telegram_mini_app.html", response_class=HTMLResponse)
async def get_telegram_mini_app_alias(request: Request):
    """
    Backward-compatible alias for those who configured the old filename in
    @BotFather. Redirects to /telegram.
    """
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/telegram")


@router.get("/api/status")
async def get_status():
    # Safe serialization - remove non-serializable objects (DataFrames, etc.)
    safe_state = {k: v for k, v in STATE.items() if not isinstance(v, (pd.DataFrame, pd.Series)) and k != "ppo_buffer"}
    # Also sanitize NaN/Infinity floats (invalid JSON) and convert datetimes
    safe_state = serialize_helper(safe_state)
    return JSONResponse(safe_state)


@router.get("/api/history")
async def get_history_endpoint(timeframe: str = "1h", limit: int = 120, offset: int = 0):
    """
    Returns historical candle bars for different timeframes (1h, 4h, 1d).
    Uses persistent database caching and Binance API.
    """
    valid_timeframes = ["1h", "4h", "1d"]
    if timeframe not in valid_timeframes:
        timeframe = "1h"
        
    interval = "1h" if timeframe == "1h" else "4h" if timeframe == "4h" else "1d"
    
    # Check persistent database cache
    cache_symbol = f"BTCUSDT_{timeframe}"
    df = db.load_candles(cache_symbol, limit=120)
    
    if df.empty or len(df) < 120:
        # Fetch from Binance API
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit=120"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    bars = []
                    for b in data:
                        bars.append({
                            "timestamp": pd.to_datetime(b[0], unit='ms'),
                            "open": float(b[1]),
                            "high": float(b[2]),
                            "low": float(b[3]),
                            "close": float(b[4]),
                            "volume": float(b[5])
                        })
                    df = pd.DataFrame(bars).set_index("timestamp")
                    db.save_candles(cache_symbol, df)
        except Exception as e:
            logger.warning(f"Failed to fetch Binance klines for {timeframe}: {str(e)}")

        # Fallback: Yahoo Finance (works even when Binance is geo-blocked / offline)
        if df.empty or len(df) < 120:
            logger.info(f"Binance unavailable for {timeframe}. Falling back to Yahoo Finance...")
            yahoo_ticker = "BTC-USD"
            try:
                df_yahoo = await fetch_yahoo_finance_candles(yahoo_ticker, interval=interval, range_str="5d")
                if df_yahoo is not None and not df_yahoo.empty:
                    df = df_yahoo
                    db.save_candles(cache_symbol, df)
            except Exception as e:
                logger.warning(f"Yahoo Finance fallback failed for {timeframe}: {str(e)}")
            
    if df.empty:
        logger.error(f"Failed to load historical candles for {timeframe}. No database or CEX feed active.")
        raise HTTPException(status_code=503, detail="Historical market data currently unavailable.")
        
    prices = df['close'].values.tolist()
    timestamps = [str(t) for t in df.index]
    limit = max(10, min(1000, limit))
    offset = max(0, offset)
    return {
        "timeframe": timeframe,
        "total": len(prices),
        "limit": limit,
        "offset": offset,
        "prices": prices[offset:offset + limit],
        "timestamps": timestamps[offset:offset + limit]
    }


@router.post("/api/login")
async def login(payload: LoginRequest):
    """
    Authenticates an operator and returns a JWT bearer token.
    AUDIT C7 (multi-user): authenticates against the `users` table (bcrypt),
    with the env ADMIN_USER/ADMIN_PASSWORD as the bootstrap admin fallback.
    On first successful bootstrap login the DB hash is upgraded to bcrypt.
    Optional TOTP second factor when ADMIN_TOTP_SECRET is set.
    """
    import hmac as _hmac
    import bcrypt as _bc

    admin_user = os.getenv("ADMIN_USER", "admin")
    admin_pass = os.getenv("ADMIN_PASSWORD", "ChangeMe!Institutionnel2026")
    totp_secret = os.getenv("ADMIN_TOTP_SECRET", "")

    # 1) Try the database users table first
    db_user = db.get_user(payload.username)
    role = Roles.VIEWER
    ok = False
    if db_user and db_user.get("password_hash"):
        ph = db_user.get("password_hash")
        if ph and not ph.startswith("$2") and ph != "hash_admin_secret":
            ok = False  # malformed hash -> cannot verify
        else:
            try:
                ok = _bc.checkpw(payload.password.encode("utf-8"), ph.encode("utf-8"))
            except Exception:
                ok = False
        role = db_user.get("role") or Roles.VIEWER

    # 2) Bootstrap admin fallback (env) - also upgrades the DB hash on success
    if not ok and payload.username == admin_user and _hmac.compare_digest(payload.password, admin_pass):
        ok = True
        role = Roles.ADMIN
        try:
            db.create_user(admin_user, _bc.hashpw(admin_pass.encode("utf-8"), _bc.gensalt()).decode("utf-8"), Roles.ADMIN)
        except Exception as e:
            logger.warning(f"Bootstrap admin hash upgrade skipped: {e}")

    if not ok:
        db.add_audit_log("AUTH_FAILURE", audit_ip(), f"Failed login attempt for '{payload.username}'.")
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    if totp_secret:  # TOTP second factor enforced whenever configured
        import pyotp as _pyotp
        totp = _pyotp.TOTP(totp_secret)
        if not totp.verify(payload.totp_code, valid_window=1):
            raise HTTPException(status_code=401, detail="Invalid TOTP code.")

    user_id = int(db_user.get("id") or 1) if db_user else 1
    token = AuthManager.create_jwt_token(user_id, payload.username, role)
    logger.info(f"🔐 Operator '{payload.username}' authenticated successfully (role {role}).")
    return {"token": token, "role": role, "username": payload.username}


@router.post("/api/toggle-strategy")
async def toggle_strategy(payload: StrategyToggle, _auth: dict = Depends(require_auth)):
    strategy = next((s for s in strategies_list if s.name == payload.name), None)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
        
    strategy.enabled = payload.enabled
    db.add_audit_log(
        "STRATEGY_TOGGLED", 
        audit_ip(), 
        f"Modified strategy '{payload.name}' enabled status to {payload.enabled}."
    )
    return {"status": "Success", "message": f"Strategy {payload.name} modified to {payload.enabled}"}


@router.post("/api/toggle-bot")
async def toggle_bot(payload: BotToggleRequest, _auth: dict = Depends(require_auth)):
    STATE["is_running"] = payload.is_running
    action_str = "STARTED" if payload.is_running else "PAUSED"
    db.add_audit_log(
        "BOT_STATE_CHANGED", 
        audit_ip(), 
        f"Automated trading loop has been manually {action_str}."
    )
    return {"status": "Success", "message": f"Automated trading loop {action_str} successfully."}


@router.post("/api/set-demo-balance")
async def set_demo_balance(payload: SetBalanceRequest, _auth: dict = Depends(require_auth)):
    if payload.balance <= 0:
        raise HTTPException(status_code=400, detail="Balance must be positive.")
    STATE["balance_demo"] = payload.balance
    risk_manager.set_initial_capital(payload.balance)
    STATE["initial_capital_demo"] = payload.balance
    STATE["current_equity"] = payload.balance
    STATE["equity_history_demo"] = [payload.balance]
    
    # Persist in DB setting so it survives server restart and browser refresh!
    db.save_setting("balance_demo", str(payload.balance))
    db.save_setting("initial_capital_demo", str(payload.balance))
    
    db.add_audit_log(
        "DEMO_BALANCE_RESET", 
        audit_ip(), 
        f"Demo balance has been manually reset to {payload.balance} USD."
    )
    return {"status": "Success", "message": f"Demo balance successfully set to {payload.balance} USD."}


@router.post("/api/retrain")
async def trigger_manual_retrain(_auth: dict = Depends(require_auth)):
    df = STATE["historical_bars"]
    if df is None:
        raise HTTPException(status_code=400, detail="No historical bars cache loaded yet.")
        
    res = mlops_trainer.execute_pipeline(df)
    return JSONResponse(res)


@router.post("/api/monte-carlo")
async def trigger_monte_carlo(_auth: dict = Depends(require_auth)):
    """
    On-Demand Monte Carlo Stress Testing API endpoint.
    Runs 10,000 simulations and returns structural safety metrics.
    """
    df = STATE["historical_bars"]
    if df is None:
        raise HTTPException(status_code=400, detail="No historical data loaded yet.")
        
    current_p = STATE["last_price"]
    if current_p is None:
        raise HTTPException(status_code=400, detail="No live price fetched yet. Please wait for WebSockets synchronization.")
        
    vols = df['close'].pct_change().std()
    res = monte_carlo_tester.execute_stress_test(
        initial_capital=STATE["balance_demo"] if STATE["mode"] == "DEMO" else STATE["balance_real"],
        current_price=current_p,
        historical_volatility=vols if not np.isnan(vols) else 0.02
    )
    
    db.add_audit_log(
        "MONTE_CARLO_TEST_EXECUTED",
        audit_ip(),
        f"Executed 10,000 Monte Carlo stress-testing simulations. Survival rate: {res['survival_probability_pct']:.2f}%."
    )
    
    return JSONResponse(res)


@router.post("/api/risk-settings")
async def update_risk_settings(payload: RiskSettingsUpdate, _auth: dict = Depends(require_auth)):
    risk_manager.params.update(payload.dict())
    db.add_audit_log(
        "RISK_SETTINGS_UPDATED", 
        audit_ip(), 
        f"Updated Risk thresholds: Max daily drawdown to {payload.max_daily_drawdown_pct*100:.2f}%."
    )
    return {"status": "Success", "message": "Risk management policies updated successfully."}


@router.post("/api/keys")
async def store_keys(payload: KeyStorage, _auth: dict = Depends(require_auth)):
    db.save_setting("api_keys_rotated_at", str(time.time()))  # audit B3-6: rotation tracking
    db.save_setting(f"{payload.exchange}_api_key", payload.api_key, encrypt=True)
    db.save_setting(f"{payload.exchange}_secret_key", payload.secret_key, encrypt=True)
    db.add_audit_log(
        "API_KEYS_STORED", 
        audit_ip(), 
        f"Stored and encrypted API key pairs for exchange {payload.exchange}."
    )
    return {"status": "Success", "message": f"Encrypted keys stored for {payload.exchange}."}


@router.post("/api/2fa-switch")
async def switch_mode(payload: SwitchModeRequest, _auth: dict = Depends(require_auth)):
    """
    Secures the Demo <-> Real trading modes transitions.
    AUDIT B3: hardcoded 2FA test codes ("123456"/"888888") REMOVED - they were a
    backdoor to REAL mode. Factors accepted now:
      - a valid TOTP code when ADMIN_TOTP_SECRET is configured, or
      - an EVM wallet address (0x...) provided by the operator.
    AUDIT D1: switching to REAL requires the autopilot paper-validation period
    to have elapsed (config autopilot.paper_validation_required / min_paper_validation_days).
    """
    global ccxt_client
    totp_secret = os.getenv("ADMIN_TOTP_SECRET", "")
    is_wallet = payload.verification_2fa.startswith("0x") and len(payload.verification_2fa) == 42
    is_totp = False
    if totp_secret:
        try:
            import pyotp
            is_totp = pyotp.TOTP(totp_secret).verify(payload.verification_2fa, valid_window=1)
        except Exception:
            is_totp = False

    if not (is_wallet or is_totp):
        db.add_audit_log("AUTH_FAILURE", audit_ip(), f"Failed 2FA transit attempt to mode {payload.target_mode}.")
        raise HTTPException(status_code=401, detail="Invalid 2FA factor. Security block triggered.")
        
    if payload.target_mode not in ["DEMO", "REAL"]:
        raise HTTPException(status_code=400, detail="Invalid target trading mode.")
        
    if payload.target_mode == "REAL":
        # AUTOPILOT GATE (audit D1): paper validation period must have elapsed
        if settings.get_bool("autopilot", "paper_validation_required", True):
            min_days = settings.get_int("autopilot", "min_paper_validation_days", 7)
            first_start = float(db.get_setting("platform_first_start_ts") or time.time())
            db.save_setting("platform_first_start_ts", str(first_start))
            elapsed_days = (time.time() - first_start) / 86400.0
            if elapsed_days < min_days:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"Autopilot gate: REAL mode requires {min_days} days of validated "
                        f"paper/DEMO trading first (currently {elapsed_days:.1f} days). "
                        f"This protects you from deploying untested logic with real money."
                    ),
                )

        # Force reload keys from database
        ccxt_client = None
        client = get_ccxt_client()
        if not client:
            raise HTTPException(
                status_code=400, 
                detail="Real Mode denied. Please configure valid and active Exchange API Keys in the dashboard first."
            )
            
    STATE["mode"] = payload.target_mode
    db.add_audit_log(
        "TRADING_MODE_CHANGED", 
        audit_ip(), 
        f"Successfully changed system trading mode to {payload.target_mode} via authorization {payload.verification_2fa[:12]}..."
    )
    return {"status": "Success", "message": f"Platform successfully switched to {payload.target_mode} Mode."}


@router.post("/api/copy-trade")
async def manage_copytrade(payload: CopyTradeRequest, _auth: dict = Depends(require_auth)):
    if payload.action == "START":
        # LOT 7 (PDF Pilier J) : plafond de capital par trader copié
        try:
            _total_cap = STATE["balance_demo"] if STATE["mode"] == "DEMO" else STATE["balance_real"]
            _cap_ok, _cap_msg = copy_manager.check_trader_risk(payload.trader_id, _total_cap) if payload.trader_id in copy_manager.copied_traders else (True, "")
            if not _cap_ok:
                return {"ok": False, "error": f"Risque copytrading: {_cap_msg}"}
        except Exception:
            pass
        ok, msg = copy_manager.start_copying(payload.trader_id, payload.allocated_capital)
        if ok:
            db.save_copy_allocation(payload.trader_id, payload.allocated_capital, 1)
            db.add_audit_log("COPY_START", audit_ip(), f"Started copytrading {payload.trader_id} with {payload.allocated_capital} USD allocation.")
            return {"status": "Success", "message": msg}
        raise HTTPException(status_code=400, detail=msg)
    else:
        ok, msg = copy_manager.stop_copying(payload.trader_id)
        if ok:
            db.save_copy_allocation(payload.trader_id, 0.0, 0)
            db.add_audit_log("COPY_STOP", audit_ip(), f"Stopped copytrading {payload.trader_id}.")
            return {"status": "Success", "message": msg}
        raise HTTPException(status_code=400, detail=msg)


@router.get("/api/walkforward-stats")
async def api_walkforward_stats(_auth: dict = Depends(require_auth)):
    """
    Poids Walk-Forward des stratégies (utilisé par la mini-app).
    Retourne les poids actuels du MetaAllocationEngine (bandit + PnL réel).
    """
    try:
        weights = meta_engine.get_strategy_weights()
        return {"weights": weights, "ts": time.time()}
    except Exception as e:
        return {"weights": {}, "error": str(e)}


@router.get("/api/v1/paper-validation")
async def api_v1_paper_validation(_auth: dict = Depends(require_auth)):
    """
    P0-6 (audit §5-P0-6) : suivi du paper-trading DATÉ et CONTINU exigé avant
    le mode REAL. Jours actifs, série consécutive, seuil requis, statut.
    Un jour ne compte que si le bot a réellement tourné ce jour-là.
    """
    return _paper_validation_stats()


@router.get("/api/v1/final-scale")
async def api_v1_final_scale(_auth: dict = Depends(require_auth)):
    """
    P0-4 (audit §2.1) : état de l'observation empirique de final_scale —
    distribution p10/p50/p90 sur la fenêtre glissante de 48h + alerte si
    p50 < 20 % (chaîne de facteurs auto-amplifiée, pas un seuil de signal).
    """
    stats = STATE.get("final_scale_stats") or _final_scale_stats()
    samples = STATE.get("final_scale_samples", [])
    return {
        "stats": stats,
        # recalcul à la volée : le rapport ne tourne que toutes les 60 min
        "limiting_factor": _limiting_factor_stats(),
        "signal_distribution": _signal_stats(),
        "samples_count": len(samples),
        "window_hours": FINAL_SCALE_WINDOW_HOURS,
        "collected_since": samples[0]["ts"] if samples else None,
        "alert_p50_below_20pct": bool(stats and stats["p50"] < 0.20),
        "note": "Observation empirique à poursuivre 24-48h de trading continu avant diagnostic.",
        "ts": time.time(),
    }


@router.get("/api/v1/honesty")
async def api_v1_honesty(_auth: dict = Depends(require_auth)):
    """
    LOT 9 (PDF Faille 7) : étiquetage honnête des modules.
    PRODUCTION / EXPÉRIMENTAL / ÉDUCATIF + gardes associées + note.
    """
    return {
        "modules": get_module_status(),
        "summary": status_summary(),
        "rule": "Un module ÉDUCATIF ne doit JAMAIS influencer une décision de trading.",
        "ts": time.time(),
    }


@router.get("/api/v1/attribution")
async def api_v1_attribution(_auth: dict = Depends(require_auth)):
    """
    LOT 8 (PDF Pilier Q) : attribution de performance — d'où vient chaque
    dollar (facteur, régime, actif, stratégie) + métriques de qualité.
    """
    return {
        "attribution": STATE.get("attribution_report", {}),
        "quality_metrics": STATE.get("quality_metrics", {}),
        "costs": STATE.get("cost_metrics", {}),
        "ts": time.time(),
    }


@router.post("/api/v1/stress")
async def api_v1_stress(_auth: dict = Depends(require_auth)):
    """
    LOT 8 (PDF Pilier N) : stress test par SCÉNARIOS de crises RÉELLES
    (COVID 2020, krach 2018, FTX 2022) sur le portefeuille COMPLET.
    """
    try:
        _positions = db.get_positions()
        _prices = STATE.get("last_known_prices", {})
        _bal = STATE["balance_demo"] if STATE["mode"] == "DEMO" else STATE["balance_real"]
        _res = scenario_tester.run_stress(_positions, _bal, _prices)
        STATE["stress_test_report"] = _res
        db.add_audit_log("STRESS_TEST", audit_ip(),
                         f"Stress crises réelles: {_res['status']} (pire {_res['worst']} {_res['worst_loss_pct']}%)")
        return _res
    except Exception as e:
        import traceback
        logger.error(f"STRESS TEST endpoint error: {e}\n{traceback.format_exc()}")
        return {"ok": False, "error": str(e)}


@router.post("/api/risk-state/reset")
async def reset_risk_state(_auth: dict = Depends(require_auth)):
    """
    LOT 2 : remet la machine à états NORMAL/CAUTION/HALT à NORMAL.
    L'opérateur humain reste le décideur final (mentalité n°10).
    """
    try:
        changed = risk_state.reset(reason="api")
        STATE["risk_state"] = risk_state.to_dict()
        if changed:
            db.add_audit_log("RISK_STATE_RESET", audit_ip(), "État risque remis à NORMAL via API")
        return {"ok": True, "risk_state": risk_state.to_dict()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/api/kill-switch")
async def engage_kill_switch(_auth: dict = Depends(require_auth)):
    STATE["kill_switch_active"] = True
    STATE["is_running"] = False
    
    positions = db.get_positions()
    active_mode = STATE["mode"]
    active_balance_key = "balance_demo" if active_mode == "DEMO" else "balance_real"
    client = get_ccxt_client() if active_mode == "REAL" else None
    
    for p in positions:
        try:
            asset_price = STATE["assets"].get(p['symbol'], {}).get("price")
            if asset_price is None:
                asset_price = STATE.get("last_known_prices", {}).get(p['symbol'])
            if asset_price is None:
                logger.error(f"Kill switch: cannot close {p['symbol']} (no real price) - manual action required.")
                continue
            if active_mode == "REAL" and client:
                client.create_order(symbol=p['symbol'].replace("USDT", "/USDT"), type='market', side='sell', amount=p['qty'])
                
            close_val = p['qty'] * asset_price * 0.999
            STATE[active_balance_key] += close_val
            db.update_position(p['symbol'], 0, 0, active_mode)
            db.add_order(
                symbol=p['symbol'],
                side="SELL",
                price=asset_price * 0.999,
                qty=p['qty'],
                status="FORCE_LIQUIDATED",
                mode=active_mode,
                strategy="EMERGENCY_KILL_SWITCH",
                order_type="MARKET"
            )
        except Exception as exc:
            logger.error(f"Emergency close failed for {p['symbol']}: {str(exc)}")
            
    db.add_audit_log("KILL_SWITCH_ENGAGED", audit_ip(), "Global KILL SWITCH activated manually. Closed all open exposures.")
    return {"status": "Success", "message": "EMERGENCY GLOBAL KILL SWITCH ENGAGED. All exposures flatted, system locked."}


@router.post("/api/reset-bot")
async def reset_bot(_auth: dict = Depends(require_auth)):
    STATE["kill_switch_active"] = False
    STATE["is_running"] = True
    risk_manager.circuit_breaker_active = False
    db.add_audit_log("SYSTEM_RESET", audit_ip(), "Unlocked system state from emergency stop.")
    return {"status": "Success", "message": "System successfully unlocked and restarted."}


@router.post("/api/run-backtest")
async def run_backtest_handler(_auth: dict = Depends(require_auth)):
    df = STATE["historical_bars"]
    if df is None:
        raise HTTPException(status_code=400, detail="Historical bars not loaded yet.")
        
    # LOT 8 (PDF Pilier N) : AUDIT DES BIAIS avant tout backtest (look-ahead,
    # survivorship, slippage). Un backtest qui échoue à l'audit est REJETÉ.
    try:
        _bias = audit_backtest(
            df,
            assets_universe=list(STATE["assets"].keys()),
            assets_tested=list(STATE["assets"].keys()),
            slippage_bps=5.0,          # coûts réalistes (jamais 0)
            commission_pct=0.001,
        )
        STATE["last_bias_audit"] = _bias
        if _bias["status"] == "REJECTED":
            db.add_audit_log("BACKTEST_BIAS_REJECTED", audit_ip(),
                             f"Backtest rejeté: {_bias['issues']}")
            return {"ok": False, "status": "REJECTED", "bias_audit": _bias}
        db.add_audit_log("BACKTEST_BIAS_OK", audit_ip(),
                         f"Audit biais passé (score {_bias['score']})")
    except Exception as _be:
        logger.warning(f"Bias audit failed: {_be}")
        _bias = {"status": "UNKNOWN"}

    backtester = EventDrivenBacktester(initial_capital=100000.0)
    local_detector = MarketRegimeDetector()
    # P0-5 (audit §4.9) : même archi que le live (hidden_dim=24).
    local_predictor = LSTMLikePredictor(5, 24)
    local_ppo = PPOTRAgent(4, 1)
    
    split = int(len(df) * 0.6)
    train_slice = df.iloc[:split]
    test_slice = df.iloc[split:]
    
    train_returns = train_slice['close'].pct_change().dropna().values
    train_vols = train_slice['close'].pct_change().rolling(5).std().dropna().values
    min_l = min(len(train_returns), len(train_vols))
    local_detector.fit(np.column_stack((train_returns[-min_l:], train_vols[-min_l:])))
    
    feats = []
    labs = []
    pct_df = train_slice[['close', 'volume', 'high', 'low', 'open']].pct_change().fillna(0)
    for i in range(5, len(pct_df) - 1):
        feats.append(pct_df.iloc[i-5:i].values)
        labs.append(pct_df['close'].iloc[i])
    local_predictor.fit(feats, np.array(labs))
    
    metrics = backtester.run(
        test_slice,
        meta_engine,
        risk_manager,
        local_detector,
        local_predictor,
        local_ppo
    )
    # LOT 8 (PDF Pilier N) : le rapport de backtest inclut l'audit des biais
    if isinstance(metrics, dict):
        metrics["bias_audit"] = _bias
    return JSONResponse(metrics)

