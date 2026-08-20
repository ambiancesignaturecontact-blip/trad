"""
Schedulers / boucles de fond extraits de main.py (LOT 7, P1-7 audit §4.1).
Corps STRICTEMENT inchangés ; les symboles partagés viennent de main via
`from main import *` (main est complet quand ce module est importé, en fin
de main.py).
"""
import json
import os
import time
import asyncio

import main  # noqa: F401
from main import *  # noqa: F401,F403
# helpers _privés de main utilisés par ces schedulers (non couverts par `*`)
from main import _final_scale_report, _load_final_scale_samples, _mark_paper_validation_day  # noqa: F401
async def final_scale_stats_loop():
    """P0-4 : loggue la distribution réelle de final_scale (p10/p50/p90).
    Au démarrage : rechargement de l'échantillon persisté + premier rapport
    immédiat, puis rapport toutes les 60 min (fenêtre glissante de 48h)."""
    try:
        _load_final_scale_samples()
        _final_scale_report()   # premier diagnostic dès le boot
    except Exception as e:
        logger.warning(f"final_scale stats boot report error: {e}")
    while True:
        await asyncio.sleep(3600)
        try:
            _final_scale_report()
        except Exception as e:
            logger.warning(f"final_scale stats loop error: {e}")



async def concierge_scheduler():
    """AUDIT D4: daily Telegram risk-concierge digest at a configured hour."""
    while True:
        hour = settings.get_int("alerts", "daily_digest_hour_utc", 18)
        try:
            now = time.gmtime()
            next_run = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hour, 0, 0, 0, 0, -1))
            if next_run <= time.time():
                next_run += 86400
            await asyncio.sleep(next_run - time.time())
            report = build_daily_report(STATE, db)
            # VISION_FUTUR §8: proactive health alert when the bot distrusts itself
            if report.get("health_score", 100) < 60:
                await telegram_bot.send_push_notification(
                    f"🚨 *SANTÉ FAIBLE : {report['health_score']}/100*\n"
                    f"Raisons : {', '.join(report.get('health_reasons', [])[:4])}\n"
                    f"Le bot réduit automatiquement ses tailles."
                )
            try:
                await telegram_bot.send_push_notification(build_concierge_message(report))
                # VISION_FUTUR §3: LLM narrative appended (OpenRouter or structured)
                try:
                    _narr = await daily_market_narrative_async(report, STATE)
                    if _narr:
                        STATE["last_narrative"] = _narr
                        await telegram_bot.send_push_notification(_narr)
                except Exception as ne:
                    logger.warning(f"Narrative skipped: {ne}")
                logger.info("✅ Concierge quotidien envoyé")
            except Exception as e:
                logger.warning(f"Concierge Telegram envoi échoué: {e}")
            db.add_audit_log("DAILY_CONCIERGE", audit_ip(), "Daily report generated.")
        except Exception as e:
            logger.warning(f"Concierge scheduler error: {e}")
            await asyncio.sleep(3600)



async def copy_mirror_scheduler():
    """
    REAL copy-trading execution (VISION §5): every 10 min, fetch the followed
    trader's real positions (Hyperliquid public API), compute the scaled delta
    vs our portfolio and execute via OMS when COPYTRADE_EXECUTION=auto + keys.
    Otherwise logs honest SIGNAL_ONLY mirror signals.
    """
    while True:
        await asyncio.sleep(600)
        try:
            if not copy_manager.copied_traders:
                continue
            # LOT 7 (PDF Pilier J) : stop global des traders sous-performants
            try:
                _total_cap = STATE["balance_demo"] if STATE["mode"] == "DEMO" else STATE["balance_real"]
                _stopped = copy_manager.enforce_trader_risk(_total_cap)
                if _stopped:
                    try:
                        asyncio.create_task(telegram_bot.send_push_notification(
                            f"🛑 *COPY STOP*\nTrader(s) arrêté(s) (stop global) : {', '.join(_stopped)}"))
                    except Exception:
                        pass
            except Exception:
                pass
            exec_mode = os.getenv("COPYTRADE_EXECUTION", "signal_only")
            for tid, alloc in list(copy_manager.copied_traders.items()):
                trader = copy_manager.traders.get(tid)
                acct_value = float(getattr(trader, "account_value", 0.0) or 0.0)
                positions = fetch_trader_positions(tid)
                if not positions:
                    continue
                my_pos = {p["symbol"]: float(p["qty"]) for p in db.get_positions()}
                orders = build_mirror_orders(
                    positions, my_pos,
                    allocated_capital=float(alloc.get("allocated_capital", 0.0)),
                    trader_account_value=acct_value,
                    max_asset_pct=settings.get_float("risk", "max_per_asset_pct", 0.25),
                )
                if not orders:
                    continue
                if STATE["mode"] == "REAL" and exec_mode == "auto" and get_ccxt_client():
                    for o in orders:
                        try:
                            submit_order_via_oms(o["symbol"], o["side"], o["qty"], STATE["assets"][o["symbol"]]["price"],
                                                 "REAL", "COPY_MIRROR",
                                                 exchange="Binance" if o["symbol"] in ("BTCUSDT","ETHUSDT","SOLUSDT") else "Bybit")
                            db.add_order(symbol=o["symbol"], side=o["side"], price=STATE["assets"][o["symbol"]]["price"],
                                         qty=o["qty"], status="FILLED", mode="REAL", strategy="COPY_MIRROR", order_type="MARKET")
                            db.add_audit_log("COPY_MIRROR_EXECUTED", audit_ip(),
                                             f"Mirror {o['side']} {o['qty']:.5f} {o['symbol']} (trader {tid[:10]}…)")
                            platform_metrics.ORDERS_TOTAL.labels(mode="REAL", side=o["side"]).inc()
                        except Exception as oe:
                            logger.warning(f"Mirror order failed {o['symbol']}: {oe}")
                    logger.info(f"🪞 COPY MIRROR: executed {len(orders)} mirror orders for {tid[:10]}…")
                else:
                    for o in orders:
                        db.add_audit_log("COPY_MIRROR_SIGNAL", audit_ip(),
                                         f"Mirror signal {o['side']} {o['qty']:.5f} {o['symbol']} "
                                         f"(execution {exec_mode} - keys required for auto)")
                    logger.info(f"🪞 COPY MIRROR: {len(orders)} mirror signals (mode {exec_mode}) for {tid[:10]}…")
        except Exception as e:
            logger.warning(f"Copy mirror scheduler error: {e}")



async def copy_trading_refresh_scheduler():
    """LOT 67: keeps the real copy-trading leaderboard fresh (every 6h)."""
    while True:
        await asyncio.sleep(6 * 3600)
        try:
            copy_manager.refresh_real_copytrader_leaderboard()
            copy_manager.refresh_allocation_pnl()
            logger.info(f"✅ Copy Trading leaderboard refreshed: {copy_manager.status}")
        except Exception as e:
            logger.warning(f"Copy Trading refresh failed: {e}")



async def reconciliation_scheduler():
    """
    Audit B7-5 / B11: periodic real-balance/position reconciliation with the
    exchange (REAL mode) and internal consistency check (DEMO). Alerts on gaps.
    """
    while True:
        await asyncio.sleep(300)  # every 5 minutes
        try:
            # P0-6 : marque le jour courant comme jour de paper-trading actif
            _mark_paper_validation_day()
            active_mode = STATE["mode"]
            client = get_ccxt_client() if active_mode == "REAL" else None

            if active_mode == "REAL" and client:
                try:
                    # LOT 7 (PDF Pilier P) : risque de contrepartie — limite de
                    # capital par exchange + signaux de santé (spread/volume)
                    try:
                        _cb = counterparty_risk.check_exchange_balance(
                            "Binance", float(STATE.get("balance_real", 0.0)),
                            STATE.get("current_equity", 0.0))
                        if _cb.get("action") == "block":
                            logger.critical(f"⚠️ CONTREPARTIE: {_cb['message']}")
                            risk_state.enter(RiskStateMachine.CAUTION, "COUNTERPARTY_CAP")
                            STATE["risk_state"] = risk_state.to_dict()
                            try:
                                asyncio.create_task(telegram_bot.send_push_notification(
                                    f"⚠️ *RISQUE CONTREPARTIE*\n{_cb['message']}"))
                            except Exception:
                                pass
                        STATE["counterparty_check"] = _cb
                    except Exception:
                        pass
                    balance = client.fetch_balance()
                    actual_balance = float(balance.get("total", {}).get("USDT", 0.0))
                    internal_balance = float(STATE.get("balance_real", 0.0))
                    ok_bal = reconciler.reconcile_balances(actual_balance, internal_balance)
                    if not ok_bal:
                        logger.error(
                            f"RECONCILIATION GAP: exchange balance {actual_balance:.2f} "
                            f"vs internal {internal_balance:.2f}"
                        )
                        try:
                            await telegram_bot.send_push_notification(
                                f"⚠️ *ÉCART DE RÉCONCILIATION*\n"
                                f"Balance exchange : *${actual_balance:,.2f}*\n"
                                f"Balance interne : ${internal_balance:,.2f}"
                            )
                        except Exception:
                            pass
                        # LOT 7 (PDF Pilier K) : en mode REAL, tout écart non
                        # expliqué -> HALT AUTOMATIQUE (les comptes doivent
                        # TOUJOURS coller ; un écart = problème de ledger ou
                        # d'exécution -> on arrête avant d'empirer).
                        if active_mode == "REAL":
                            risk_state.enter(RiskStateMachine.HALT,
                                             f"RECONCILIATION_BALANCE:{actual_balance:.0f}!={internal_balance:.0f}")
                            STATE["risk_state"] = risk_state.to_dict()
                            try:
                                await telegram_bot.send_push_notification(
                                    "🔴 *HALT RÉCONCILIATION*\nÉcart de balance non expliqué en mode REAL\n→ Trading arrêté. Vérifier le ledger avant /resume.")
                            except Exception:
                                pass
                    else:
                        STATE["balance_real"] = actual_balance
                        logger.info(f"RECONCILIATION OK: balance {actual_balance:.2f} USDT")

                    positions = client.fetch_positions()
                    actual_pos = {}
                    for p in positions:
                        sym = p.get("symbol", "").replace("/", "")
                        qty = float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0.0)
                        if qty:
                            actual_pos[sym] = qty
                            # AUDIT B10-3: liquidation proximity alert (futures)
                            liq = p.get("liquidationPrice") or p.get("info", {}).get("liquidationPrice")
                            mark = p.get("markPrice") or p.get("info", {}).get("markPrice")
                            if liq and mark:
                                try:
                                    dist = abs(float(mark) - float(liq)) / float(mark)
                                    if dist < 0.05:
                                        logger.critical(f"⚠️ LIQUIDATION PROXIMITY {sym}: {dist*100:.1f}% from liq price")
                                        try:
                                            await telegram_bot.send_push_notification(
                                                f"⚠️ *PROXIMITÉ LIQUIDATION*\n{sym} à {dist*100:.1f}% du prix de liquidation !"
                                            )
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                    _pos_ok = reconciler.reconcile_positions(actual_pos, "REAL")
                    # LOT 7 (PDF Pilier K) : HALT auto en REAL sur écart de positions
                    if not _pos_ok and active_mode == "REAL":
                        risk_state.enter(RiskStateMachine.HALT, "RECONCILIATION_POSITIONS")
                        STATE["risk_state"] = risk_state.to_dict()
                        try:
                            await telegram_bot.send_push_notification(
                                "🔴 *HALT RÉCONCILIATION*\nÉcart de positions non expliqué en mode REAL\n→ Trading arrêté. Vérifier avant /resume.")
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"REAL reconciliation failed: {e}")
            else:
                # DEMO internal consistency: equity should match balance + positions value
                try:
                    bal = float(STATE.get("balance_demo", 0.0))
                    pos_value = 0.0
                    for p in db.get_positions():
                        sym = p["symbol"]
                        price = STATE["assets"].get(sym, {}).get("price") or 0.0
                        pos_value += float(p["qty"]) * float(price)
                    computed = bal + pos_value
                    diff_pct = abs(computed - STATE["current_equity"]) / max(STATE["current_equity"], 1.0)
                    if diff_pct > 0.02:
                        logger.warning(f"DEMO internal gap {diff_pct*100:.2f}% (bal {bal:.2f} + pos {pos_value:.2f} vs equity {STATE['current_equity']:.2f})")
                        STATE["current_equity"] = computed
                except Exception as e:
                    logger.debug(f"DEMO reconciliation skip: {e}")
        except Exception as e:
            logger.warning(f"Reconciliation scheduler error: {e}")



async def db_backup_scheduler():
    """LOT 64 (roadmap #3): automatic daily database backup."""
    while True:
        await asyncio.sleep(24 * 3600)
        try:
            backup_path = db.create_backup()
            logger.info(f"✅ Daily database backup created: {backup_path}")
        except Exception as e:
            logger.warning(f"DB backup scheduler error: {e}")

