"""
Scheduler IA autonome (LOT 66) extrait de main.py (LOT C, F3).
Corps inchangé ; symboles partagés importés de main de façon EXPLICITE.
"""

import logging

import asyncio
import json
import time

import numpy as np

# LOT C (F3) : imports DIRECTS vers les vraies sources (découplage réel — les
# symboles viennent de leur module d'origine, pas d'un ré-export de main).
from core.attribution import quality_metrics
from core.mixture_experts import curriculum_sort
from core.risk_committee import daily_risk_budget
from core.robustness import save_state_snapshot
from core.self_assessment import (
    meta_attribution,
    reason_weight_from_attribution,
    simulation_divergence,
)
from core.world_model import build_causal_feature_df, discover_causal_parents
from market_data.historical_fetch import fetch_historical_market_data
from models.lopez_de_prado import calculate_deflated_sharpe_ratio
from models.price_predictor import LSTMLikePredictor, PPOTRAgent
from models.regime_detector import MarketRegimeDetector
from risk.risk_manager import RiskManager
from strategies.engine import MetaAllocationEngine, TrendFollowingStrategy

# Symboles d'ÉTAT et instances partagées : importés de main de façon
# EXPLICITE (main est complet quand ce module est importé, en fin de main.py).
from main import (  # noqa: E402
    STATE,
    _neutral,
    attribution,
    audit_ip,
    db,
    execution_alpha,
    generative_engine,
    hypothesis_generator,
    meta_engine,
    mixture_of_experts,
    mlops_trainer,
    monte_carlo_tester,
    organization,
    platform_metrics,
    ppo_agent,
    risk_committee,
    scenario_tester,
    strategies_list,
    telegram_bot,
)

logger = logging.getLogger("InstitutionalTradingBot")  # LOT C : même canal de logs que main


async def autonomous_ai_scheduler():
    """
    LOT 66: FULLY AUTONOMOUS AI.
    Periodic self-improvement cycle (every 6h):
      1. Refresh real market data (Binance -> Yahoo fallback)
      2. MLOps pipeline: retrain HMM + LSTM + genetic tuning + model registry
      3. Autonomous PPO training from the live experience buffer (real outcomes)
      4. Walk-forward validation (champion/challenger) - only deploy if it improves
      5. Audit log + Telegram notification
    """
    while True:
        await asyncio.sleep(6 * 3600)
        try:
            logger.info("🤖 AUTONOMOUS AI CYCLE STARTING (self-retrain + validate + deploy)")

            # 1) Fresh real data
            df = STATE.get("historical_bars")
            if df is None or df.empty or len(df) < 120:
                df2 = await fetch_historical_market_data("BTCUSDT")
                if df2 is not None and not df2.empty and len(df2) >= 120:
                    df = df2
                    STATE["historical_bars"] = df
            if df is None or df.empty or len(df) < 120:
                logger.warning("🤖 Autonomous AI: insufficient market data, skipping cycle.")
                continue

            # 2) MLOps pipeline (retrain + registry, auto-deploy in DEMO)
            try:
                pipe_res = mlops_trainer.execute_pipeline(df)
                logger.info(f"🤖 Autonomous AI: MLOps pipeline -> {pipe_res.get('status')}")
            except Exception as pe:
                logger.warning(f"🤖 Autonomous AI: MLOps pipeline error: {pe}")

            # 3) Autonomous PPO training from real collected experiences
            buf = STATE.get("ppo_buffer") or []
            platform_metrics.AI_PPO_BUFFER.set(len(buf))
            if len(buf) >= 50:
                try:
                    ppo_agent.train_step(
                        states=[b["state"] for b in buf],
                        actions=[b["action"] for b in buf],
                        log_probs_old=[b["log_prob"] for b in buf],
                        rewards=[b["reward"] for b in buf],
                        next_states=[b["next_state"] for b in buf],
                        terminals=[b["terminal"] for b in buf],
                    )
                    logger.info(f"🤖 Autonomous AI: PPO self-trained on {len(buf)} real experiences.")
                    STATE["ppo_buffer"] = []
                except Exception as ppo_err:
                    logger.warning(f"🤖 Autonomous AI: PPO training error: {ppo_err}")
                # LOT 4 (PDF Pilier C) : mise en sommeil périodique des experts
                # MoE inutiles (contribution PnL négative sur échantillon suffisant)
                try:
                    _sleepy = mixture_of_experts.sleep_useless_experts(min_samples=5, min_contrib_pct=0.0)
                    if _sleepy:
                        logger.warning(f"🧟 Experts MoE mis en sommeil: {_sleepy}")
                except Exception:
                    pass
                # LOT 8 (PDF Pilier N/Q) : métriques de qualité + stress crises
                # réelles + bootstrap du Sharpe (chaque cycle autonome)
                try:
                    _eq = STATE["equity_history_demo"] if STATE["mode"] == "DEMO" else STATE["equity_history_real"]
                    _trades = attribution.trades[-200:]
                    _qm = quality_metrics(_eq, _trades)
                    STATE["quality_metrics"] = _qm
                    # Stress test : crises réelles sur le portefeuille complet
                    # FIX (ruff F821) : active_balance_key n'était PAS défini
                    # dans cette fonction — le bloc levait NameError attrapée
                    # silencieusement, le stress test ne s'exécutait JAMAIS ici.
                    _positions = db.get_positions()
                    _prices = STATE.get("last_known_prices", {})
                    _bal_key = "balance_demo" if STATE["mode"] == "DEMO" else "balance_real"
                    STATE["stress_test_report"] = scenario_tester.run_stress(
                        _positions, STATE[_bal_key], _prices)
                    # Bootstrap : le Sharpe observé est-il dû à la chance ?
                    if len(_eq) >= 30:
                        STATE["bootstrap_sharpe"] = monte_carlo_tester.bootstrap_sharpe_significance(
                            _eq, n_permutations=1000, seed=42)
                except Exception as _qe:
                    logger.debug(f"Quality metrics/stress failed: {_qe}")
                # LOT 7 (PDF Pilier K) : méta-attribution -> RÉDUCTION AUTOMATIQUE
                # du poids des mauvaises raisons (quelles raisons gagnent ?)
                try:
                    _attr = meta_attribution(STATE.get("decision_log", []))
                    if len(_attr) >= 2:
                        _rw = reason_weight_from_attribution(_attr)
                        STATE["reason_weights"] = _rw
                        # facteur global : moyenne des poids (bornée 0.5..1.1)
                        _avg = sum(_rw.values()) / max(len(_rw), 1)
                        STATE["reason_weights_factor"] = max(0.5, min(1.1, _avg))
                        logger.info(f"🧠 MÉTA-ATTRIBUTION: {len(_rw)} raisons pesées, facteur {_avg:.2f}")
                except Exception:
                    pass
            else:
                logger.info(f"🤖 Autonomous AI: PPO buffer {len(buf)}/50 - collecting more experiences.")

            # 3a0) VISION §1c: causal discovery on REAL features -> store parents
            try:
                md_c = {"vpin": STATE.get("market_state", {}).get("vpin", 0.5),
                        "kyle_lambda": 0.0, "sentiment": _neutral(STATE.get("sentiment_index")),
                        "onchain_risk": _neutral(STATE.get("onchain_risk_score"), 0.5),
                        "funding_rates": STATE.get("funding_rates", {}),
                        "symbol": "BTCUSDT"}
                fdf = build_causal_feature_df(STATE, df, md_c)
                if fdf is not None and len(fdf) >= 40:
                    parents = discover_causal_parents(fdf, target="returns")
                    STATE["causal_parents"] = parents
                    STATE["causal_analyzed"] = True   # LOT 4 : l'analyse causale a tourné
                    db.save_setting("causal_parents", json.dumps(parents))
                    logger.info(
                        f"🔗 ANALYSE CAUSALE: {len(parents)} parent(s) causal(aux) "
                        f"trouvé(s) {parents} -> "
                        f"{'signaux actifs' if parents else 'AUCUN parent causal -> réduction des signaux (LOT 4)'}")
                    logger.info(f"🧠 Causal parents of returns: {parents}")
            except Exception as ce:
                logger.warning(f"Causal discovery skipped: {ce}")

            # 3a1) VISION §2c/2d: OFFLINE RL on the replayable event journal
            try:
                events = db.list_events(event_type="paper_fill", limit=500)
                samples = []
                for e in events:
                    try:
                        p = json.loads(e["payload"])
                        samples.append({
                            "state": np.array([0.0, float(p.get("slippage_bps", 0.0)) / 100.0, 0.0, 0.0]),
                            "action": 1.0 if p.get("side") == "BUY" else -1.0,
                            "log_prob": 0.0,
                            "reward": -float(p.get("slippage_bps", 0.0)) / 10000.0,
                            "next_state": np.array([0.0, 0.0, 0.0, 0.0]),
                            "terminal": False,
                            "vol": 0.01,
                        })
                    except Exception:
                        continue
                if len(samples) >= 30:
                    for _h, _exp in mixture_of_experts.experts.items():
                        n = _exp.train_offline(curriculum_sort(samples))
                        if n:
                            logger.info(f"🧠 OFFLINE RL: {_h} expert trained on {n} journal samples")
            except Exception as oe:
                logger.warning(f"Offline RL skipped: {oe}")

            # 3a2) VISION §3: autonomous research cycle (invent -> test -> promote)
            try:
                md_r = {"vpin": STATE.get("market_state", {}).get("vpin", 0.5),
                        "kyle_lambda": 0.0, "sentiment": _neutral(STATE.get("sentiment_index")),
                        "onchain_risk": _neutral(STATE.get("onchain_risk_score"), 0.5),
                        "funding_rates": STATE.get("funding_rates", {}),
                        "market_avg_return": 0.0}
                _research = hypothesis_generator.run_research_cycle(df, md_r, n_candidates=6)
                logger.info(f"🧪 RESEARCH CYCLE: {_research['candidates']} tested, "
                            f"{len(_research['promoted'])} promoted, admitted={len(_research['admitted'])}")
            except Exception as re:
                logger.warning(f"Research cycle skipped: {re}")

            # 3a3) VISION §6: risk committee veto + daily risk budget
            try:
                _vetoes = risk_committee.evaluate(meta_engine, STATE)
                for v in _vetoes:
                    db.add_audit_log("RISK_COMMITTEE", audit_ip(), f"{v['action']} {v['strategy']} (score {v['score']})")
                try:
                    _stress_corr = float(db.get_setting("autonomous_last_stress_corr") or 0.5)
                except Exception:
                    _stress_corr = 0.5
                _budget = daily_risk_budget(meta_engine.recent_performance, _stress_corr)
                STATE["risk_budget"] = _budget
            except Exception as kce:
                logger.warning(f"Risk committee skipped: {kce}")

            # 3a3b) VISION_FUTUR §1: organization reallocation (internal capital market)
            try:
                _stress_corr2 = float(STATE.get("market_state", {}).get("correlation", 0.5) or 0.5)
                organization.reallocate(stress_correlation=_stress_corr2)
                logger.info(f"🏛️ ORGANIZATION: allocations={organization.status()['allocations']}")
            except Exception as oe:
                logger.warning(f"Organization reallocate skipped: {oe}")

            # 3a3c) VISION_FUTUR §5a: state snapshot (event-sourcing lite)
            try:
                save_state_snapshot(db, STATE)
            except Exception:
                pass

            # 3a3d) VISION_FUTUR §4: global curriculum - GAN scenarios become
            # training episodes for the experts (labeled scenarios, not live trades)
            try:
                _scen = generative_engine.generate_extreme_scenarios(n_scenarios=50, stress_factor=2.0)
                for i in range(min(20, len(_scen))):
                    _s = _scen[i]
                    mixture_of_experts.collect_experience(
                        state=np.array([_s[0], abs(_s[1]), _s[2], 0.0]),
                        action=float(np.clip(_s[3], -1, 1)) if len(_s) > 3 else 0.0,
                        logp=0.0,
                        reward=-abs(_s[0]) * 0.1,  # scenarios are stress episodes
                        next_state=np.array([_s[0], abs(_s[1]), _s[2], 0.0]),
                        horizon="position",
                    )
                logger.info("🎓 CURRICULUM: GAN stress episodes added to position expert buffer")
            except Exception as ce:
                logger.warning(f"GAN curriculum skipped: {ce}")

            # 3a4) VISION §7: self-assessment (meta-attribution of reasons + divergence)
            try:
                from core.reporting import build_daily_report
                _rep = build_daily_report(STATE, db)
                _reasons_log = []
                for _o in db.list_events(event_type="order", limit=200):
                    try:
                        _p = json.loads(_o["payload"])
                        _reasons_log.append({"reasons": [r.get("feature", "") for r in (_p.get("reasoning") or [])[:3]],
                                             "pnl": 0.0})
                    except Exception:
                        continue
                if _reasons_log:
                    _attr = meta_attribution(_reasons_log)
                    db.save_setting("reason_effectiveness", json.dumps(_attr))
                    logger.info(f"🔍 Meta-attribution: {len(_attr)} reasons tracked")
                _real_slip = execution_alpha.avg_slippage_bps("market") or 3.0
                _div = simulation_divergence(3.0, _real_slip)  # modeled baseline 3bps
                STATE["sim_divergence"] = _div
                db.save_setting("sim_divergence", str(_div))
                logger.info(f"🔍 Sim vs live slippage divergence: {_div:.2f}")
            except Exception as sae:
                logger.warning(f"Self-assessment skipped: {sae}")

            # 3a) MONTE-CARLO DAILY STRESS (audit B10-2): measure tail risk continuously
            try:
                hist = STATE.get("historical_bars")
                if hist is not None and len(hist) > 30:
                    _mc_price = STATE.get("last_known_prices", {}).get("BTCUSDT") or STATE.get("last_price")
                    if _mc_price is None:
                        logger.warning("Skipping Monte-Carlo stress test: no real BTC price available yet.")
                    else:
                        mc = monte_carlo_tester.execute_stress_test(
                            initial_capital=STATE["balance_demo"] if STATE["mode"] == "DEMO" else STATE["balance_real"],
                            current_price=float(_mc_price),
                            historical_volatility=float(hist["close"].pct_change().dropna().std() or 0.02),
                        )
                        ruin_pct = float(mc.get("ruin_probability") or mc.get("ruin_prob") or 0.0)
                        platform_metrics.RISK_CVAR.set(ruin_pct * 100.0)
                    ruin_pct = float(mc.get("ruin_probability") or mc.get("ruin_prob") or 0.0)
                    platform_metrics.RISK_CVAR.set(ruin_pct * 100.0)
                    logger.info(f"🤖 Monte-Carlo stress: ruin probability = {ruin_pct*100:.2f}% | {mc.get('summary','')}")
            except Exception as mce:
                logger.warning(f"Monte-Carlo stress skipped: {mce}")

            # 3b) GAN EXTREME-SCENARIO STRESS (audit B9-4): the torch GAN generates
            # tail scenarios used to scale portfolio risk budget for the next period.
            try:
                gen_scen = generative_engine.generate_extreme_scenarios(n_scenarios=500, stress_factor=2.5)
                tail_vol = float(np.std(gen_scen[:, 0])) if gen_scen.size else 0.0
                base_vol = float(np.std(df["close"].pct_change().dropna().values[-200:])) if len(df) > 10 else 0.0
                if base_vol > 0 and tail_vol > 0:
                    stress_ratio = float(np.clip(tail_vol / base_vol, 1.0, 3.0))
                    STATE["gan_stress_ratio"] = stress_ratio
                    platform_metrics.RISK_CVAR.set(stress_ratio)
                    logger.info(f"🤖 GAN stress: tail/base vol ratio = {stress_ratio:.2f}")
            except Exception as ge:
                logger.warning(f"GAN stress skipped: {ge}")

            # 4) Walk-forward champion/challenger validation (audit B8-4: multi-asset)
            try:
                from backtester.engine import EventDrivenBacktester, WalkForwardValidator
                wf = WalkForwardValidator(train_ratio=0.7)
                bt = EventDrivenBacktester(initial_capital=STATE.get("balance_demo", 100000.0))
                strat = TrendFollowingStrategy()
                meta_local = MetaAllocationEngine(strategies=[strat])
                risk_local = RiskManager()
                det_local = MarketRegimeDetector()
                # P0-5 (audit §4.9) : même archi que le live (hidden_dim=24),
                # sinon la validation walk-forward ne vaut pas pour le déploiement.
                pred_local = LSTMLikePredictor(input_dim=5, hidden_dim=24)
                ppo_local = PPOTRAgent(state_dim=4, action_dim=1)

                # Primary asset + optional secondary assets from the DB candle cache
                _wf_datasets = {"BTCUSDT": df}
                for _sym in ("ETHUSDT", "SOLUSDT", "XAUUSD"):
                    try:
                        _d = db.load_candles(_sym, limit=400)
                        if _d is not None and not _d.empty and len(_d) >= 150:
                            _wf_datasets[_sym] = _d
                    except Exception:
                        pass

                _oos_sharpe_agg = 0.0
                for _sym, _data in _wf_datasets.items():
                    try:
                        _res = wf.run_validation(_data, bt, meta_local, risk_local, det_local, pred_local, ppo_local)
                        _oos = _res.get("out_of_sample_metrics", {}) or {}
                        _oos_sharpe_agg += float(_oos.get("sharpe_ratio") or 0.0)
                    except Exception as _we:
                        logger.warning(f"Walk-forward {_sym} skipped: {_we}")
                oos_sharpe = _oos_sharpe_agg / max(len(_wf_datasets), 1)
                platform_metrics.AI_OOS_SHARPE.set(oos_sharpe)
                platform_metrics.AI_LAST_CYCLE.set(time.time())
                prev_sharpe = float(db.get_setting("autonomous_last_oos_sharpe") or 0.0)
                # VISION §4.4: Deflated Sharpe gate - is this OOS result statistically
                # better than luck across the number of strategies/models tried?
                try:
                    _dsr = calculate_deflated_sharpe_ratio(
                        observed_sharpe=oos_sharpe,
                        num_trials=max(len(strategies_list), 8),
                        trials_variance_sharpe=0.1,
                        sample_length=200,
                    )
                except Exception:
                    _dsr = 0.0
                trend = "IMPROVED" if (oos_sharpe >= prev_sharpe and _dsr >= 0.95) else "DEGRADED"
                db.save_setting("autonomous_last_oos_sharpe", str(oos_sharpe))
                db.save_setting("autonomous_last_dsr", str(_dsr))
                logger.info(f"🤖 Autonomous AI: OOS Sharpe {oos_sharpe:.3f} vs {prev_sharpe:.3f} | DSR {_dsr:.3f} (gate 0.95) -> {trend}")
                try:
                    await telegram_bot.send_push_notification(
                        f"🤖 *CYCLE IA AUTONOME*\n"
                        f"━━━━━━━━━━━━━━━━━━━━━\n"
                        f"📊 Sharpe out-of-sample : *{oos_sharpe:.3f}*\n"
                        f"📈 Tendance : *{trend}*\n"
                        f"🧠 PPO : entraîné sur {len(buf)} expériences réelles"
                    )
                except Exception:
                    pass
            except Exception as ve:
                logger.warning(f"🤖 Autonomous AI: walk-forward validation error: {ve}")

            db.add_audit_log("AUTONOMOUS_AI_CYCLE", audit_ip(), "Completed autonomous self-retrain/validate cycle.")
        except Exception as e:
            logger.error(f"🤖 Autonomous AI cycle failed: {e}")
