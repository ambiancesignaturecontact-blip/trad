"""
DIGITAL TWIN (PHASE 4 — P4-F, axe 7 du mandat).

« Construire un environnement miroir de QUANT-PORTAL. Il reproduit :
portefeuille, stratégies, risk, exécution, coûts, changements de régime.
On peut alors tester une nouvelle version COMPLÈTE du système avant de la
laisser toucher la production. »

Le Digital Twin = une instance décisionnelle COMPLÈTE et CONFIGURABLE :
  - sa propre logique (seuil conviction × scale — la « nouvelle version ») ;
  - ses propres GATES (friction + exposition factorielle, réutilisant les
    fonctions de production : friction_gate_blocks, exposure_gate_blocks) ;
  - son propre SIZING (notionnel borné, documenté) ;
  - sa propre EXÉCUTION simulée (frais 0,1 %/side + slippage, p95 par
    symbole quand mesuré, sinon défaut) ;
  - son propre book virtuel avec SL/TP (stop_loss_pct, take_profit_rr) et
    durée max de détention.

Deux modes :
  1. LIVE : reçoit les mêmes ticks que la production (comme le shadow) ;
  2. REPLAY : rejoue une « nouvelle version » sur les décisions RÉELLES
     enregistrées dans le decision_journal (prix ≈ close candle 1h —
     approximation documentée : le prix tick n'est pas persisté). Permet de
     tester une variante sur ~6000 décisions réelles IMMÉDIATEMENT.

Règles absolues :
  - FICTIF : rien n'est exécuté, rien ne modifie la production.
  - DÉMO == RÉAL ; jamais bloquant.
  - Le replay est une approximation (prix candle, SL/TP au tick suivant) —
    les résultats servent à CLASSER des variantes, pas à mesurer un edge
    précis (c'est le rôle du paper/limited).
"""
import logging
import time

logger = logging.getLogger("InstitutionalTradingBot")

MIN_TRADES_COMPARE = 10
DEFAULT_PARAMS = {
    "threshold_scale": 0.85,       # seuil twin = seuil production × scale
    "max_exposure_pct": 0.05,      # 5 % équité max par position
    "stop_loss_pct": 0.03,         # 3 % stop loss
    "take_profit_rr": 1.8,         # take profit = RR × stop_loss
    "max_hold_sec": 24 * 3600,     # durée max de détention (24 h)
    "fee_pct": 0.1,                # frais % par side (frais réels mesurés)
    "slippage_bps": 10.0,          # slippage défaut (bps) si non mesuré
    "use_friction_gate": True,     # réplique le gate friction
    "friction_threshold_pct": 1.0,
    "use_exposure_gate": True,     # réplique le gate d'exposition factorielle
    "btc_beta_limit_pct": 50.0,
}


class TwinEngine:
    """Instance miroir complète et configurable (une « nouvelle version »)."""

    def __init__(self, twin_id: str = "twin_vnext", params: dict | None = None):
        self.twin_id = twin_id
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.positions: dict[str, dict] = {}
        self.decision_count = 0
        self.trade_count = 0

    # ------------------------------------------------------------------ #
    def configure(self, params: dict) -> None:
        """Applique une « nouvelle version » (nouveaux paramètres)."""
        self.params.update(params or {})

    # ------------------------------------------------------------------ #
    # Logique de décision (PURE)
    # ------------------------------------------------------------------ #
    def decide(self, signal: float, conviction: float,
               production_threshold: float) -> tuple:
        """(direction, reason) : seuil twin = production × threshold_scale,
        appliqué à la conviction calibrée (même règle que la production)."""
        thr = production_threshold * self.params["threshold_scale"]
        if conviction == 0.0 and signal == 0.0:
            return 0, "no_signal"
        if abs(conviction) >= thr:
            return (1 if signal > 0 else -1), \
                f"twin_trade (conv {conviction:.4f} >= seuil twin {thr:.4f})"
        return 0, f"twin_wait (conv {conviction:.4f} < seuil twin {thr:.4f})"

    # ------------------------------------------------------------------ #
    # Sizing (documenté : notionnel borné proportionnel à la conviction)
    # ------------------------------------------------------------------ #
    def size(self, equity: float, conviction: float,
             production_threshold: float) -> float:
        """Notionnel cible en $ : equity × max_exposure_pct ×
        min(1, |conviction| / (threshold × threshold_scale)). Borné, jamais
        nul si conviction > 0. Sizing SIMPLIFIÉ et documenté (variante
        testée — pas une copie du pipeline Kelly de production)."""
        thr = production_threshold * self.params["threshold_scale"]
        if equity <= 0 or conviction == 0.0:
            return 0.0
        intensity = min(1.0, abs(conviction) / thr) if thr > 0 else 0.0
        return float(equity * self.params["max_exposure_pct"] * intensity)

    # ------------------------------------------------------------------ #
    # Gates répliqués (réutilisent les fonctions de PRODUCTION)
    # ------------------------------------------------------------------ #
    def _gates_block(self, symbol: str, side: str, qty: float, price: float,
                     equity: float, betas: dict, friction_cache: dict,
                     positions: list[dict], correlations: dict) -> tuple:
        """(block, reason) — applique les gates de production avec les
        paramètres du twin (configurables : on peut tester sans gate)."""
        # gate friction
        if self.params.get("use_friction_gate"):
            try:
                from core.execution_intel import friction_gate_blocks
                blk, reason, _ = friction_gate_blocks(
                    symbol, friction_cache,
                    self.params.get("friction_threshold_pct", 1.0))
                if blk:
                    return True, f"friction: {reason}"
            except Exception:
                pass
        # gate exposition factorielle
        if self.params.get("use_exposure_gate"):
            try:
                from core.portfolio_intel import exposure_gate_blocks
                blk, reason, _ = exposure_gate_blocks(
                    symbol, side, qty, price, equity, betas, positions,
                    max_btc_beta_pct=self.params.get("btc_beta_limit_pct", 50.0),
                    correlations=correlations)
                if blk:
                    return True, f"exposure: {reason}"
            except Exception:
                pass
        return False, None

    # ------------------------------------------------------------------ #
    # Exécution simulée (frais + slippage)
    # ------------------------------------------------------------------ #
    def _execution_price(self, price: float, side: str,
                         slippage_bps: float) -> float:
        """Prix d'exécution adverse : slippage en faveur du marché."""
        slip = price * slippage_bps / 10000.0
        return price + slip if side == "BUY" else price - slip

    # ------------------------------------------------------------------ #
    # Feed d'une barre (mêmes données que la production)
    # ------------------------------------------------------------------ #
    def feed_bar(self, db, symbol: str, price: float, signal: float,
                 conviction: float, production_threshold: float,
                 equity: float, betas: dict | None = None,
                 friction_cache: dict | None = None,
                 positions: list[dict] | None = None,
                 correlations: dict | None = None,
                 ts: float | None = None) -> dict:
        """Reçoit un tick réel, applique SA logique complète (décision →
        gates → sizing → exécution simulée → book). Jamais bloquant."""
        out = {"decision": "WAIT", "reason": "", "trade_closed": None,
               "trade_opened": None}
        try:
            ts = ts or time.time()
            betas = betas or {}
            friction_cache = friction_cache or {}
            positions = positions or []
            correlations = correlations or {}
            direction, reason = self.decide(signal, conviction,
                                            production_threshold)
            self.decision_count += 1
            out["reason"] = reason
            pos = self.positions.get(symbol)
            now = ts

            # ---- clôture de la position existante (si applicable) ----- #
            if pos is not None:
                pos_dir = 1 if pos["side"] == "BUY" else -1
                sl = pos["entry_price"] * (1 - self.params["stop_loss_pct"]
                                           * pos_dir)
                tp = pos["entry_price"] * (1 + self.params["take_profit_rr"]
                                           * self.params["stop_loss_pct"]
                                           * pos_dir)
                exit_price = None
                exit_reason = None
                if pos_dir > 0 and price <= sl:
                    exit_price, exit_reason = sl, "SL"
                elif pos_dir > 0 and price >= tp:
                    exit_price, exit_reason = tp, "TP"
                elif pos_dir < 0 and price >= sl:
                    exit_price, exit_reason = sl, "SL"
                elif pos_dir < 0 and price <= tp:
                    exit_price, exit_reason = tp, "TP"
                elif direction == 0 or direction != pos_dir:
                    exit_price, exit_reason = price, "signal_exit"
                elif now - pos["entry_ts"] > self.params["max_hold_sec"]:
                    exit_price, exit_reason = price, "max_hold"
                if exit_price is not None:
                    pnl_pct = (exit_price - pos["entry_price"]) / \
                        pos["entry_price"] * pos_dir
                    # frais 2× + slippage 2× (aller-retour) — coûts réels
                    cost_ar = 2 * self.params["fee_pct"] / 100.0 + \
                        2 * self.params["slippage_bps"] / 10000.0
                    pnl_net = pnl_pct - cost_ar
                    if db is not None and hasattr(db, "save_twin_trade"):
                        try:
                            db.save_twin_trade(
                                self.twin_id, pos["entry_ts"], now, symbol,
                                pos["side"], pos["entry_price"], exit_price,
                                pnl_net, now - pos["entry_ts"], exit_reason)
                        except Exception:
                            pass
                    self.trade_count += 1
                    out["trade_closed"] = {"symbol": symbol,
                                           "pnl_pct": round(pnl_net, 6),
                                           "exit_reason": exit_reason}
                    self.positions.pop(symbol, None)

            # ---- ouverture d'une nouvelle position --------------------- #
            if direction != 0 and self.positions.get(symbol) is None:
                side = "BUY" if direction > 0 else "SELL"
                notional = self.size(equity, conviction, production_threshold)
                qty = notional / price if price > 0 else 0.0
                if qty > 0:
                    blk, blk_reason = self._gates_block(
                        symbol, side, qty, price, equity, betas,
                        friction_cache, positions, correlations)
                    if not blk:
                        fill = self._execution_price(
                            price, side,
                            friction_cache.get(symbol,
                                               self.params["slippage_bps"]))
                        self.positions[symbol] = {
                            "side": side, "entry_price": float(fill),
                            "qty": qty, "entry_ts": now}
                        out["decision"] = "TRADE"
                        out["trade_opened"] = {"symbol": symbol, "side": side,
                                               "qty": round(qty, 8),
                                               "price": float(fill)}
                    else:
                        out["reason"] = f"gate_block: {blk_reason}"
            # journal de décision twin
            if db is not None and hasattr(db, "save_twin_decision"):
                try:
                    db.save_twin_decision(self.twin_id, ts, symbol, signal,
                                          conviction, out["decision"], price)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"twin feed_bar failed: {e}")
        return out

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.positions.clear()
        self.decision_count = 0
        self.trade_count = 0


# --------------------------------------------------------------------------- #
# REPLAY sur les décisions réelles (decision_journal)
# --------------------------------------------------------------------------- #
def _price_from_candles(candles, ts: float) -> float | None:
    """Prix ≈ close de la dernière candle 1h <= ts (approximation documentée :
    le prix tick n'est pas persisté dans le journal)."""
    try:
        idx = candles.index
        past = idx[idx <= ts]
        if len(past) == 0:
            return None
        return float(candles.loc[past[-1], "close"])
    except Exception:
        return None


def replay_journal(db, twin: TwinEngine, limit: int = 6000) -> dict:
    """Rejoue la « nouvelle version » (twin) sur les décisions RÉELLES du
    decision_journal, triées par ts. Prix ≈ close candle 1h (proxy). Les
    SL/TP sont checkés au tick suivant (approximation). Résultats :
    trades clôturés (net de frais/slippage), win rate, expectancy, cumul —
    à utiliser pour CLASSER des variantes, pas pour mesurer un edge précis."""
    out = {"twin_id": twin.twin_id, "n_decisions": 0, "n_trades_closed": 0,
           "win_rate": None, "expectancy_pct": None,
           "cumulative_pnl_pct": None, "n_open": 0, "status": "INSUFFICIENT",
           "note": None}
    try:
        if db is None or not hasattr(db, "get_connection"):
            out["note"] = "db indisponible"
            return out
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT ts, symbol, signal, conviction, threshold, "
                "risk_state FROM decision_journal ORDER BY ts ASC "
                "LIMIT ?", (int(limit),))
            rows = cur.fetchall()
        if not rows:
            out["note"] = "aucune décision à rejouer"
            return out
        # cache des candles par symbole (close par timestamp)
        candle_cache = {}
        prices = []
        for r in rows:
            ts, symbol = float(r[0]), str(r[1])
            signal, conv = float(r[2] or 0.0), float(r[3] or 0.0)
            thr = float(r[4] or 0.15)
            if symbol not in candle_cache:
                try:
                    candle_cache[symbol] = db.load_candles(symbol, limit=2000)
                except Exception:
                    candle_cache[symbol] = None
            df = candle_cache.get(symbol)
            price = _price_from_candles(df, ts) if df is not None else None
            prices.append(price)
            twin.feed_bar(db, symbol, price if price else 100.0, signal,
                          conv, thr, equity=100000.0, ts=ts)
        trades = []
        if hasattr(db, "load_twin_trades"):
            trades = db.load_twin_trades(twin.twin_id)
        out["n_decisions"] = len(rows)
        out["n_trades_closed"] = len(trades)
        out["n_open"] = len(twin.positions)
        if trades:
            pnls = [t["pnl_pct"] for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            out["win_rate"] = round(wins / len(pnls), 4)
            out["expectancy_pct"] = round(sum(pnls) / len(pnls) * 100.0, 4)
            out["cumulative_pnl_pct"] = round(sum(pnls) * 100.0, 4)
        if len(trades) < MIN_TRADES_COMPARE:
            out["status"] = "INSUFFICIENT"
            out["note"] = (f"{len(trades)} trades clôturés twin (replay "
                           f"{out['n_decisions']} décisions) — il en faut ≥ "
                           f"{MIN_TRADES_COMPARE} pour comparer. Le replay "
                           f"classe des VARIANTES, pas un edge précis "
                           f"(prix ≈ candle, SL/TP au tick suivant).")
        else:
            out["status"] = "OK"
            out["note"] = ("replay sur décisions RÉELLES ; approximation "
                           "prix candle 1h + SL/TP au tick suivant ; PnL net "
                           "de frais 0,1 %/side et slippage p95 par symbole "
                           "quand mesuré")
    except Exception as e:
        out["note"] = f"indisponible ({e})"
        logger.debug(f"replay_journal failed: {e}")
    return out
