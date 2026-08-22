"""
SHADOW MODE (PHASE 4 — P4-C, axe 9 du mandat).

Une nouvelle version du système (ou une variante de paramètres) reçoit les
MÊMES données que la production (mêmes ticks, mêmes prix, mêmes signaux,
même conviction) et prend des décisions FICTIVES, jamais exécutées. On
compare ensuite : production vs shadow, mêmes conditions de marché.

    Candidate -> Shadow (décisions fictives) -> Paper -> Limited -> Full

Ce module implémente l'instance shadow par défaut :
  `shadow_vnext` — une version PLUS PERMISSIVE (seuil de conviction × 0,85,
  configurable) : elle teste « est-ce que prendre plus de trades (ceux que la
  production refuse par conviction) aurait été meilleur ? ». C'est LA question
  que posent les données actuelles (12k+ NO_TRADE).

Règles absolues :
  1. FICTIF : aucune décision shadow n'est exécutée ni ne modifie la
     production (STATE de prod non touché hors cache de position shadow).
  2. Jamais bloquant : une erreur shadow = log, le trading continue.
  3. DÉMO == RÉAL : aucun flag de mode.
  4. Le PnL shadow est un mark-to-market virtuel (notionnel = % équité),
     SANS contrainte de min-notional ni frais (fictif, comparatif) — le
     rapport le documente explicitement.
  5. Comparaison honnête : < 10 trades clôturés -> « échantillon
     insuffisant », jamais de conclusion hâtive.
"""
import logging
import time

logger = logging.getLogger("InstitutionalTradingBot")

SHADOW_ID_DEFAULT = "shadow_vnext"
MIN_TRADES_COMPARE = 10


# --------------------------------------------------------------------------- #
# Instance shadow (mini-book virtuel)
# --------------------------------------------------------------------------- #
class ShadowInstance:
    """Une instance shadow : sa propre logique de décision (seuil × scale)
    et son propre mini-book de positions virtuelles."""

    def __init__(self, shadow_id: str = SHADOW_ID_DEFAULT,
                 threshold_scale: float = 0.85,
                 notional_pct: float = 0.02,
                 description: str = ""):
        self.shadow_id = shadow_id
        self.threshold_scale = float(threshold_scale)
        self.notional_pct = float(notional_pct)
        self.description = description or (
            f"shadow v_next : seuil conviction × {threshold_scale} "
            f"(plus permissif), notionnel {notional_pct*100:.0f}% équité/trade")
        self.positions: dict[str, dict] = {}   # symbol -> position virtuelle
        self.decision_count = 0
        self.trade_count = 0

    # -- logique de décision (PURE : testable) --------------------------- #
    def decide(self, signal: float, conviction: float,
               production_threshold: float) -> tuple:
        """(direction, reason) : 0 = WAIT, +1/-1 = TRADE fictif.
        Le shadow utilise SON seuil = production_threshold × threshold_scale
        sur la CONVICTION CALIBRÉE (même règle que la production, seuil
        différent — c'est la variante testée)."""
        thr = production_threshold * self.threshold_scale
        if conviction == 0.0 and signal == 0.0:
            return 0, "no_signal"
        if abs(conviction) >= thr:
            return (1 if signal > 0 else -1), \
                f"shadow_trade (conv {conviction:.4f} >= seuil shadow {thr:.4f})"
        return 0, f"shadow_wait (conv {conviction:.4f} < seuil shadow {thr:.4f})"

    # -- feed d'une barre (mêmes données que la production) -------------- #
    def feed_bar(self, db, symbol: str, price: float, signal: float,
                 conviction: float, production_threshold: float,
                 equity: float) -> dict:
        """Reçoit le tick de production, applique SA logique, gère son
        mini-book. Retourne {decision, reason, trade_closed, trade_opened}.
        Jamais bloquant."""
        out = {"decision": "WAIT", "reason": "", "trade_closed": None,
               "trade_opened": None}
        try:
            direction, reason = self.decide(signal, conviction,
                                            production_threshold)
            self.decision_count += 1
            out["reason"] = reason
            pos = self.positions.get(symbol)

            # clôture si position existante et signal opposé/nul
            if pos is not None:
                pos_dir = 1 if pos["side"] == "BUY" else -1
                if direction == 0 or direction != pos_dir:
                    exit_price = float(price)
                    pnl_pct = (exit_price - pos["entry_price"]) / \
                        pos["entry_price"] * pos_dir
                    duration = time.time() - pos["entry_ts"]
                    if db is not None and hasattr(db, "save_shadow_trade"):
                        try:
                            db.save_shadow_trade(
                                self.shadow_id, pos["entry_ts"], time.time(),
                                symbol, pos["side"], pos["entry_price"],
                                exit_price, pnl_pct, duration)
                        except Exception as e:
                            logger.debug(f"save_shadow_trade failed: {e}")
                    self.trade_count += 1
                    out["trade_closed"] = {"symbol": symbol,
                                           "pnl_pct": round(pnl_pct, 6)}
                    self.positions.pop(symbol, None)

            # ouverture si signal et pas de position
            if direction != 0 and self.positions.get(symbol) is None:
                if equity and equity > 0:
                    notional = equity * self.notional_pct
                    qty = notional / price if price > 0 else 0.0
                    if qty > 0:
                        self.positions[symbol] = {
                            "side": "BUY" if direction > 0 else "SELL",
                            "entry_price": float(price), "qty": qty,
                            "entry_ts": time.time()}
                        out["decision"] = "TRADE"
                        out["trade_opened"] = {"symbol": symbol,
                                               "side": self.positions[symbol]["side"],
                                               "qty": round(qty, 8),
                                               "price": float(price)}
            # journal de décision shadow (persistance best-effort)
            if db is not None and hasattr(db, "save_shadow_decision"):
                try:
                    db.save_shadow_decision(self.shadow_id, time.time(),
                                            symbol, signal, conviction,
                                            out["decision"], price)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"shadow feed_bar failed: {e}")
        return out


# --------------------------------------------------------------------------- #
# Comparaison production vs shadow (honnête)
# --------------------------------------------------------------------------- #
def shadow_compare(db, shadow_id: str = SHADOW_ID_DEFAULT) -> dict:
    """Rapport de comparaison : trades clôturés du shadow (pnl, win rate,
    expectancy, durée) + volume de décisions. Insuffisant < MIN_TRADES."""
    out = {"shadow_id": shadow_id, "n_decisions": 0, "n_trades_closed": 0,
           "n_open_positions": 0, "win_rate": None, "expectancy_pct": None,
           "cumulative_pnl_pct": None, "avg_duration_sec": None,
           "status": "INSUFFICIENT", "note": None}
    try:
        if db is None or not hasattr(db, "load_shadow_trades"):
            out["note"] = "persistance shadow indisponible"
            return out
        trades = db.load_shadow_trades(shadow_id)
        decisions = db.load_shadow_decisions(shadow_id) \
            if hasattr(db, "load_shadow_decisions") else []
        out["n_decisions"] = len(decisions)
        out["n_trades_closed"] = len(trades)
        if trades:
            pnls = [t["pnl_pct"] for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            durations = [t.get("duration_sec") for t in trades
                         if t.get("duration_sec") is not None]
            out["win_rate"] = round(wins / len(pnls), 4)
            out["expectancy_pct"] = round(sum(pnls) / len(pnls) * 100.0, 4)
            out["cumulative_pnl_pct"] = round(sum(pnls) * 100.0, 4)
            if durations:
                out["avg_duration_sec"] = round(
                    sum(durations) / len(durations), 1)
        if len(trades) < MIN_TRADES_COMPARE:
            out["status"] = "INSUFFICIENT"
            out["note"] = (f"{len(trades)} trade(s) clôturé(s) shadow — il en "
                           f"faut ≥ {MIN_TRADES_COMPARE} pour comparer à la "
                           f"production (le shadow accumule les mêmes ticks "
                           f"que la production, sans rien exécuter)")
        else:
            out["status"] = "OK"
            out["note"] = ("comparaison indicative : le shadow est FICTIF "
                           "(pas de min-notional, pas de frais) — à confronter "
                           "à la friction réelle avant toute conclusion")
    except Exception as e:
        out["note"] = f"indisponible ({e})"
    return out
