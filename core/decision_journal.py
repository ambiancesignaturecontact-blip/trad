"""
DECISION JOURNAL — TRADE INTELLIGENCE (LOT 3 du mandat).

Enregistre CHAQUE décision (TRADE ou WAIT/NO_TRADE) dans la table
`decision_journal` (db_manager), avec :
  - contexte : actif, régime, état risque, seuil
  - décision : signal, conviction (niveau), edge net, win rate, raison, détail
  - exécution (complété après fill) : qty, prix, slippage attendu/réel
  - clôture (complété à la sortie) : pnl %, durée, raison de sortie,
    MFE/MAE quand mesurables (sinon NULL — jamais de chiffre inventé)

Principes :
  1. JAMAIS bloquant : toute erreur d'écriture est loggée, jamais levée.
  2. Une ligne = UNE décision ; les mises à jour se font par id (l'entrée
     est créée à la décision, complétée à l'exécution, finalisée à la clôture).
  3. Les NON-DÉCISIONS (WAIT) sont de première classe : raison stable + détail.
  4. DÉMO == RÉAL : aucun flag de mode.
"""
import json
import logging
import time

import pandas as pd

logger = logging.getLogger("InstitutionalTradingBot")

# Colonnes optionnelles portées dans le payload JSON (le reste est en colonnes)
_PAYLOAD_KEYS = ("sources", "uncalibrated", "modifiers", "signal_by_strategy",
                 "meta_label_scale", "entry_id")


def _safe_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def journal_decision(db, decision: str, symbol: str, regime: str,
                     signal: float, conviction: float, level: str,
                     edge_net, win_rate, reason: str, detail: str,
                     threshold: float, risk_state: str, strategy: str = "",
                     qty=None, price=None, slippage_bps_expected=None,
                     payload: dict | None = None) -> int:
    """
    Crée l'entrée de décision. Retourne l'id (pour les mises à jour) ou 0.
    L'appelant mémorise l'id par symbole pour la clôture.
    """
    try:
        entry = {
            "ts": time.time(),
            "decision": str(decision),
            "symbol": str(symbol),
            "regime": str(regime or ""),
            "signal": _safe_float(signal, 0.0),
            "conviction": _safe_float(conviction, 0.0),
            "level": str(level or ""),
            "edge_net": _safe_float(edge_net),
            "win_rate": _safe_float(win_rate),
            "reason": str(reason or ""),
            "detail": str(detail or "")[:200],
            "threshold": _safe_float(threshold),
            "risk_state": str(risk_state or ""),
            "strategy": str(strategy or ""),
            "qty": _safe_float(qty),
            "price": _safe_float(price),
            "slippage_bps_expected": _safe_float(slippage_bps_expected),
            "payload": json.dumps({k: v for k, v in (payload or {}).items()
                                   if k in _PAYLOAD_KEYS}, default=str),
        }
        return int(db.log_decision_entry(entry))
    except Exception as e:
        logger.debug(f"journal_decision failed: {e}")
        return 0


def journal_fill(db, entry_id: int, qty: float, price: float,
                 slippage_bps_real=None) -> None:
    """Complète l'entrée à l'exécution (taille, prix, slippage réel)."""
    if not entry_id:
        return
    try:
        db.update_decision_outcome(entry_id, {
            "qty": _safe_float(qty), "price": _safe_float(price),
            "slippage_bps_real": _safe_float(slippage_bps_real)})
    except Exception as e:
        logger.debug(f"journal_fill failed: {e}")


def journal_close(db, entry_id: int, pnl_pct: float, duration_sec=None,
                  exit_reason: str = "", mfe_pct=None, mae_pct=None) -> None:
    """Finalise l'entrée à la clôture : pnl, durée, raison de sortie, MFE/MAE
    (None si non mesurables — jamais de chiffre inventé)."""
    if not entry_id:
        return
    try:
        outcome = {"pnl_pct": _safe_float(pnl_pct),
                   "duration_sec": _safe_float(duration_sec),
                   "exit_reason": str(exit_reason or "")[:80],
                   "mfe_pct": _safe_float(mfe_pct),
                   "mae_pct": _safe_float(mae_pct)}
        db.update_decision_outcome(entry_id, outcome)
    except Exception as e:
        logger.debug(f"journal_close failed: {e}")


def close_journal_entry(db, state: dict, symbol: str, entry_price: float,
                        exit_price: float, side: str, pnl_pct: float) -> None:
    """
    Finalise l'entrée du journal à la clôture (appelé par record_closed_trade) :
    durée, raison de sortie, MFE/MAE estimés sur les candles réelles (fenêtre
    depuis l'entrée, approximatif — None si non mesurable, jamais inventé).
    La référence par symbole est retirée de l'état. JAMAIS bloquant.
    """
    try:
        dj = (state.get("decision_journal_per_symbol", {}) or {}).pop(symbol, None) or {}
        entry_id = dj.get("id")
        if not entry_id:
            return
        entry_ts = float(dj.get("ts") or 0.0)
        duration = (time.time() - entry_ts) if entry_ts else None
        mfe = mae = None
        try:
            df = db.load_candles(symbol, limit=48)
            if df is not None and not df.empty and entry_ts:
                df = df[df.index >= pd.to_datetime(entry_ts, unit="s")]
            mfe, mae = mfe_mae_from_candles(df, entry_price, exit_price, side)
        except Exception:
            pass
        journal_close(db, int(entry_id), pnl_pct, duration, str(side), mfe, mae)
    except Exception as e:
        logger.debug(f"close_journal_entry failed: {e}")


def mfe_mae_from_candles(df, entry_price: float, exit_price: float,
                         side: str) -> tuple:
    """
    MFE/MAE estimés depuis les candles réelles du cache (high/low entre
    l'entrée et la sortie — approximatif, basé sur la série disponible).
    Retourne (mfe_pct, mae_pct) ou (None, None) si données insuffisantes.
    """
    try:
        if df is None or df.empty or entry_price <= 0:
            return None, None
        direction = 1.0 if side == "SELL" else -1.0  # SELL clôt un long
        high = float(df["high"].max())
        low = float(df["low"].min())
        mfe = (high - entry_price) / entry_price * direction
        mae = (low - entry_price) / entry_price * direction
        return float(mfe * 100.0), float(mae * 100.0)
    except Exception:
        return None, None


def non_trade_analysis(state: dict) -> dict:
    """
    Analyse des NON-DÉCISIONS depuis l'état (no_trade_stats enrichi par
    decide_no_trade — LOT 1/2) : breakdown par catégorie + dernière raison.
    """
    stats = state.get("no_trade_stats", {}) or {}
    return {
        "count": int(stats.get("count", 0)),
        "by_reason": dict(stats.get("reasons", {}) or {}),
        "last_reasons": (state.get("last_no_trade_reasons", []) or [])[-10:],
    }
