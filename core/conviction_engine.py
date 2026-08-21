"""
CONVICTION ENGINE + TRADE OPPORTUNITY ENGINE (LOT 2 du mandat).

Objectif : transformer la conviction en une estimation CALIBRÉE et décidable
(et non un score arbitraire), et rendre la décision TRADE / WAIT explicite,
de première classe, instrumentée et réversible.

Principes (les mêmes qui protègent un système en production) :
  1. La base de la conviction reste `calibrated_conviction` (core/risk_pipeline)
     — aucun ré-implémentation : ce module L'ENVELOPPE et n'ajoute que des
     modificateurs DÉFENSIFS bornés (≤ 1.0) quand l'information existe.
  2. JAMAIS d'augmentation artificielle de la conviction pour trader plus.
     Le boost ne vient que du win rate réel (déjà dans calibrated_conviction).
  3. L'edge estimé est NET des coûts : E = p·R − (1−p) − coûts/SL
     (espérance par unité de capital risqué). C'est la base de la décision
     TRADE / WAIT (« aucun trade car l'edge estimé est insuffisant après
     frais + slippage »).
  4. La calibration est MESURÉE : buckets de conviction -> win rate observé,
     expectancy par bucket, calibration error — le système sait si sa
     conviction est bien calibrée (et peut le montrer).
  5. DÉMO == RÉAL : aucun flag de mode dans ce module.
  6. Le moteur ne contourne JAMAIS le Risk Engine : il ne rend qu'un verdict
     TRADE/WAIT + des raisons ; la TAILLE reste décidée par le pipeline de
     risque (RiskManager + apply_risk_pipeline).
"""
import logging
import time

import numpy as np

from core.config import settings
from core.risk_pipeline import (
    CONVICTION_CALIB_MAX,
    REWARD_RISK_RATIO,
    ROUND_TRIP_COST_PCT,
    STOP_LOSS_PCT,
    calibrated_conviction,
)

logger = logging.getLogger("InstitutionalTradingBot")  # même canal que main

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
# Seuil d'entrée minimal (aligné sur trading.signal_threshold)
MIN_ENTRY_THRESHOLD: float = settings.get_float("trading", "signal_threshold", 0.08)
# Seuil au-dessus duquel un signal est « fort » (marge pour HIGH_CONVICTION)
HIGH_CONVICTION_THRESHOLD: float = settings.get_float("conviction", "high_conviction_threshold", 0.25)
# Seuil pour LOW_CONVICTION (entre entrée et normal)
LOW_CONVICTION_THRESHOLD: float = settings.get_float("conviction", "low_conviction_threshold", 0.15)
# Slippage (bps) au-dessus duquel l'exécution est jugée risquée
EXECUTION_RISK_SLIPPAGE_BPS: float = settings.get_float("conviction", "execution_risk_slippage_bps", 50.0)
# Un signal sous ce niveau SANS historique de win rate est jugé « non calibré »
UNCALIBRATED_SIGNAL_MAX: float = settings.get_float("conviction", "uncalibrated_signal_max", 0.12)
# Confiance de régime sous laquelle la conviction est réduite (léger)
LOW_REGIME_CONFIDENCE: float = settings.get_float("conviction", "low_regime_confidence", 0.35)

# Buckets de calibration (borne inf incluse, borne sup exclue)
CALIBRATION_BUCKETS: list[tuple[float, float]] = [
    (0.00, 0.08), (0.08, 0.15), (0.15, 0.25), (0.25, 0.40), (0.40, 1.01),
]

# Niveaux de conviction (mandat LOT 2)
HIGH_CONVICTION = "HIGH_CONVICTION"
NORMAL = "NORMAL"
LOW_CONVICTION = "LOW_CONVICTION"
UNCERTAIN = "UNCERTAIN"
NO_TRADE = "NO_TRADE"

# Raisons WAIT (Trade Opportunity Engine) — codes stables, instrumentés
WAIT_CONVICTION = "conviction"            # |signal| sous le seuil d'entrée
WAIT_EDGE_INSUFFICIENT = "EDGE_INSUFFICIENT"  # edge net <= 0 (frais + slippage)
WAIT_EXECUTION_RISK = "EXECUTION_RISK"    # slippage attendu trop élevé
WAIT_UNCALIBRATED = "UNCALIBRATED"        # setup intéressant mais conviction non calibrée
WAIT_HALT = "halt"                        # machine à états HALT


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def expected_edge_net(win_rate: float | None,
                      reward_risk: float = REWARD_RISK_RATIO,
                      round_trip_cost_pct: float = ROUND_TRIP_COST_PCT,
                      stop_loss_pct: float = STOP_LOSS_PCT) -> float | None:
    """
    Espérance de gain NETTE par unité de capital risqué (fraction) :
        E = p·R − (1−p) − coûts/SL
    où p = win rate, R = reward/risk, coûts = coût aller-retour en fraction
    du notionnel, SL = distance du stop en fraction.

    E > 0  -> l'edge couvre les coûts (trade justifiable)
    E <= 0 -> EDGE INSUFFICIENT (« aucun trade car l'edge estimé est
              insuffisant après frais + slippage »)

    win_rate None (pas d'historique) -> None (on n'invente pas un edge) —
    l'appelant décide via uncalibrated.
    """
    if win_rate is None:
        return None
    p = float(win_rate)
    R = max(float(reward_risk), 0.1)
    sl = max(float(stop_loss_pct), 0.001)
    cost_per_risk = float(round_trip_cost_pct) / sl
    return float(p * R - (1.0 - p) - cost_per_risk)


def conviction_level(conviction: float) -> str:
    """
    Niveau de conviction (mandat LOT 2) :
        NO_TRADE        : conviction < seuil d'entrée
        LOW_CONVICTION  : [seuil, LOW_CONVICTION_THRESHOLD)
        NORMAL          : [LOW_CONVICTION_THRESHOLD, HIGH_CONVICTION_THRESHOLD)
        HIGH_CONVICTION : >= HIGH_CONVICTION_THRESHOLD
    (UNCERTAIN est décidé par le moteur — données insuffisantes.)
    """
    c = float(conviction)
    if c < MIN_ENTRY_THRESHOLD:
        return NO_TRADE
    if c < LOW_CONVICTION_THRESHOLD:
        return LOW_CONVICTION
    if c < HIGH_CONVICTION_THRESHOLD:
        return NORMAL
    return HIGH_CONVICTION


class ConvictionEngine:
    """
    Conviction calibrée + mesure de calibration.

    calibrate()  : enveloppe calibrated_conviction() avec des modificateurs
                   défensifs bornés (régime incertain, exécution dégradée) et
                   produit {conviction, level, edge_net, uncalibrated, reasons}.
    track_outcome(): enregistre (conviction, succès) dans des buckets — la
                   calibration devient MESURABLE (win rate par bucket,
                   expectancy, calibration error).
    calibration_report(): état agrégé pour la télémétrie / l'API.
    """

    def __init__(self, state: dict | None = None):
        self.state = state if state is not None else {}
        self.state.setdefault("conviction_buckets", {})
        self.state.setdefault("conviction_history", [])

    # ------------------------------------------------------------------ #
    def calibrate(self, signal: float,
                  win_rate: float | None = None,
                  has_history: bool = False,
                  regime_confidence: float | None = None,
                  slippage_bps: float | None = None) -> dict:
        """
        Calibre la conviction |signal| :
          1. base = calibrated_conviction(signal, win_rate)  [pas de réimpl.]
          2. modificateurs DÉFENSIFS bornés (jamais de boost artificiel) :
             - régime incertain (< LOW_REGIME_CONFIDENCE)      -> x0.95
             - slippage attendu élevé (> 50 bps)               -> x0.90
          3. edge_net = expected_edge_net(win_rate) (None si pas d'historique)
          4. level = conviction_level(conviction), UNCERTAIN si signal faible
             sans historique de win rate (on ne sait pas calibrer).

        Retourne un dict auto-suffisant (utilisable par TradeOpportunityEngine
        et par la télémétrie).
        """
        base = float(np.clip(calibrated_conviction(signal, win_rate),
                             0.0, CONVICTION_CALIB_MAX))
        reasons: list[str] = []
        conviction = base

        # Modificateurs DÉFENSIFS (≤ 1.0, bornés, réversibles)
        if regime_confidence is not None:
            try:
                rc = _clamp(float(regime_confidence), 0.0, 1.0)
                if rc < LOW_REGIME_CONFIDENCE:
                    conviction *= 0.95
                    reasons.append("régime incertain (x0.95)")
            except (TypeError, ValueError):
                pass
        if slippage_bps is not None:
            try:
                if float(slippage_bps) > EXECUTION_RISK_SLIPPAGE_BPS:
                    conviction *= 0.90
                    reasons.append(f"slippage attendu {float(slippage_bps):.0f} bps (x0.90)")
            except (TypeError, ValueError):
                pass
        conviction = float(_clamp(conviction, 0.0, CONVICTION_CALIB_MAX))

        edge_net = expected_edge_net(win_rate)
        uncalibrated = bool(not has_history) and abs(float(signal)) < UNCALIBRATED_SIGNAL_MAX
        level = conviction_level(conviction)
        if uncalibrated:
            level = UNCERTAIN

        return {
            "conviction": round(conviction, 4),
            "level": level,
            "edge_net": round(edge_net, 6) if edge_net is not None else None,
            "uncalibrated": uncalibrated,
            "win_rate": round(float(win_rate), 4) if win_rate is not None else None,
            "reasons": reasons,
            "base": round(base, 4),
            "ts": time.time(),
        }

    # ------------------------------------------------------------------ #
    def track_outcome(self, conviction: float, success: bool,
                      pnl_pct: float | None = None,
                      strategy: str = "", symbol: str = "") -> None:
        """
        Enregistre l'issue d'un trade dans le bucket de sa conviction
        (calibration mesurée). JAMAIS bloquant.
        """
        try:
            c = _clamp(float(conviction), 0.0, 1.0)
            bucket = self._bucket_of(c)
            entry = self.state["conviction_buckets"].setdefault(bucket, {
                "n": 0, "wins": 0, "pnl_sum": 0.0, "pnl_n": 0})
            entry["n"] += 1
            if success:
                entry["wins"] += 1
            if pnl_pct is not None:
                entry["pnl_sum"] += float(pnl_pct)
                entry["pnl_n"] += 1
            hist = self.state["conviction_history"]
            hist.append({"ts": time.time(), "conviction": c, "success": bool(success),
                         "strategy": strategy, "symbol": symbol})
            if len(hist) > 2000:
                del hist[: len(hist) - 2000]
        except Exception as e:
            logger.warning(f"conviction track_outcome failed ({e})")

    @staticmethod
    def _bucket_of(conviction: float) -> str:
        for lo, hi in CALIBRATION_BUCKETS:
            if lo <= conviction < hi:
                return f"[{lo:.2f},{hi:.2f})"
        return f"[{CALIBRATION_BUCKETS[-1][0]:.2f},{CALIBRATION_BUCKETS[-1][1]:.2f})"

    # ------------------------------------------------------------------ #
    def calibration_report(self) -> dict:
        """
        Calibration MESURÉE :
          - par bucket : n, win rate observé, expectancy moyenne
          - calibration error : moyenne pondérée |win_rate_obs - centre(bucket)|
          - nb de trades trackés, fréquence de trade (par rapport aux ticks ?)
        Sans échantillon -> dict vide (aucun chiffre fabriqué).
        """
        buckets = self.state.get("conviction_buckets", {})
        if not buckets:
            return {"n": 0, "note": "aucun trade tracké — calibration en attente de données"}
        rows = []
        total_n = 0
        err_num = 0.0
        for bucket, b in sorted(buckets.items()):
            n = int(b.get("n", 0))
            if n == 0:
                continue
            wins = int(b.get("wins", 0))
            wr = wins / n
            exp = (float(b.get("pnl_sum", 0.0)) / b.get("pnl_n", 1)
                   if b.get("pnl_n") else None)
            centre = sum(float(x) for x in bucket.strip("[]()").split(",")) / 2.0
            rows.append({"bucket": bucket, "n": n, "win_rate": round(wr, 4),
                         "expectancy_pct": round(exp * 100.0, 4) if exp is not None else None,
                         "centre": round(centre, 4)})
            total_n += n
            err_num += n * abs(wr - centre)
        calib_err = err_num / max(total_n, 1)
        return {
            "n": total_n,
            "buckets": rows,
            "calibration_error": round(calib_err, 4),
            "note": "calibration_error = |win rate observé − centre du bucket| pondéré par n. "
                    "Proche de 0 = conviction bien calibrée.",
        }


class TradeOpportunityEngine:
    """
    Décision explicite TRADE / WAIT (mandat LOT 2, axe 9).

    Évalue l'opportunité AVANT le sizing (la taille reste au Risk Engine) :
        - |signal| sous le seuil d'entrée      -> WAIT conviction
        - machine à états HALT                 -> WAIT halt
        - edge net (après frais + slippage) <= 0 -> WAIT EDGE_INSUFFICIENT
        - slippage attendu > tolérance         -> WAIT EXECUTION_RISK
        - setup non calibré (pas d'historique de win rate) -> WAIT UNCALIBRATED
        - sinon                                -> TRADE

    Chaque WAIT est instrumenté (code stable + détail) — le NO_TRADE devient
    une décision de première classe, explicable.
    """

    def evaluate(self, signal: float, conviction: float,
                 threshold: float = MIN_ENTRY_THRESHOLD,
                 win_rate: float | None = None,
                 edge_net: float | None = None,
                 uncalibrated: bool = False,
                 slippage_bps: float | None = None,
                 risk_state: str | None = None,
                 edge_uncertain: bool = False) -> dict:
        """
        Verdict TRADE/WAIT. L'ordre des vérifications est FIXE (documenté) :
        sécurité d'abord (halt), puis edge, puis exécution, puis calibration.
        Retourne {decision, reason, detail, conviction, edge_net}.
        """
        # Défensif : inputs manquants -> 0.0 (jamais d'exception en boucle live)
        try:
            sig = float(signal) if signal is not None else 0.0
        except (TypeError, ValueError):
            sig = 0.0
        try:
            conv = _clamp(float(conviction), 0.0, CONVICTION_CALIB_MAX) \
                if conviction is not None else 0.0
        except (TypeError, ValueError):
            conv = 0.0

        if risk_state == "HALT":
            return self._wait(WAIT_HALT, "machine à états HALT — aucun nouvel ordre", conv, edge_net)

        if abs(sig) < threshold:
            return self._wait(WAIT_CONVICTION,
                              f"|signal| {abs(sig):.3f} < seuil {threshold:.3f}", conv, edge_net)

        if edge_net is not None and edge_net <= 0.0:
            return self._wait(WAIT_EDGE_INSUFFICIENT,
                              f"edge net {edge_net:.4f} <= 0 après frais + slippage "
                              f"(win rate {win_rate:.3f})", conv, edge_net)

        if slippage_bps is not None and slippage_bps > EXECUTION_RISK_SLIPPAGE_BPS:
            return self._wait(WAIT_EXECUTION_RISK,
                              f"slippage attendu {slippage_bps:.0f} bps > "
                              f"{EXECUTION_RISK_SLIPPAGE_BPS:.0f} bps", conv, edge_net)

        if (uncalibrated or edge_uncertain) and abs(sig) < UNCALIBRATED_SIGNAL_MAX:
            return self._wait(WAIT_UNCALIBRATED,
                              "setup intéressant mais conviction non calibrée "
                              "(pas d'historique de win rate) -> attente", conv, edge_net)

        return {
            "decision": "TRADE",
            "reason": "opportunity_ok",
            "detail": f"edge net {edge_net:.4f} > 0, conviction {conv:.3f} "
                      f"(niveau {conviction_level(conv)})",
            "conviction": conv,
            "edge_net": edge_net,
        }

    @staticmethod
    def _wait(reason: str, detail: str, conviction: float,
              edge_net: float | None) -> dict:
        return {"decision": "WAIT", "reason": reason, "detail": detail,
                "conviction": conviction, "edge_net": edge_net}
